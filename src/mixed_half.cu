// Mixed-precision SPIKE solver: a double matrix factorized in CUDA __half and
// corrected with four double-precision residual iterations.
//
// Parallel to mixed.cu: the float-based mixed_solver is left untouched.

#include "common.cuh"
#include "gspike/spike_half.h"
#include "kernels_half.cuh"

#include <algorithm>
#include <climits>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <vector>

namespace gspike {

    // Half-precision factors need more refinement than float factors: each
    // pass with a double residual shrinks the error by roughly cond * u_half.
    constexpr int kMixedHalfIterations = 4;

    struct mixed_half_solver::impl {
        int n = 0, kl = 0, ku = 0, lda = 0, block = detail::kBlock;
        const double* A = nullptr;
        detail::device_buffer<__half> d_af;   // half copy of A
        detail::device_buffer<__half> d_work; // half right-hand side / correction
        detail::device_buffer<double> d_b;    // original right-hand side
        detail::device_buffer<double> d_r;    // solution, residual and correction
        detail::device_buffer<double> d_max;  // per-column max |residual| for scaling
        std::unique_ptr<solver<__half>> fs;

        void resize(const int nrhs) {
            const size_t need = (size_t)n * nrhs;
            if(need <= d_work.size()) return;
            d_work.alloc(need);
            d_b.alloc(need);
            d_r.alloc(need);
            d_max.alloc(nrhs);
        }
    };

    mixed_half_solver::mixed_half_solver(int n, int kl, int ku, const double* A, const int lda, const opts& o)
        : p_(std::make_unique<impl>()) {
        impl* const q = p_.get();
        if(o.block >= 32 && o.block <= 1024) q->block = o.block;
        if(n < 1 || kl < 0 || ku < 0 || !A) throw std::invalid_argument("gspike: bad dimensions");
        if(lda < kl + ku + 1) throw std::invalid_argument("gspike: lda < kl+ku+1");

        q->n = n;
        q->kl = kl;
        q->ku = ku;
        q->lda = lda;
        q->A = A;

        const int lda_f = kl + ku + 1;
        q->d_af.alloc((size_t)lda_f * n);
        dk::convert_mat<__half, double><<<n, q->block>>>(lda_f, A, lda, q->d_af.get(), lda_f);
        detail::check(cudaGetLastError());
        q->fs = std::make_unique<solver<__half>>(n, kl, ku, q->d_af.get(), lda_f, o);
    }

    mixed_half_solver::~mixed_half_solver() = default;

    int mixed_half_solver::solve(const int nrhs, double* B, const int ldb, stats* st) const {
        impl* const q = p_.get();
        if(nrhs < 0) return -1;
        if(!B) return -2;
        if(ldb < q->n) return -3;
        if(st) st->parts = q->fs->parts();
        if(nrhs == 0) return 0;
        if((long long)q->n * nrhs > INT_MAX) return -4;

        q->resize(nrhs);
        const int n = q->n, block = q->block;
        const dim3 gcol(nrhs), grow(nrhs, (n + block - 1) / block);

        dk::copy_mat<double><<<gcol, block>>>(n, B, ldb, q->d_b.get(), n);
        dk::convert_mat<__half, double><<<gcol, block>>>(n, q->d_b.get(), n, q->d_work.get(), n);
        detail::check(cudaGetLastError());

        int info = q->fs->solve(nrhs, q->d_work.get(), n);
        // The first half solve is the initial approximation; keep the
        // accumulated solution in double precision across refinements.
        dk::convert_mat<double, __half><<<gcol, block>>>(n, q->d_work.get(), n, B, ldb);

        for(int it = 0; it < kMixedHalfIterations; ++it) {
            // r = b - A*x with x and the residual both in double precision.
            dk::band_residual<double><<<grow, block>>>(n, q->kl, q->ku, q->A, q->lda, B, ldb, q->d_b.get(), n, q->d_r.get(), n);
            // Bring the residual into the normal range of half before solving:
            // half subnormals below ~6e-8 carry almost no significant bits, which
            // otherwise stalls refinement around 1e-7.
            dk::max_abs_double<<<nrhs, 256>>>(n, q->d_r.get(), n, q->d_max.get());
            std::vector<double> hmax((size_t)nrhs);
            detail::check(cudaMemcpy(hmax.data(), q->d_max.get(), (size_t)nrhs * sizeof(double), cudaMemcpyDeviceToHost));
            double mx = 0;
            for(const double v : hmax) mx = std::max(mx, v);
            const double scale = mx > 0 ? 0.5 / mx : 1.0;
            dk::convert_scaled_mat<__half, double><<<gcol, block>>>(n, q->d_r.get(), n, q->d_work.get(), n, scale);
            const int step_info = q->fs->solve(nrhs, q->d_work.get(), n);
            if(step_info && !info) info = step_info;
            // Undo the scaling in double, then accumulate on the solution.
            dk::convert_scaled_mat<double, __half><<<gcol, block>>>(n, q->d_work.get(), n, q->d_r.get(), n, 1.0 / scale);
            dk::add_mat<double><<<gcol, block>>>(n, q->d_r.get(), n, B, ldb, 1.0);
            detail::check(cudaGetLastError());
        }

        return info;
    }

    int mixed_half_solver::n() const { return p_->n; }
    int mixed_half_solver::kl() const { return p_->kl; }
    int mixed_half_solver::ku() const { return p_->ku; }
    int mixed_half_solver::parts() const { return p_->fs->parts(); }
    double mixed_half_solver::det() const { return (double)p_->fs->det(); }
    int mixed_half_solver::det_sign() const { return p_->fs->det_sign(); }

} // namespace gspike
