// Mixed-precision SPIKE solver: a double matrix factorized in float and
// corrected with a double-precision residual.

#include "common.cuh"
#include "gspike/spike.h"
#include "kernels_float_double.cuh"

#include <climits>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>

namespace gspike {

    ////////////////////////////////////////////////////////////////////////////////

    struct mixed_solver::impl {
        int n = 0, kl = 0, ku = 0, lda = 0, block = detail::kBlock;
        const double* A = nullptr;
        detail::device_buffer<float> d_af;   // float copy of A
        detail::device_buffer<float> d_work; // float right-hand side / correction
        detail::device_buffer<double> d_b;   // original right-hand side
        detail::device_buffer<double> d_r;   // solution, residual and correction
        std::unique_ptr<solver<float>> fs;

        void resize(const int nrhs) {
            const size_t need = (size_t)n * nrhs;
            if(need <= d_work.size()) return;
            d_work.alloc(need);
            d_b.alloc(need);
            d_r.alloc(need);
        }
    };

    ////////////////////////////////////////////////////////////////////////////////

    mixed_solver::mixed_solver(int n, int kl, int ku, const double* A, const int lda, const opts& o)
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
        dk::convert_mat<float, double><<<n, q->block>>>(lda_f, A, lda, q->d_af.get(), lda_f);
        detail::check(cudaGetLastError());
        q->fs = std::make_unique<solver<float>>(n, kl, ku, q->d_af.get(), lda_f, o);
    }

    mixed_solver::~mixed_solver() = default;

    int mixed_solver::solve(const int nrhs, double* B, const int ldb, stats* st) const {
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
        dk::convert_mat<float, double><<<gcol, block>>>(n, q->d_b.get(), n, q->d_work.get(), n);
        detail::check(cudaGetLastError());

        const int info = q->fs->solve(nrhs, q->d_work.get(), n);

        // x = float solve, then one correction from the double residual.
        dk::convert_mat<double, float><<<gcol, block>>>(n, q->d_work.get(), n, B, ldb);
        dk::band_residual<double><<<grow, block>>>(n, q->kl, q->ku, q->A, q->lda, B, ldb, q->d_b.get(), n, q->d_r.get(), n);
        dk::convert_mat<float, double><<<gcol, block>>>(n, q->d_r.get(), n, q->d_work.get(), n);
        detail::check(cudaGetLastError());

        const int info2 = q->fs->solve(nrhs, q->d_work.get(), n);

        dk::convert_mat<double, float><<<gcol, block>>>(n, q->d_work.get(), n, q->d_r.get(), n);
        dk::add_mat<double><<<gcol, block>>>(n, q->d_r.get(), n, B, ldb, 1.0);
        detail::check(cudaGetLastError());

        return info ? info : info2;
    }

    int mixed_solver::n() const { return p_->n; }
    int mixed_solver::kl() const { return p_->kl; }
    int mixed_solver::ku() const { return p_->ku; }
    int mixed_solver::parts() const { return p_->fs->parts(); }
    double mixed_solver::det() const { return (double)p_->fs->det(); }
    int mixed_solver::det_sign() const { return p_->fs->det_sign(); }

} // namespace gspike
