// Mixed-precision SPIKE solver: a double matrix factorized in float and
// corrected with a double-precision residual.
#include "gspike/spike.h"
#include "kernels.cuh"

#include <algorithm>
#include <climits>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace gspike {
    namespace {

        constexpr int kBlock = 256;

        inline void ck(cudaError_t e) {
            if(e != cudaSuccess)
                throw std::runtime_error(std::string("gspike: ") + cudaGetErrorString(e));
        }

    } // namespace

    ////////////////////////////////////////////////////////////////////////////////

    struct mixed_solver::impl {
        int n = 0, kl = 0, ku = 0, lda = 0;
        const double* A = nullptr;
        float* d_Af = nullptr;
        float* d_work = nullptr; // float right-hand side / correction
        double* d_b = nullptr;   // original right-hand side
        double* d_r = nullptr;   // solution, residual and correction
        size_t cap = 0;
        std::unique_ptr<solver<float>> fs;

        ~impl() {
            if(d_Af) cudaFree(d_Af);
            if(d_work) cudaFree(d_work);
            if(d_b) cudaFree(d_b);
            if(d_r) cudaFree(d_r);
        }

        void alloc(int nrhs) {
            const size_t need = (size_t)n * nrhs;
            if(need <= cap) return;
            if(d_work) cudaFree(d_work);
            if(d_b) cudaFree(d_b);
            if(d_r) cudaFree(d_r);
            d_work = nullptr;
            d_b = nullptr;
            d_r = nullptr;
            cap = need;
            ck(cudaMalloc(&d_work, std::max<size_t>(need, 1) * sizeof(float)));
            ck(cudaMalloc(&d_b, std::max<size_t>(need, 1) * sizeof(double)));
            ck(cudaMalloc(&d_r, std::max<size_t>(need, 1) * sizeof(double)));
        }
    };

    ////////////////////////////////////////////////////////////////////////////////

    mixed_solver::mixed_solver(int n, int kl, int ku, const double* A, int lda, const opts& o)
        : p_(std::make_unique<impl>()) {
        impl* q = p_.get();
        if(n < 1 || kl < 0 || ku < 0 || !A) throw std::invalid_argument("gspike: bad dimensions");
        if(lda < kl + ku + 1) throw std::invalid_argument("gspike: lda < kl+ku+1");

        q->n = n;
        q->kl = kl;
        q->ku = ku;
        q->lda = lda;
        q->A = A;

        const int lda_f = kl + ku + 1;
        ck(cudaMalloc(&q->d_Af, (size_t)lda_f * n * sizeof(float)));
        dk::convert_mat<float, double><<<n, kBlock>>>(lda_f, n, A, lda, q->d_Af, lda_f);
        ck(cudaGetLastError());
        q->fs = std::make_unique<solver<float>>(n, kl, ku, q->d_Af, lda_f, o);
    }

    mixed_solver::~mixed_solver() = default;

    int mixed_solver::solve(int nrhs, double* B, int ldb, stats* st) const {
        impl* q = p_.get();
        if(nrhs < 0) return -1;
        if(!B) return -2;
        if(ldb < q->n) return -3;
        if(nrhs == 0) {
            if(st) st->parts = q->fs->parts();
            return 0;
        }
        if((long long)q->n * nrhs > INT_MAX) return -4;

        q->alloc(nrhs);
        const int n = q->n, block = kBlock;
        const dim3 gcol(nrhs), grow(nrhs, (n + block - 1) / block);

        dk::copy_mat<double><<<gcol, block>>>(n, nrhs, B, ldb, q->d_b, n);
        dk::convert_mat<float, double><<<gcol, block>>>(n, nrhs, q->d_b, n, q->d_work, n);
        ck(cudaGetLastError());

        int info = q->fs->solve(nrhs, q->d_work, n);

        // x = float solve, then one correction from the double residual.
        dk::convert_mat<double, float><<<gcol, block>>>(n, nrhs, q->d_work, n, B, ldb);
        dk::band_residual<double><<<grow, block>>>(n, q->kl, q->ku, q->A, q->lda, B, ldb, q->d_b, n, q->d_r, n);
        dk::convert_mat<float, double><<<gcol, block>>>(n, nrhs, q->d_r, n, q->d_work, n);
        ck(cudaGetLastError());

        const int info2 = q->fs->solve(nrhs, q->d_work, n);

        dk::convert_mat<double, float><<<gcol, block>>>(n, nrhs, q->d_work, n, q->d_r, n);
        dk::add_mat<double><<<gcol, block>>>(n, nrhs, q->d_r, n, B, ldb, 1.0);
        ck(cudaGetLastError());

        if(st) st->parts = q->fs->parts();
        return info ? info : info2;
    }

    int mixed_solver::n() const { return p_->n; }
    int mixed_solver::kl() const { return p_->kl; }
    int mixed_solver::ku() const { return p_->ku; }
    int mixed_solver::parts() const { return p_->fs->parts(); }

} // namespace gspike
