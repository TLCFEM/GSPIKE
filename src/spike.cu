// SPIKE (DS) banded solver -- CUDA implementation.
//
// Single level SPIKE factorization:
//   1. split the band matrix into p contiguous diagonal blocks, each factored
//      with a band LU with partial pivoting (parallel over partitions),
//   2. form the interface spikes  Vp_i = A_ii^-1 C_i ,  Vm_i = A_ii^-1 B_i,
//   3. eliminate the interior and solve the 2*klu*(p-1) interface system
//      (block tridiagonal, factored once with the same band LU),
//   4. retrieve each block solution with a correction on the coupling rows,
//   5. optional iterative refinement against the original matrix.
#include "gspike/spike.h"
#include "kernels.cuh"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace gspike {
    namespace {

        constexpr int kMaxParts = 32;
        constexpr int kInfoNone = 0x7f7f7f7f;

        inline void ck(cudaError_t e) {
            if(e != cudaSuccess)
                throw std::runtime_error(std::string("gspike: ") + cudaGetErrorString(e));
        }
        inline void ckb(cublasStatus_t s) {
            if(s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("gspike: cuBLAS failure");
        }

        template<class T>
        struct num;
        template<>
        struct num<float> {
            static constexpr double rtol0 = 1e-5; // default refinement target
            static cublasStatus_t nrm2(cublasHandle_t h, int n, const float* x, float* r) {
                return cublasSnrm2(h, n, x, 1, r);
            }
        };
        template<>
        struct num<double> {
            static constexpr double rtol0 = 1e-12;
            static cublasStatus_t nrm2(cublasHandle_t h, int n, const double* x, double* r) {
                return cublasDnrm2(h, n, x, 1, r);
            }
        };

    } // namespace

    ////////////////////////////////////////////////////////////////////////////////

    template<class T>
    struct solver<T>::impl {
        int n = 0, kl = 0, ku = 0, klu = 0, p = 1, lda_f = 0, block = 256;
        opts o;
        const T* A = nullptr;
        int lda = 0;
        int info0 = 0;

        cublasHandle_t h = nullptr;
        dk::pdesc<T>* d_pd = nullptr;
        dk::pdesc<T>* d_rp = nullptr;
        T** d_spp = nullptr;
        T* d_ab = nullptr;
        int* d_piv = nullptr;
        T* d_sp = nullptr;
        T* d_bc = nullptr;
        T* d_cc = nullptr;
        T* d_R = nullptr;
        int* d_pivR = nullptr;
        int* d_info = nullptr;
        int m = 0, klr = 0, ldaR = 0;

        T* arena = nullptr;
        size_t arena_n = 0;
        int arena_nrhs = 0;

        ~impl() {
            if(d_pd) cudaFree(d_pd);
            if(d_rp) cudaFree(d_rp);
            if(d_spp) cudaFree(d_spp);
            if(d_ab) cudaFree(d_ab);
            if(d_piv) cudaFree(d_piv);
            if(d_sp) cudaFree(d_sp);
            if(d_bc) cudaFree(d_bc);
            if(d_cc) cudaFree(d_cc);
            if(d_R) cudaFree(d_R);
            if(d_pivR) cudaFree(d_pivR);
            if(d_info) cudaFree(d_info);
            if(arena) cudaFree(arena);
            if(h) cublasDestroy(h);
        }

        void factor();

        void alloc_arena(int nrhs) {
            const size_t need = (size_t)(4LL * n * nrhs + (long long)2 * klu * p * nrhs +
                                         (long long)m * nrhs);
            if(need <= arena_n && nrhs <= arena_nrhs) return;
            if(arena) {
                ck(cudaFree(arena));
                arena = nullptr;
            }
            arena_n = need;
            arena_nrhs = nrhs;
            ck(cudaMalloc(&arena, std::max<size_t>(need, 1) * sizeof(T)));
        }

        // X (n x nrhs, ld = n) holds f on entry and the solution on exit.
        void solve_inplace(T* X, T* g, T* grj, T* rs, int nrhs) const {
            const dim3 gcol(1, (nrhs + block - 1) / block);
            const dim3 gpar(p, (nrhs + block - 1) / block);
            const dim3 gred(p > 1 ? p - 1 : 1, (nrhs + block - 1) / block);

            if(p == 1) {
                dk::band_solve<T><<<gcol, block>>>(d_pd, lda_f, kl, ku, X, n, nrhs);
                ck(cudaGetLastError());
                return;
            }
            dk::copy_mat<T><<<nrhs, block>>>(n, nrhs, X, n, g, n);
            dk::band_solve<T><<<gpar, block>>>(d_pd, lda_f, kl, ku, g, n, nrhs);
            dk::extract_corners<T><<<gpar, block>>>(d_pd, g, n, klu, nrhs, grj);
            dk::reduced_rhs<T><<<gred, block>>>(grj, m, klu, rs, nrhs);
            dk::band_solve<T><<<gcol, block>>>(d_rp, ldaR, klr, klr, rs, m, nrhs);
            dk::correct_rhs<T><<<gpar, block>>>(d_pd, d_bc, d_cc, rs, klu, m, p, X, n, nrhs);
            dk::band_solve<T><<<gpar, block>>>(d_pd, lda_f, kl, ku, X, n, nrhs);
            ck(cudaGetLastError());
        }
    };

    ////////////////////////////////////////////////////////////////////////////////

    template<class T>
    void solver<T>::impl::factor() {
        impl* q = this;
        const int n = q->n, kl = q->kl, ku = q->ku, klu = q->klu, p = q->p, lda_f = q->lda_f;
        const int block = q->block;

        std::vector<int> row0(p), size(p);
        for(int i = 0; i < p; ++i) {
            const int base = n / p, rem = n % p;
            size[i] = base + (i < rem ? 1 : 0);
            row0[i] = (i == 0) ? 0 : row0[i - 1] + size[i - 1];
        }

        ck(cudaMalloc(&q->d_ab, (size_t)lda_f * n * sizeof(T)));
        ck(cudaMalloc(&q->d_piv, (size_t)n * sizeof(int)));
        ck(cudaMalloc(&q->d_pd, (size_t)p * sizeof(dk::pdesc<T>)));
        ck(cudaMalloc(&q->d_sp, (size_t)p * 4 * klu * klu * sizeof(T)));
        ck(cudaMalloc(&q->d_bc, (size_t)p * klu * klu * sizeof(T)));
        ck(cudaMalloc(&q->d_cc, (size_t)p * klu * klu * sizeof(T)));
        ck(cudaMalloc(&q->d_spp, (size_t)p * sizeof(T*)));
        ck(cudaMalloc(&q->d_info, sizeof(int)));
        ck(cudaMemset(q->d_info, 0x7f, sizeof(int)));
        ck(cudaMemset(q->d_sp, 0, (size_t)p * 4 * klu * klu * sizeof(T)));

        std::vector<dk::pdesc<T>> hd(p);
        std::vector<T*> hspp(p);
        for(int i = 0; i < p; ++i) {
            hd[i].ab = q->d_ab + (long long)lda_f * row0[i];
            hd[i].piv = q->d_piv + row0[i];
            hd[i].row0 = row0[i];
            hd[i].size = size[i];
            hspp[i] = q->d_sp + (long long)i * 4 * klu * klu;
        }
        ck(cudaMemcpy(q->d_pd, hd.data(), p * sizeof(dk::pdesc<T>), cudaMemcpyHostToDevice));
        ck(cudaMemcpy(q->d_spp, hspp.data(), p * sizeof(T*), cudaMemcpyHostToDevice));

        dk::make_blocks<T><<<p, block>>>(q->d_pd, q->A, n, q->lda, lda_f, ku, kl, klu, q->d_bc, q->d_cc);
        dk::band_lu<T><<<p, block>>>(q->d_pd, lda_f, kl, ku, q->d_info);
        ck(cudaGetLastError());

        if(p == 1) return;

        // spikes: A_ii [C_i | B_i], solved in chunks of columns (2*klu in total).
        // All spike columns are solved at once (one thread each) as long as the
        // temporary buffer stays within budget.
        const int ncol = 2 * klu;
        int chunk = ncol;
        while(chunk > 1 && (long long)n * chunk * (long long)sizeof(T) > (512LL << 20)) chunk /= 2;
        T* rhs = nullptr;
        ck(cudaMalloc(&rhs, (size_t)n * chunk * sizeof(T)));
        const int nb = (2 * klu * klu + block - 1) / block;
        for(int c0 = 0; c0 < ncol; c0 += chunk) {
            const int nc = std::min(chunk, ncol - c0);
            const int nt = std::min(block, ((nc + 31) / 32) * 32);
            dk::spike_rhs<T><<<dim3(p, nc), block>>>(q->d_pd, q->d_bc, q->d_cc, klu, rhs, n, c0, ncol);
            dk::band_solve<T><<<dim3(p, 1), nt>>>(q->d_pd, lda_f, kl, ku, rhs, n, nc);
            dk::spike_extract<T><<<dim3(p, nb), block>>>(q->d_pd, rhs, n, klu, c0, nc, q->d_sp);
            ck(cudaGetLastError());
        }
        ck(cudaFree(rhs));
    }

    ////////////////////////////////////////////////////////////////////////////////

    template<class T>
    solver<T>::solver(int n, int kl, int ku, const T* A, int lda, const opts& o)
        : p_(new impl) {
        impl* q = p_;
        q->o = o;
        if(o.block >= 32 && o.block <= 1024) q->block = o.block;
        if(n < 1 || kl < 0 || ku < 0 || !A) throw std::invalid_argument("gspike: bad dimensions");
        if(lda < kl + ku + 1) throw std::invalid_argument("gspike: lda < kl+ku+1");

        q->n = n;
        q->kl = kl;
        q->ku = ku;
        q->klu = std::max(kl, ku);
        q->lda_f = 2 * kl + ku + 1;
        q->A = A;
        q->lda = lda;
        ckb(cublasCreate(&q->h));

        const int pmax = (q->klu == 0) ? 1 : std::max(1, n / (2 * q->klu + 1));
        int p = o.parts;
        if(p <= 0) {
            p = (q->klu == 0) ? 1 : std::max(1, n / (16 * q->klu + 32));
            p = std::min(p, kMaxParts);
        }
        p = std::min(std::max(p, 1), pmax);
        while(p > 1) { // keep the interface system within a memory budget
            const long long m = 2LL * q->klu * (p - 1);
            const long long ldaR = 9LL * q->klu + 1;
            if(m * ldaR * (long long)sizeof(T) < (256LL << 20)) break;
            --p;
        }
        q->p = p;
        q->m = 2 * q->klu * (p - 1);
        q->klr = 3 * q->klu;
        q->ldaR = 3 * q->klr + 1;

        q->factor();
        ck(cudaDeviceSynchronize());

        int info = 0;
        ck(cudaMemcpy(&info, q->d_info, sizeof(int), cudaMemcpyDeviceToHost));
        q->info0 = (info == kInfoNone) ? 0 : info;

        if(p > 1) {
            ck(cudaMalloc(&q->d_R, (size_t)q->ldaR * q->m * sizeof(T)));
            ck(cudaMalloc(&q->d_pivR, (size_t)q->m * sizeof(int)));
            ck(cudaMalloc(&q->d_rp, sizeof(dk::pdesc<T>)));
            ck(cudaMemset(q->d_R, 0, (size_t)q->ldaR * q->m * sizeof(T)));

            dk::reduced_diag<T><<<p - 1, q->block>>>(q->d_spp, p, q->klu, q->klr, q->ldaR, q->d_R);
            ck(cudaGetLastError());

            dk::pdesc<T> rp{};
            rp.ab = q->d_R;
            rp.piv = q->d_pivR;
            rp.row0 = 0;
            rp.size = q->m;
            ck(cudaMemcpy(q->d_rp, &rp, sizeof(rp), cudaMemcpyHostToDevice));
            ck(cudaMemset(q->d_info, 0x7f, sizeof(int)));
            dk::band_lu<T><<<1, q->block>>>(q->d_rp, q->ldaR, q->klr, q->klr, q->d_info);
            ck(cudaGetLastError());
            ck(cudaDeviceSynchronize());
            ck(cudaMemcpy(&info, q->d_info, sizeof(int), cudaMemcpyDeviceToHost));
            if(info != kInfoNone) {
                // row index in the reduced system; report it offset past the matrix
                if(q->info0 == 0) q->info0 = n + info;
            }
        }

        if(o.verbose) {
            std::printf("[gspike] n=%d kl=%d ku=%d parts=%d lda_f=%d reduced_dim=%d\n", n, kl, ku, p, q->lda_f, q->m);
            if(q->info0) std::printf("[gspike] zero pivot at %d (perturbed)\n", q->info0);
        }
    }

    template<class T>
    solver<T>::~solver() {
        delete p_;
    }

    template<class T>
    int solver<T>::solve(int nrhs, T* B, int ldb, stats* st) const {
        impl* q = p_;
        if(nrhs < 0) return -1;
        if(!B) return -2;
        if(ldb < q->n) return -3;
        if(nrhs == 0) return 0;
        if((long long)q->n * nrhs > INT_MAX) return -4; // buffers are counted in int

        q->alloc_arena(nrhs);
        const int n = q->n, block = q->block;
        const long long tot = (long long)n * nrhs;

        T* x = q->arena;
        T* g = x + tot;
        T* bsave = g + tot;
        T* res = bsave + tot;
        T* grj = res + tot;
        T* rs = grj + (long long)2 * q->klu * q->p * nrhs;

        dk::copy_mat<T><<<nrhs, block>>>(n, nrhs, B, ldb, x, n);
        dk::copy_n<T><<<(tot + block - 1) / block, block>>>(tot, x, bsave);
        ck(cudaGetLastError());

        q->solve_inplace(x, g, grj, rs, nrhs);

        const double tol = (q->o.rtol > 0) ? q->o.rtol : num<T>::rtol0;
        double resn = 0.0, bn = 1.0;
        int iters = 0;

        auto residual = [&]() {
            dk::band_matvec<T><<<dim3(nrhs, (n + block - 1) / block), block>>>(
                n, q->kl, q->ku, q->A, q->lda, x, n, res, n, nrhs
            );
            dk::bsub<T><<<(tot + block - 1) / block, block>>>(tot, bsave, res);
            ck(cudaGetLastError());
            T r2 = 0, b2 = 0;
            ckb(num<T>::nrm2(q->h, (int)tot, res, &r2));
            ckb(num<T>::nrm2(q->h, (int)tot, bsave, &b2));
            resn = (double)r2;
            bn = (double)b2;
        };

        residual();
        double prev = resn;
        while(iters < q->o.max_refine && resn > tol * bn) {
            q->solve_inplace(res, g, grj, rs, nrhs); // res <- A^-1 res
            dk::axpby<T><<<(tot + block - 1) / block, block>>>(tot, T(1), res, T(1), x);
            ++iters;
            residual();
            if(!(resn < prev)) break; // no further improvement
            prev = resn;
        }

        dk::copy_mat<T><<<nrhs, block>>>(n, nrhs, x, n, B, ldb);
        ck(cudaGetLastError());

        const int info = q->info0; // factorization status only; refinement quality is in stats
        if(st) {
            st->parts = q->p;
            st->refine = iters;
            st->residual = (bn > 0) ? resn / bn : resn;
        }
        if(q->o.verbose)
            std::printf("[gspike] solve nrhs=%d refine=%d relres=%.3e\n", nrhs, iters, (bn > 0) ? resn / bn : resn);
        return info;
    }

    template<class T>
    int solver<T>::n() const { return p_->n; }
    template<class T>
    int solver<T>::kl() const { return p_->kl; }
    template<class T>
    int solver<T>::ku() const { return p_->ku; }
    template<class T>
    int solver<T>::parts() const { return p_->p; }

    template<class T>
    int spike_gbsv(int n, int kl, int ku, int nrhs, const T* A, int lda, T* B, int ldb, const opts& o, stats* st) {
        try {
            solver<T> s(n, kl, ku, A, lda, o);
            return s.solve(nrhs, B, ldb, st);
        }
        catch(const std::invalid_argument&) {
            return -1;
        }
    }

    template class solver<float>;
    template class solver<double>;
    template int spike_gbsv<float>(int, int, int, int, const float*, int, float*, int, const opts&, stats*);
    template int spike_gbsv<double>(int, int, int, int, const double*, int, double*, int, const opts&, stats*);

} // namespace gspike
