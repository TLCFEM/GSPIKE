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
//
// Mixed precision factorizes a float copy of a double matrix once and builds
// the double solution with float solves (see gspike::mixed_solver).
#include "gspike/spike.h"
#include "kernels.cuh"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gspike {
    namespace {

        constexpr int kMaxParts = 32;
        constexpr int kInfoNone = 0x7f7f7f7f;

        void ck(const cudaError_t e) {
            if(e != cudaSuccess)
                throw std::runtime_error(std::string("gspike: ") + cudaGetErrorString(e));
        }
        void ckb(const cublasStatus_t s) {
            if(s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("gspike: cuBLAS failure");
        }

        template<class T>
        struct num;
        template<>
        struct num<float> {
            static constexpr double rtol0 = 1e-5; // default refinement target
            static cublasStatus_t nrm2(const cublasHandle_t h, const int n, const float* x, float* r) {
                return cublasSnrm2(h, n, x, 1, r);
            }
        };
        template<>
        struct num<double> {
            static constexpr double rtol0 = 1e-12;
            static cublasStatus_t nrm2(const cublasHandle_t h, const int n, const double* x, double* r) {
                return cublasDnrm2(h, n, x, 1, r);
            }
        };

        // 2-norm of an index_t-long vector.  cuBLAS only counts with int, so
        // long vectors (64-bit builds) are reduced in chunks and combined with
        // hypot; for n <= INT_MAX this is a single cuBLAS call.
        template<class T>
        T nrm2(const cublasHandle_t h, const index_t n, const T* x) {
            T acc = T(0);
            bool first = true;
            for(index_t off = 0; off < n;) {
                const int len = (int)std::min<index_t>(n - off, (index_t)INT_MAX);
                T part = T(0);
                ckb(num<T>::nrm2(h, len, x + off, &part));
                acc = first ? part : std::hypot(acc, part);
                first = false;
                off += len;
            }
            return acc;
        }

    } // namespace

    ////////////////////////////////////////////////////////////////////////////////

    template<class T>
    struct solver<T>::impl {
        index_t n = 0, kl = 0, ku = 0, klu = 0, lda = 0;
        int p = 1, block = 256;
        index_t lda_f = 0;
        opts o;
        const T* A = nullptr;
        int info0 = 0;

        cublasHandle_t h = nullptr;
        dk::pdesc<T, index_t>* d_pd = nullptr;
        dk::pdesc<T, index_t>* d_rp = nullptr;
        T** d_spp = nullptr;
        T* d_ab = nullptr;
        index_t* d_piv = nullptr;
        T* d_sp = nullptr;
        T* d_bc = nullptr;
        T* d_cc = nullptr;
        T* d_R = nullptr;
        index_t* d_pivR = nullptr;
        int* d_info = nullptr;
        index_t m = 0, klr = 0, ldaR = 0;

        T* arena = nullptr;
        size_t arena_n = 0;
        index_t arena_nrhs = 0;

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

        void alloc_arena(const index_t nrhs) {
            const size_t need = (size_t)4 * n * nrhs + (size_t)2 * klu * p * nrhs + (size_t)m * nrhs;
            if(need <= arena_n && nrhs <= arena_nrhs) return;
            if(arena) {
                ck(cudaFree(arena));
                arena = nullptr;
            }
            arena_n = need;
            arena_nrhs = nrhs;
            ck(cudaMalloc(&arena, std::max<size_t>(need, 1) * sizeof(T)));
        }

        // X (n x nrhs, leading dimension ldx) holds f on entry and the solution
        // on exit.
        void solve_inplace(T* X, T* g, T* grj, T* rs, const index_t nrhs, const index_t ldx) const {
            const unsigned gc = (unsigned)((nrhs + block - 1) / block);
            const dim3 gcol(1, gc);
            const dim3 gpar(p, gc);
            const dim3 gred(p > 1 ? p - 1 : 1, gc);

            if(p == 1) {
                dk::band_solve<T, index_t><<<gcol, block>>>(d_pd, lda_f, kl, ku, X, ldx, nrhs);
                ck(cudaGetLastError());
                return;
            }
            dk::copy_mat<T, index_t><<<(unsigned)nrhs, block>>>(n, nrhs, X, ldx, g, n);
            dk::band_solve<T, index_t><<<gpar, block>>>(d_pd, lda_f, kl, ku, g, n, nrhs);
            dk::extract_corners<T, index_t><<<gpar, block>>>(d_pd, g, n, klu, nrhs, grj);
            dk::reduced_rhs<T, index_t><<<gred, block>>>(grj, m, klu, rs, nrhs);
            dk::band_solve<T, index_t><<<gcol, block>>>(d_rp, ldaR, klr, klr, rs, m, nrhs);
            dk::correct_rhs<T, index_t><<<gpar, block>>>(d_pd, d_bc, d_cc, rs, klu, m, p, X, ldx, nrhs);
            dk::band_solve<T, index_t><<<gpar, block>>>(d_pd, lda_f, kl, ku, X, ldx, nrhs);
            ck(cudaGetLastError());
        }
    };

    ////////////////////////////////////////////////////////////////////////////////

    template<class T>
    void solver<T>::impl::factor() {
        impl* q = this;
        const index_t n = q->n, kl = q->kl, ku = q->ku, klu = q->klu, lda_f = q->lda_f;
        const int p = q->p, block = q->block;

        std::vector<index_t> row0(p), size(p);
        for(int i = 0; i < p; ++i) {
            const index_t base = n / p, rem = n % p;
            size[i] = base + ((index_t)i < rem ? 1 : 0);
            row0[i] = (i == 0) ? 0 : row0[i - 1] + size[i - 1];
        }

        ck(cudaMalloc(&q->d_ab, (size_t)lda_f * n * sizeof(T)));
        ck(cudaMalloc(&q->d_piv, (size_t)n * sizeof(index_t)));
        ck(cudaMalloc(&q->d_pd, (size_t)p * sizeof(dk::pdesc<T, index_t>)));
        ck(cudaMalloc(&q->d_sp, (size_t)p * 4 * klu * klu * sizeof(T)));
        ck(cudaMalloc(&q->d_bc, (size_t)p * klu * klu * sizeof(T)));
        ck(cudaMalloc(&q->d_cc, (size_t)p * klu * klu * sizeof(T)));
        ck(cudaMalloc(&q->d_spp, (size_t)p * sizeof(T*)));
        ck(cudaMalloc(&q->d_info, sizeof(int)));
        ck(cudaMemset(q->d_info, 0x7f, sizeof(int)));
        ck(cudaMemset(q->d_sp, 0, (size_t)p * 4 * klu * klu * sizeof(T)));

        std::vector<dk::pdesc<T, index_t>> hd(p);
        std::vector<T*> hspp(p);
        for(int i = 0; i < p; ++i) {
            hd[i].ab = q->d_ab + lda_f * row0[i];
            hd[i].piv = q->d_piv + row0[i];
            hd[i].row0 = row0[i];
            hd[i].size = size[i];
            hspp[i] = q->d_sp + (index_t)i * 4 * klu * klu;
        }
        ck(cudaMemcpy(q->d_pd, hd.data(), p * sizeof(dk::pdesc<T, index_t>), cudaMemcpyHostToDevice));
        ck(cudaMemcpy(q->d_spp, hspp.data(), p * sizeof(T*), cudaMemcpyHostToDevice));

        dk::make_blocks<T, index_t><<<p, block>>>(q->d_pd, q->A, n, q->lda, lda_f, ku, kl, klu, q->d_bc, q->d_cc);
        dk::band_lu<T, index_t><<<p, block>>>(q->d_pd, lda_f, kl, ku, q->d_info);
        ck(cudaGetLastError());

        if(p == 1) return;

        // spikes: A_ii [C_i | B_i], solved in chunks of columns (2*klu in total).
        // All spike columns are solved at once (one thread each) as long as the
        // temporary buffer stays within budget.
        const index_t ncol = 2 * klu;
        index_t chunk = ncol;
        while(chunk > 1 && (size_t)n * (size_t)chunk * sizeof(T) > (512ULL << 20)) chunk /= 2;
        T* rhs = nullptr;
        ck(cudaMalloc(&rhs, (size_t)n * (size_t)chunk * sizeof(T)));
        const unsigned nb = (unsigned)((2 * klu * klu + block - 1) / block);
        for(index_t c0 = 0; c0 < ncol; c0 += chunk) {
            const index_t nc = std::min<index_t>(chunk, ncol - c0);
            const int nt = (nc < block) ? (int)(((nc + 31) / 32) * 32) : block;
            dk::spike_rhs<T, index_t><<<dim3(p, (unsigned)nc), block>>>(q->d_pd, q->d_bc, q->d_cc, klu, rhs, n, c0, ncol);
            dk::band_solve<T, index_t><<<dim3(p, 1), nt>>>(q->d_pd, lda_f, kl, ku, rhs, n, nc);
            dk::spike_extract<T, index_t><<<dim3(p, nb), block>>>(q->d_pd, rhs, n, klu, c0, nc, q->d_sp);
            ck(cudaGetLastError());
        }
        ck(cudaFree(rhs));
    }

    ////////////////////////////////////////////////////////////////////////////////

    template<class T>
    solver<T>::solver(index_t n, index_t kl, index_t ku, const T* A, index_t lda, const opts& o)
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

        const index_t pmax = (q->klu == 0) ? 1 : std::max<index_t>(1, n / (2 * q->klu + 1));
        index_t p = o.parts;
        if(p <= 0) {
            p = (q->klu == 0) ? 1 : std::max<index_t>(1, n / (16 * q->klu + 32));
            p = std::min<index_t>(p, kMaxParts);
        }
        p = std::min(std::max<index_t>(p, 1), pmax);
        while(p > 1) { // keep the interface system within a memory budget
            const index_t m = 2 * q->klu * (p - 1);
            const index_t ldaR = 9 * q->klu + 1;
            if((size_t)m * ldaR * sizeof(T) < (256ULL << 20)) break;
            --p;
        }
        q->p = (int)p;
        q->m = 2 * q->klu * (q->p - 1);
        q->klr = 3 * q->klu;
        q->ldaR = 3 * q->klr + 1;

        q->factor();
        ck(cudaDeviceSynchronize());

        int info = 0;
        ck(cudaMemcpy(&info, q->d_info, sizeof(int), cudaMemcpyDeviceToHost));
        q->info0 = (info == kInfoNone) ? 0 : info;

        if(p > 1) {
            ck(cudaMalloc(&q->d_R, (size_t)q->ldaR * q->m * sizeof(T)));
            ck(cudaMalloc(&q->d_pivR, (size_t)q->m * sizeof(index_t)));
            ck(cudaMalloc(&q->d_rp, sizeof(dk::pdesc<T, index_t>)));
            ck(cudaMemset(q->d_R, 0, (size_t)q->ldaR * q->m * sizeof(T)));

            dk::reduced_diag<T, index_t><<<(int)p - 1, q->block>>>(q->d_spp, q->p, q->klu, q->klr, q->ldaR, q->d_R);
            ck(cudaGetLastError());

            dk::pdesc<T, index_t> rp{};
            rp.ab = q->d_R;
            rp.piv = q->d_pivR;
            rp.row0 = 0;
            rp.size = q->m;
            ck(cudaMemcpy(q->d_rp, &rp, sizeof(rp), cudaMemcpyHostToDevice));
            ck(cudaMemset(q->d_info, 0x7f, sizeof(int)));
            dk::band_lu<T, index_t><<<1, q->block>>>(q->d_rp, q->ldaR, q->klr, q->klr, q->d_info);
            ck(cudaGetLastError());
            ck(cudaDeviceSynchronize());
            ck(cudaMemcpy(&info, q->d_info, sizeof(int), cudaMemcpyDeviceToHost));
            if(info != kInfoNone) {
                // row index in the reduced system; report it offset past the matrix
                if(q->info0 == 0) q->info0 = (info > INT_MAX - n) ? INT_MAX : (int)(n + info);
            }
        }

        if(o.verbose) {
            std::printf("[gspike] n=%lld kl=%lld ku=%lld parts=%d lda_f=%lld reduced_dim=%lld (%d-bit index)\n",
                        (long long)n, (long long)kl, (long long)ku, q->p, (long long)q->lda_f, (long long)q->m,
                        (int)(8 * sizeof(index_t)));
            if(q->info0) std::printf("[gspike] zero pivot at %d (perturbed)\n", q->info0);
        }
    }

    template<class T>
    solver<T>::~solver() {
        delete p_;
    }

    template<class T>
    int solver<T>::solve(const index_t nrhs, T* B, const index_t ldb, stats* st) const {
        impl* q = p_;
        if(nrhs < 0) return -1;
        if(!B) return -2;
        if(ldb < q->n) return -3;
        if(nrhs == 0) return 0;
        // Every buffer and offset is counted in index_t, and the arena holds
        // 4 * n * nrhs plus the spike/interface scratch.
        if(nrhs > std::numeric_limits<index_t>::max() / q->n) return -4;
        const index_t tot = q->n * nrhs;
        if(tot > std::numeric_limits<index_t>::max() / 8) return -4;

        q->alloc_arena(nrhs);
        const index_t n = q->n;
        const int block = q->block;

        T* x = q->arena;
        T* g = x + tot;
        T* bsave = g + tot;
        T* res = bsave + tot;
        T* grj = res + tot;
        T* rs = grj + 2 * q->klu * q->p * nrhs;

        dk::copy_mat<T, index_t><<<(unsigned)nrhs, block>>>(n, nrhs, B, ldb, x, n);
        dk::copy_n<T, index_t><<<(unsigned)((tot + block - 1) / block), block>>>(tot, x, bsave);
        ck(cudaGetLastError());

        q->solve_inplace(x, g, grj, rs, nrhs, n);

        const double tol = (q->o.rtol > 0) ? q->o.rtol : num<T>::rtol0;
        double resn = 0.0, bn = 1.0;
        int iters = 0;

        auto residual = [&]() {
            dk::band_matvec<T, index_t><<<dim3((unsigned)nrhs, (unsigned)((n + block - 1) / block)), block>>>(
                n, q->kl, q->ku, q->A, q->lda, x, n, res, n, nrhs
            );
            dk::bsub<T, index_t><<<(unsigned)((tot + block - 1) / block), block>>>(tot, bsave, res);
            ck(cudaGetLastError());
            const T r2 = nrm2<T>(q->h, tot, res);
            const T b2 = nrm2<T>(q->h, tot, bsave);
            resn = (double)r2;
            bn = (double)b2;
        };

        residual();
        double prev = resn;
        while(iters < q->o.max_refine && resn > tol * bn) {
            q->solve_inplace(res, g, grj, rs, nrhs, n); // res <- A^-1 res
            dk::axpby<T, index_t><<<(unsigned)((tot + block - 1) / block), block>>>(tot, T(1), res, T(1), x);
            ++iters;
            residual();
            if(!(resn < prev)) break; // no further improvement
            prev = resn;
        }

        dk::copy_mat<T, index_t><<<(unsigned)nrhs, block>>>(n, nrhs, x, n, B, ldb);
        ck(cudaGetLastError());

        const int info = q->info0; // factorization status only; refinement quality is in stats
        if(st) {
            st->parts = q->p;
            st->refine = iters;
            st->residual = (bn > 0) ? resn / bn : resn;
        }
        if(q->o.verbose)
            std::printf("[gspike] solve nrhs=%lld refine=%d relres=%.3e\n", (long long)nrhs, iters, (bn > 0) ? resn / bn : resn);
        return info;
    }

    template<class T>
    void solver<T>::solve_direct(const index_t nrhs, T* RHS, const index_t ldr) const {
        impl* q = p_;
        if(nrhs < 0) throw std::invalid_argument("gspike: negative right-hand side count");
        if(!RHS) throw std::invalid_argument("gspike: null right-hand side");
        if(ldr < q->n) throw std::invalid_argument("gspike: ldr < n");
        if(nrhs == 0) return;
        if(nrhs > std::numeric_limits<index_t>::max() / q->n)
            throw std::invalid_argument("gspike: workspace overflow");
        const index_t tot = q->n * nrhs;
        if(tot > std::numeric_limits<index_t>::max() / 8)
            throw std::invalid_argument("gspike: workspace overflow");

        q->alloc_arena(nrhs);
        T* g = q->arena;
        T* grj = g + tot;
        T* rs = grj + 2 * q->klu * q->p * nrhs;
        q->solve_inplace(RHS, g, grj, rs, nrhs, ldr);
    }

    template<class T>
    int solver<T>::info() const { return p_->info0; }

    template<class T>
    index_t solver<T>::n() const { return p_->n; }
    template<class T>
    index_t solver<T>::kl() const { return p_->kl; }
    template<class T>
    index_t solver<T>::ku() const { return p_->ku; }
    template<class T>
    int solver<T>::parts() const { return p_->p; }

    ////////////////////////////////////////////////////////////////////////////////
    // Mixed precision: factorize a float copy of A once, refine in double.

    struct mixed_solver::impl {
        index_t n = 0, kl = 0, ku = 0, lda = 0;
        int iters = 2, block = 256;
        opts o;
        const double* A = nullptr;
        float* dA_f = nullptr;
        solver<float>* fs = nullptr;
        cublasHandle_t h = nullptr;

        double* arena = nullptr; // x, b (contiguous copy of the RHS), r
        float* farena = nullptr; // rf, the single-precision residual/correction
        index_t arena_nrhs = 0;

        ~impl() {
            delete fs;
            if(dA_f) cudaFree(dA_f);
            if(arena) cudaFree(arena);
            if(farena) cudaFree(farena);
            if(h) cublasDestroy(h);
        }

        void alloc(const index_t nrhs) {
            if(nrhs <= arena_nrhs) return;
            if(arena) {
                ck(cudaFree(arena));
                arena = nullptr;
            }
            if(farena) {
                ck(cudaFree(farena));
                farena = nullptr;
            }
            const size_t tot = (size_t)n * (size_t)nrhs;
            ck(cudaMalloc(&arena, 3 * tot * sizeof(double)));
            ck(cudaMalloc(&farena, tot * sizeof(float)));
            arena_nrhs = nrhs;
        }
    };

    mixed_solver::mixed_solver(const index_t n, const index_t kl, const index_t ku, const double* A, const index_t lda, const opts& o)
        : p_(new impl) {
        impl* q = p_;
        q->o = o;
        if(o.block >= 32 && o.block <= 1024) q->block = o.block;
        if(n < 1 || kl < 0 || ku < 0 || !A) throw std::invalid_argument("gspike: bad dimensions");
        if(lda < kl + ku + 1) throw std::invalid_argument("gspike: lda < kl+ku+1");
        if((size_t)lda * n > (size_t)std::numeric_limits<index_t>::max())
            throw std::invalid_argument("gspike: band storage does not fit the index type");

        q->n = n;
        q->kl = kl;
        q->ku = ku;
        q->lda = lda;
        q->A = A;
        q->iters = std::max(1, o.mixed_iters);
        ckb(cublasCreate(&q->h));

        // The single-precision problem is just a copy of the double band matrix.
        const index_t na = lda * n;
        ck(cudaMalloc(&q->dA_f, (size_t)na * sizeof(float)));
        dk::cast_copy<float, double, index_t><<<(unsigned)((na + q->block - 1) / q->block), q->block>>>(na, A, q->dA_f);
        ck(cudaGetLastError());

        opts fo = o;
        fo.max_refine = 0; // solve_direct() only; no float refinement
        q->fs = new solver<float>(n, kl, ku, q->dA_f, lda, fo);

        if(o.verbose)
            std::printf("[gspike] mixed: n=%lld kl=%lld ku=%lld parts=%d float iters=%d\n",
                        (long long)n, (long long)kl, (long long)ku, q->fs->parts(), q->iters);
    }

    mixed_solver::~mixed_solver() { delete p_; }

    int mixed_solver::solve(const index_t nrhs, double* B, const index_t ldb, stats* st) const {
        impl* q = p_;
        if(nrhs < 0) return -1;
        if(!B) return -2;
        if(ldb < q->n) return -3;
        if(nrhs == 0) {
            if(st) {
                st->parts = q->fs->parts();
                st->refine = 0;
                st->residual = 0.0;
            }
            return q->fs->info();
        }
        if(nrhs > std::numeric_limits<index_t>::max() / q->n) return -4;
        const index_t tot = q->n * nrhs;
        if(tot > std::numeric_limits<index_t>::max() / 8) return -4;

        q->alloc(nrhs);
        const index_t n = q->n;
        const int block = q->block;
        const unsigned gc = (unsigned)((tot + block - 1) / block);
        const unsigned gr = (unsigned)((n + block - 1) / block);

        double* x = q->arena;
        double* b = x + tot;
        double* r = b + tot;
        float* rf = q->farena;

        dk::copy_mat<double, index_t><<<(unsigned)nrhs, block>>>(n, nrhs, B, ldb, b, n);
        dk::copy_n<double, index_t><<<gc, block>>>(tot, b, r); // residual of x = 0
        ck(cudaMemset(x, 0, (size_t)tot * sizeof(double)));
        ck(cudaGetLastError());

        const double bn = nrm2<double>(q->h, tot, b);
        double resn = bn, prev = bn;
        int iters = 0;
        const double tol = (q->o.rtol > 0) ? q->o.rtol : num<double>::rtol0;

        while(iters < q->iters) {
            if(resn <= tol * bn) break;

            // correction d = A_f^-1 r, accumulated into the double solution
            dk::cast_copy<float, double, index_t><<<gc, block>>>(tot, r, rf);
            q->fs->solve_direct(nrhs, rf, n);
            dk::cast_axpy<double, float, index_t><<<gc, block>>>(tot, 1.0, rf, x);
            ck(cudaGetLastError());

            dk::band_matvec<double, index_t><<<dim3((unsigned)nrhs, gr), block>>>(n, q->kl, q->ku, q->A, q->lda, x, n, r, n, nrhs);
            dk::bsub<double, index_t><<<gc, block>>>(tot, b, r);
            ck(cudaGetLastError());
            resn = (double)nrm2<double>(q->h, tot, r);
            ++iters;

            if(iters > 1 && !(resn < prev)) { // the correction did not help: drop it
                dk::cast_axpy<double, float, index_t><<<gc, block>>>(tot, -1.0, rf, x);
                ck(cudaGetLastError());
                resn = prev;
                --iters;
                break;
            }
            prev = resn;
        }

        dk::copy_mat<double, index_t><<<(unsigned)nrhs, block>>>(n, nrhs, x, n, B, ldb);
        ck(cudaGetLastError());

        if(st) {
            st->parts = q->fs->parts();
            st->refine = iters;
            st->residual = (bn > 0) ? resn / bn : resn;
        }
        if(q->o.verbose)
            std::printf("[gspike] mixed solve nrhs=%lld iters=%d relres=%.3e\n",
                        (long long)nrhs, iters, (bn > 0) ? resn / bn : resn);
        return q->fs->info();
    }

    index_t mixed_solver::n() const { return p_->n; }
    index_t mixed_solver::kl() const { return p_->kl; }
    index_t mixed_solver::ku() const { return p_->ku; }
    int mixed_solver::parts() const { return p_->fs->parts(); }

    int spike_gbsv_mixed(const index_t n, const index_t kl, const index_t ku, const index_t nrhs, const double* A, const index_t lda, double* B, const index_t ldb, const opts& o, stats* st) {
        try {
            mixed_solver s(n, kl, ku, A, lda, o);
            return s.solve(nrhs, B, ldb, st);
        }
        catch(const std::invalid_argument&) {
            return -1;
        }
    }

    template<class T>
    int spike_gbsv(const index_t n, const index_t kl, const index_t ku, const index_t nrhs, const T* A, const index_t lda, T* B, const index_t ldb, const opts& o, stats* st) {
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
    template int spike_gbsv<float>(index_t, index_t, index_t, index_t, const float*, index_t, float*, index_t, const opts&, stats*);
    template int spike_gbsv<double>(index_t, index_t, index_t, index_t, const double*, index_t, double*, index_t, const opts&, stats*);

} // namespace gspike
