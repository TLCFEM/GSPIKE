// Head-to-head on the same banded matrix:
//   * gspike mixed precision (float factor + double correction)
//   * Intel MKL general band LU (s/dgbtrf + s/dgbtrs)
//   * cuSPARSE CSR SpMV inside a Jacobi-preconditioned BiCGStab
// Optional target, enabled with -DGSPIKE_DUEL=ON when MKL is installed.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <gspike/spike.h>
#include <mkl/mkl.h>
#include <random>
#include <vector>

namespace {

    void ck(cudaError_t e, const char* what) {
        if(e != cudaSuccess) {
            std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
            std::exit(1);
        }
    }
    template<class E>
    void cke(E e, const char* what) {
        if(e != 0) {
            std::fprintf(stderr, "%s: %d\n", what, (int)e);
            std::exit(1);
        }
    }

    // Identical generator to spike_bench.
    template<class T>
    void make_band(int n, int klu, bool dominant, std::vector<T>& a, std::vector<double>& b, int nrhs, int lda) {
        a.assign((size_t)lda * n, T(0));
        b.assign((size_t)n * nrhs, 0.0);
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        for(int j = 0; j < n; ++j) {
            double rs = 0;
            for(int i = std::max(0, j - klu); i <= std::min(n - 1, j + klu); ++i) {
                if(i == j) continue;
                const double v = u(gen) / (1.0 + std::abs(i - j));
                a[(size_t)j * lda + klu + i - j] = (T)v;
                rs += std::abs(v);
            }
            a[(size_t)j * lda + klu] = (T)(dominant ? rs + 1.0 : u(gen) * 0.5);
        }
        for(auto& v : b) v = u(gen);
    }

    template<class T>
    double residual(int n, int klu, int lda, const std::vector<T>& a, const T* x, int ldx, const std::vector<double>& b, int nrhs) {
        double rn = 0, bn = 0;
        for(int c = 0; c < nrhs; ++c)
            for(int i = 0; i < n; ++i) {
                double s = 0;
                for(int j = std::max(0, i - klu); j <= std::min(n - 1, i + klu); ++j)
                    s += (double)a[(size_t)j * lda + klu + i - j] * (double)x[(size_t)c * ldx + j];
                const double r = b[(size_t)c * n + i] - s;
                rn += r * r;
                bn += b[(size_t)c * n + i] * b[(size_t)c * n + i];
            }
        return std::sqrt(rn / std::max(bn, 1e-300));
    }

    void elapsed(cudaEvent_t a, cudaEvent_t b, double& out) {
        float ms = 0;
        ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
        out = ms;
    }

    struct result {
        double setup_ms = 0, solve_ms = 0, res = 0, iters = 0;
    };

    // ---- gspike -------------------------------------------------------------
    result run_gspike(int n, int klu, int nrhs, const std::vector<double>& a, const std::vector<double>& b, int lda) {
        result t;
        double *da = nullptr, *db = nullptr;
        ck(cudaMalloc(&da, a.size() * sizeof(double)), "malloc A");
        ck(cudaMalloc(&db, b.size() * sizeof(double)), "malloc B");
        ck(cudaMemcpy(da, a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice), "copy A");
        cudaEvent_t e0, e1;
        ck(cudaEventCreate(&e0), "event");
        ck(cudaEventCreate(&e1), "event");
        ck(cudaEventRecord(e0), "record");
        gspike::mixed_solver s(n, klu, klu, da, lda, gspike::opts{});
        ck(cudaEventRecord(e1), "record");
        ck(cudaEventSynchronize(e1), "sync");
        elapsed(e0, e1, t.setup_ms);
        ck(cudaMemcpy(db, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice), "copy B");
        ck(cudaEventRecord(e0), "record");
        s.solve(nrhs, db, n, nullptr);
        ck(cudaEventRecord(e1), "record");
        ck(cudaEventSynchronize(e1), "sync");
        elapsed(e0, e1, t.solve_ms);
        std::vector<double> x(b.size());
        ck(cudaMemcpy(x.data(), db, x.size() * sizeof(double), cudaMemcpyDeviceToHost), "copy X");
        t.res = residual(n, klu, lda, a, x.data(), n, b, nrhs);
        cudaFree(da);
        cudaFree(db);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        return t;
    }

    // ---- gspike double precision ---------------------------------------------
    result run_gspike_double(int n, int klu, int nrhs, const std::vector<double>& a, const std::vector<double>& b, int lda) {
        result t;
        double *da = nullptr, *db = nullptr;
        ck(cudaMalloc(&da, a.size() * sizeof(double)), "malloc A");
        ck(cudaMalloc(&db, b.size() * sizeof(double)), "malloc B");
        ck(cudaMemcpy(da, a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice), "copy A");
        cudaEvent_t e0, e1;
        ck(cudaEventCreate(&e0), "event");
        ck(cudaEventCreate(&e1), "event");
        ck(cudaEventRecord(e0), "record");
        gspike::solver<double> s(n, klu, klu, da, lda, gspike::opts{});
        ck(cudaEventRecord(e1), "record");
        ck(cudaEventSynchronize(e1), "sync");
        elapsed(e0, e1, t.setup_ms);
        ck(cudaMemcpy(db, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice), "copy B");
        ck(cudaEventRecord(e0), "record");
        s.solve(nrhs, db, n, nullptr);
        ck(cudaEventRecord(e1), "record");
        ck(cudaEventSynchronize(e1), "sync");
        elapsed(e0, e1, t.solve_ms);
        std::vector<double> x(b.size());
        ck(cudaMemcpy(x.data(), db, x.size() * sizeof(double), cudaMemcpyDeviceToHost), "copy X");
        t.res = residual(n, klu, lda, a, x.data(), n, b, nrhs);
        cudaFree(da);
        cudaFree(db);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        return t;
    }

    // ---- MKL general band LU ------------------------------------------------
    template<class T>
    result run_mkl(int n, int klu, int nrhs, const std::vector<T>& a, const std::vector<double>& b, int lda) {
        result t;
        const int ldab = 2 * klu + klu + 1, off = klu + klu;
        std::vector<T> ab((size_t)ldab * n, T(0));
        std::vector<T> rhs(b.size());
        for(int j = 0; j < n; ++j)
            for(int i = std::max(0, j - klu); i <= std::min(n - 1, j + klu); ++i)
                ab[(size_t)j * ldab + off + i - j] = a[(size_t)j * lda + klu + i - j];
        for(size_t i = 0; i < b.size(); ++i) rhs[i] = (T)b[i];
        std::vector<int> ipiv(n);
        int info = 0;
        const auto start = std::chrono::steady_clock::now();
        if constexpr(sizeof(T) == 4)
            sgbtrf_(&n, &n, &klu, &klu, ab.data(), &ldab, ipiv.data(), &info);
        else
            dgbtrf_(&n, &n, &klu, &klu, ab.data(), &ldab, ipiv.data(), &info);
        const auto fac = std::chrono::steady_clock::now();
        const char trans = 'N';
        if constexpr(sizeof(T) == 4)
            sgbtrs_(&trans, &n, &klu, &klu, &nrhs, ab.data(), &ldab, ipiv.data(), rhs.data(), &n, &info);
        else
            dgbtrs_(&trans, &n, &klu, &klu, &nrhs, ab.data(), &ldab, ipiv.data(), rhs.data(), &n, &info);
        const auto sol = std::chrono::steady_clock::now();
        t.setup_ms = std::chrono::duration<double, std::milli>(fac - start).count();
        t.solve_ms = std::chrono::duration<double, std::milli>(sol - fac).count();
        t.res = residual(n, klu, lda, a, rhs.data(), n, b, nrhs);
        return t;
    }

    // ---- cuSPARSE CSR + BiCGStab -------------------------------------------
    __global__ void fill_csr(int n, int klu, int lda, const double* __restrict__ band, const int* __restrict__ rowptr, int* __restrict__ colind, double* __restrict__ val) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n) return;
        int p = rowptr[i];
        for(int j = max(0, i - klu); j <= min(n - 1, i + klu); ++j) {
            colind[p] = j;
            val[p] = band[(long long)j * lda + klu + i - j];
            ++p;
        }
    }
    // Column-oriented band SpMV: one warp per band column, lanes walk the
    // contiguous band segment.  No column indices and no CSR conversion.
    __global__ void band_spmv(int n, int klu, int lda, const double* __restrict__ band, const double* __restrict__ x, double* __restrict__ y) {
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        const int j = blockIdx.x * (blockDim.x / 32) + warp;
        if(j >= n) return;
        const double xj = x[j];
        const int i0 = max(0, j - klu), i1 = min(n - 1, j + klu);
        const double* c = band + (long long)j * lda + klu - j;
        for(int i = i0 + lane; i <= i1; i += 32) atomicAdd(y + i, c[i] * xj);
    }

    __global__ void k_copy(int n, const double* __restrict__ x, double* __restrict__ y) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) y[i] = x[i];
    }
    __global__ void k_precond(int n, const double* __restrict__ dinv, const double* __restrict__ x, double* __restrict__ y) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) y[i] = dinv[i] * x[i];
    }
    // out aliases p by design (in-place direction update), so it is not restrict.
    __global__ void k_pupdate(int n, const double* __restrict__ beta, const double* __restrict__ omega, const double* __restrict__ r, const double* __restrict__ p, const double* __restrict__ v, double* out) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) out[i] = r[i] + (*beta) * (p[i] - (*omega) * v[i]);
    }
    __global__ void k_supdate(int n, const double* __restrict__ alpha, const double* __restrict__ r, const double* __restrict__ v, double* __restrict__ s) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) s[i] = r[i] - (*alpha) * v[i];
    }
    __global__ void k_xupdate(int n, const double* __restrict__ alpha, const double* __restrict__ ph, const double* __restrict__ omega, const double* __restrict__ sh, double* __restrict__ x) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) x[i] += (*alpha) * ph[i] + (*omega) * sh[i];
    }
    __global__ void k_rupdate(int n, const double* __restrict__ omega, const double* __restrict__ t, const double* __restrict__ s, double* __restrict__ r) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n) r[i] = s[i] - (*omega) * t[i];
    }
    __global__ void scal_beta(double* rho, double* rhoold, double* alpha, double* omega, double* beta) {
        if(threadIdx.x == 0) *beta = (*rho / *rhoold) * (*alpha / *omega);
    }
    __global__ void scal_div(double* num, double* den, double* out) {
        if(threadIdx.x == 0) *out = *num / *den;
    }
    __global__ void scal_copy(double* a, double* b) {
        if(threadIdx.x == 0) *b = *a;
    }

    result run_sparse(int n, int klu, int nrhs, const std::vector<double>& a, const std::vector<double>& b, int lda) {
        result t;
        std::vector<int> rowptr(n + 1);
        for(int i = 0; i < n; ++i) {
            const int j0 = std::max(0, i - klu), j1 = std::min(n - 1, i + klu);
            rowptr[i + 1] = rowptr[i] + (j1 - j0 + 1);
        }
        const int nnz = rowptr[n];
        const size_t sz = (size_t)n * sizeof(double);
        double *dband = nullptr, *dval = nullptr;
        int *dptr = nullptr, *dcol = nullptr;
        ck(cudaMalloc(&dband, a.size() * sizeof(double)), "band");
        ck(cudaMalloc(&dval, (size_t)nnz * sizeof(double)), "csr val");
        ck(cudaMalloc(&dptr, (size_t)(n + 1) * sizeof(int)), "csr ptr");
        ck(cudaMalloc(&dcol, (size_t)nnz * sizeof(int)), "csr col");
        ck(cudaMemcpy(dband, a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice), "band copy");
        ck(cudaMemcpy(dptr, rowptr.data(), (size_t)(n + 1) * sizeof(int), cudaMemcpyHostToDevice), "ptr copy");

        cudaEvent_t e0, e1;
        ck(cudaEventCreate(&e0), "event");
        ck(cudaEventCreate(&e1), "event");
        cusparseHandle_t sp = nullptr;
        cublasHandle_t bl = nullptr;
        cke(cusparseCreate(&sp), "cusparse");
        cke(cublasCreate(&bl), "cublas");
        cke(cublasSetPointerMode(bl, CUBLAS_POINTER_MODE_DEVICE), "pointer mode");
        ck(cudaEventRecord(e0), "record");
        fill_csr<<<(n + 255) / 256, 256>>>(n, klu, lda, dband, dptr, dcol, dval);
        ck(cudaGetLastError(), "fill csr");

        double *r = nullptr, *rhat = nullptr, *p = nullptr, *ph = nullptr;
        double *v = nullptr, *s = nullptr, *sh = nullptr, *tvec = nullptr, *x = nullptr, *bb = nullptr, *dinv = nullptr;
        const auto allocv = [&](double*& q) { ck(cudaMalloc(&q, sz), "alloc vector"); };
        allocv(r);
        allocv(rhat);
        allocv(p);
        allocv(ph);
        allocv(v);
        allocv(s);
        allocv(sh);
        allocv(tvec);
        allocv(x);
        allocv(bb);
        allocv(dinv);
        double *rho = nullptr, *rhoold = nullptr, *alpha = nullptr, *omega = nullptr;
        double *beta = nullptr, *num = nullptr, *den = nullptr, *norm = nullptr;
        const auto allocs = [&](double*& q) {
            ck(cudaMalloc(&q, sizeof(double)), "alloc scalar");
            ck(cudaMemset(q, 0, sizeof(double)), "zero scalar");
        };
        allocs(rho);
        allocs(rhoold);
        allocs(alpha);
        allocs(omega);
        allocs(beta);
        allocs(num);
        allocs(den);
        allocs(norm);
        std::vector<double> hdiag(n);
        for(int i = 0; i < n; ++i) hdiag[i] = 1.0 / a[(size_t)i * lda + klu];
        ck(cudaMemcpy(dinv, hdiag.data(), sz, cudaMemcpyHostToDevice), "diag");

        cusparseSpMatDescr_t mat = nullptr;
        cke(cusparseCreateCsr(&mat, n, n, nnz, dptr, dcol, dval, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F), "csr descr");
        cusparseDnVecDescr_t vph = nullptr, vsh = nullptr, vv = nullptr, vt = nullptr;
        cke(cusparseCreateDnVec(&vph, n, ph, CUDA_R_64F), "vec ph");
        cke(cusparseCreateDnVec(&vsh, n, sh, CUDA_R_64F), "vec sh");
        cke(cusparseCreateDnVec(&vv, n, v, CUDA_R_64F), "vec v");
        cke(cusparseCreateDnVec(&vt, n, tvec, CUDA_R_64F), "vec t");
        const double one = 1.0, zero = 0.0;
        size_t bufsize = 0;
        cke(cusparseSpMV_bufferSize(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat, vph, &zero, vv, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufsize), "spmv size");
        void* buffer = nullptr;
        ck(cudaMalloc(&buffer, bufsize), "spmv buffer");
        cke(cusparseSpMV_preprocess(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat, vph, &zero, vv, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer), "spmv preprocess");
        cke(cusparseSpMV_preprocess(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat, vsh, &zero, vt, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer), "spmv2 preprocess");
        ck(cudaEventRecord(e1), "record");
        ck(cudaEventSynchronize(e1), "sync");
        elapsed(e0, e1, t.setup_ms);

        ck(cudaEventRecord(e0), "record");
        std::vector<double> xh((size_t)n * nrhs);
        double max_iters = 0;
        const dim3 grid((n + 255) / 256), th(256);
        for(int c = 0; c < nrhs; ++c) {
            ck(cudaMemcpy(bb, b.data() + (size_t)c * n, sz, cudaMemcpyHostToDevice), "b copy");
            ck(cudaMemset(x, 0, sz), "zero x");
            k_copy<<<grid, th>>>(n, bb, r);
            k_copy<<<grid, th>>>(n, r, rhat);
            k_copy<<<grid, th>>>(n, r, p);
            k_copy<<<grid, th>>>(n, r, v);
            double init = 1.0;
            ck(cudaMemcpy(rhoold, &init, sizeof(double), cudaMemcpyHostToDevice), "rhoold");
            init = 0.0;
            ck(cudaMemcpy(alpha, &init, sizeof(double), cudaMemcpyHostToDevice), "alpha");
            init = 1.0;
            ck(cudaMemcpy(omega, &init, sizeof(double), cudaMemcpyHostToDevice), "omega");
            cke(cublasDnrm2(bl, n, r, 1, norm), "norm b");
            double bn = 0;
            ck(cudaMemcpy(&bn, norm, sizeof(double), cudaMemcpyDeviceToHost), "read norm b");
            const double tol = 1e-9 * bn;
            int it = 0;
            for(; it < 2000; ++it) {
                cke(cublasDdot(bl, n, rhat, 1, r, 1, rho), "rho");
                scal_beta<<<1, 32>>>(rho, rhoold, alpha, omega, beta);
                k_pupdate<<<grid, th>>>(n, beta, omega, r, p, v, p);
                k_precond<<<grid, th>>>(n, dinv, p, ph);
                cke(cusparseSpMV(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat, vph, &zero, vv, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer), "spmv");
                cke(cublasDdot(bl, n, rhat, 1, v, 1, den), "den");
                scal_div<<<1, 32>>>(rho, den, alpha);
                k_supdate<<<grid, th>>>(n, alpha, r, v, s);
                k_precond<<<grid, th>>>(n, dinv, s, sh);
                cke(cusparseSpMV(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat, vsh, &zero, vt, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer), "spmv2");
                cke(cublasDdot(bl, n, tvec, 1, s, 1, num), "num");
                cke(cublasDdot(bl, n, tvec, 1, tvec, 1, den), "den2");
                scal_div<<<1, 32>>>(num, den, omega);
                k_xupdate<<<grid, th>>>(n, alpha, ph, omega, sh, x);
                k_rupdate<<<grid, th>>>(n, omega, tvec, s, r);
                scal_copy<<<1, 32>>>(rho, rhoold);
                cke(cublasDnrm2(bl, n, r, 1, norm), "norm r");
                double rn = 0;
                ck(cudaMemcpy(&rn, norm, sizeof(double), cudaMemcpyDeviceToHost), "read norm r");
                if(rn < tol) {
                    ++it;
                    break;
                }
            }
            max_iters = std::max(max_iters, (double)it);
            ck(cudaMemcpy(xh.data() + (size_t)c * n, x, sz, cudaMemcpyDeviceToHost), "x copy");
        }
        ck(cudaEventRecord(e1), "record");
        ck(cudaEventSynchronize(e1), "sync");
        elapsed(e0, e1, t.solve_ms);
        t.iters = max_iters;
        t.res = residual(n, klu, lda, a, xh.data(), n, b, nrhs);
        for(double* q : {r, rhat, p, ph, v, s, sh, tvec, x, bb, dinv}) cudaFree(q);
        for(double* q : {rho, rhoold, alpha, omega, beta, num, den, norm}) cudaFree(q);
        cusparseDestroySpMat(mat);
        cusparseDestroyDnVec(vph);
        cusparseDestroyDnVec(vsh);
        cusparseDestroyDnVec(vv);
        cusparseDestroyDnVec(vt);
        cusparseDestroy(sp);
        cublasDestroy(bl);
        cudaFree(buffer);
        cudaFree(dband);
        cudaFree(dval);
        cudaFree(dptr);
        cudaFree(dcol);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        return t;
    }

    // Same BiCGStab, but with a band-aware SpMV (compact band, no CSR).
    result run_band_sparse(int n, int klu, int nrhs, const std::vector<double>& a, const std::vector<double>& b, int lda) {
        result t;
        const size_t sz = (size_t)n * sizeof(double);
        double *dband = nullptr, *r = nullptr, *rhat = nullptr, *p = nullptr, *ph = nullptr;
        double *v = nullptr, *ss = nullptr, *sh = nullptr, *tv = nullptr, *x = nullptr, *bb = nullptr, *dinv = nullptr;
        ck(cudaMalloc(&dband, a.size() * sizeof(double)), "band");
        ck(cudaMemcpy(dband, a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice), "band copy");
        const auto allocv = [&](double*& q) { ck(cudaMalloc(&q, sz), "alloc vector"); };
        allocv(r);
        allocv(rhat);
        allocv(p);
        allocv(ph);
        allocv(v);
        allocv(ss);
        allocv(sh);
        allocv(tv);
        allocv(x);
        allocv(bb);
        allocv(dinv);
        double *rho = nullptr, *rhoold = nullptr, *alpha = nullptr, *omega = nullptr;
        double *beta = nullptr, *num = nullptr, *den = nullptr, *norm = nullptr;
        const auto allocs = [&](double*& q) {
            ck(cudaMalloc(&q, sizeof(double)), "alloc scalar");
            ck(cudaMemset(q, 0, sizeof(double)), "zero scalar");
        };
        allocs(rho);
        allocs(rhoold);
        allocs(alpha);
        allocs(omega);
        allocs(beta);
        allocs(num);
        allocs(den);
        allocs(norm);
        std::vector<double> hdiag(n);
        for(int i = 0; i < n; ++i) hdiag[i] = 1.0 / a[(size_t)i * lda + klu];
        ck(cudaMemcpy(dinv, hdiag.data(), sz, cudaMemcpyHostToDevice), "diag");
        cublasHandle_t bl = nullptr;
        cke(cublasCreate(&bl), "cublas");
        cke(cublasSetPointerMode(bl, CUBLAS_POINTER_MODE_DEVICE), "pointer mode");

        const dim3 grid((n + 255) / 256), th(256);
        const dim3 bgrid((n + 7) / 8), bth(256); // 8 warps per block
        const auto spmv = [&](const double* in, double* out) {
            ck(cudaMemset(out, 0, sz), "zero output");
            band_spmv<<<bgrid, bth>>>(n, klu, lda, dband, in, out);
        };
        cudaEvent_t e0, e1;
        ck(cudaEventCreate(&e0), "event");
        ck(cudaEventCreate(&e1), "event");
        ck(cudaEventRecord(e0), "record");
        std::vector<double> xh((size_t)n * nrhs);
        double max_iters = 0;
        for(int c = 0; c < nrhs; ++c) {
            ck(cudaMemcpy(bb, b.data() + (size_t)c * n, sz, cudaMemcpyHostToDevice), "b copy");
            ck(cudaMemset(x, 0, sz), "zero x");
            k_copy<<<grid, th>>>(n, bb, r);
            k_copy<<<grid, th>>>(n, r, rhat);
            k_copy<<<grid, th>>>(n, r, p);
            k_copy<<<grid, th>>>(n, r, v);
            double init = 1.0;
            ck(cudaMemcpy(rhoold, &init, sizeof(double), cudaMemcpyHostToDevice), "rhoold");
            init = 0.0;
            ck(cudaMemcpy(alpha, &init, sizeof(double), cudaMemcpyHostToDevice), "alpha");
            init = 1.0;
            ck(cudaMemcpy(omega, &init, sizeof(double), cudaMemcpyHostToDevice), "omega");
            cke(cublasDnrm2(bl, n, r, 1, norm), "norm b");
            double bn = 0;
            ck(cudaMemcpy(&bn, norm, sizeof(double), cudaMemcpyDeviceToHost), "norm b read");
            const double tol = 1e-9 * bn;
            int it = 0;
            for(; it < 2000; ++it) {
                cke(cublasDdot(bl, n, rhat, 1, r, 1, rho), "rho");
                scal_beta<<<1, 32>>>(rho, rhoold, alpha, omega, beta);
                k_pupdate<<<grid, th>>>(n, beta, omega, r, p, v, p);
                k_precond<<<grid, th>>>(n, dinv, p, ph);
                spmv(ph, v);
                cke(cublasDdot(bl, n, rhat, 1, v, 1, den), "den");
                scal_div<<<1, 32>>>(rho, den, alpha);
                k_supdate<<<grid, th>>>(n, alpha, r, v, ss);
                k_precond<<<grid, th>>>(n, dinv, ss, sh);
                spmv(sh, tv);
                cke(cublasDdot(bl, n, tv, 1, ss, 1, num), "num");
                cke(cublasDdot(bl, n, tv, 1, tv, 1, den), "den2");
                scal_div<<<1, 32>>>(num, den, omega);
                k_xupdate<<<grid, th>>>(n, alpha, ph, omega, sh, x);
                k_rupdate<<<grid, th>>>(n, omega, tv, ss, r);
                scal_copy<<<1, 32>>>(rho, rhoold);
                cke(cublasDnrm2(bl, n, r, 1, norm), "norm r");
                double rn = 0;
                ck(cudaMemcpy(&rn, norm, sizeof(double), cudaMemcpyDeviceToHost), "norm r read");
                if(rn < tol) {
                    ++it;
                    break;
                }
            }
            max_iters = std::max(max_iters, (double)it);
            ck(cudaMemcpy(xh.data() + (size_t)c * n, x, sz, cudaMemcpyDeviceToHost), "x copy");
        }
        ck(cudaEventRecord(e1), "record");
        ck(cudaEventSynchronize(e1), "sync");
        elapsed(e0, e1, t.solve_ms);
        t.iters = max_iters;
        t.res = residual(n, klu, lda, a, xh.data(), n, b, nrhs);
        for(double* q : {r, rhat, p, ph, v, ss, sh, tv, x, bb, dinv}) cudaFree(q);
        for(double* q : {rho, rhoold, alpha, omega, beta, num, den, norm}) cudaFree(q);
        cublasDestroy(bl);
        cudaFree(dband);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        return t;
    }

} // namespace

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 10000;
    const int klu = argc > 2 ? std::atoi(argv[2]) : n / 100;
    const int nrhs = argc > 3 ? std::atoi(argv[3]) : 1;
    const bool dominant = argc < 5 || std::atoi(argv[4]) != 0;
    const int lda = 2 * klu + 1;
    const char* env = std::getenv("DUEL_MKL_THREADS");
    mkl_set_num_threads(env ? std::atoi(env) : 8);
    std::vector<double> a, b;
    make_band(n, klu, dominant, a, b, nrhs, lda);
    std::vector<float> af(a.size());
    for(size_t i = 0; i < a.size(); ++i) af[i] = (float)a[i];

    const auto g = run_gspike(n, klu, nrhs, a, b, lda);
    const auto gd = run_gspike_double(n, klu, nrhs, a, b, lda);
    const auto md = run_mkl<double>(n, klu, nrhs, a, b, lda);
    const auto ms = run_mkl<float>(n, klu, nrhs, af, b, lda);
    const auto sp = run_sparse(n, klu, nrhs, a, b, lda);
    const auto bs = run_band_sparse(n, klu, nrhs, a, b, lda);
    std::printf("n=%d klu=%d nrhs=%d %s (MKL threads=%d)\n", n, klu, nrhs, dominant ? "dominant" : "non-dominant", mkl_get_max_threads());
    const auto row = [](const char* name, const result& r) {
        std::printf("  %-16s setup=%8.2f solve=%8.2f total=%8.2f res=%.2e iters=%.0f\n", name, r.setup_ms, r.solve_ms, r.setup_ms + r.solve_ms, r.res, r.iters);
    };
    row("gspike mixed", g);
    row("gspike double", gd);
    row("MKL double band", md);
    row("MKL float band", ms);
    row("cuSPARSE BiCGStab", sp);
    row("band BiCGStab", bs);
    return 0;
}
