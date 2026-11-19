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
#include <cusolverDn.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gspike {
    namespace {

        // Hard ceiling on the partition count; the device's SM count and the
        // interface budgets in the solver constructor clamp it further.
        constexpr int kMaxParts = 128;
        // Ceiling on the partition floor below.  It only binds for wide bands,
        // where the 2*klu-wide interface and the spike phase grow faster than
        // the extra partitions save.
        constexpr int kFloorParts = 18;
        constexpr int kInfoNone = 0x7f7f7f7f;

        // Dynamic shared memory per CUDA block that needs no opt-in.
        constexpr size_t kSmemBudget = 48u << 10;
        // Dense interface systems are factored with cuSOLVER, larger ones stay
        // banded and use the band LU.  cuSOLVER's dense LU is O(m^3) against
        // O(m*klr^2) for the band path, so dense only pays for the sizes in
        // num<>::dense_ifc_max_elems.  The banded alternative is capped by
        // its storage, m * ldaR * sizeof(T).
        constexpr size_t kBandIfcBytes = 256u << 20;
        // The banded interface factorization is sequential in m and costs
        // roughly num<>::band_ifc_ns * m * klr^1.5.  Do not route the automatic
        // partition count into a banded system that would take longer than this.
        constexpr double kBandIfcMs = 25.0;
        // Internal switch (CMake option GSPIKE_LEGACY_SOLVE): keep the original
        // one-thread-per-RHS band solve available for comparison runs.  The warp
        // kernel is the default.
#ifdef GSPIKE_LEGACY_SOLVE
        constexpr bool kLegacyBandSolve = true;
#else
        constexpr bool kLegacyBandSolve = false;
#endif

        void ck(const cudaError_t e) {
            if(e != cudaSuccess)
                throw std::runtime_error(std::string("gspike: ") + cudaGetErrorString(e));
        }
        void ckb(const cublasStatus_t s) {
            if(s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("gspike: cuBLAS failure");
        }
        void cks(const cusolverStatus_t s) {
            if(s != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("gspike: cuSOLVER failure");
        }

        template<class T>
        struct num;
        template<>
        struct num<float> {
            static constexpr double rtol0 = 1e-5; // default refinement target
            // Above m ~ 3600 the dense interface costs more than the extra
            // partitions it would buy, and the per-partition band work no
            // longer shrinks enough to pay for it.
            static constexpr size_t dense_ifc_max_elems = static_cast<size_t>(3584) * 3584;
            // Automatic partition count, see auto_parts().
            static constexpr double parts_c = 1.65, parts_n = 0.42, parts_klu = 0.5;
            // Measured factor cost of the two interface paths: dense getrf
            // (ns, fitted over m = 1..5k) and the serial banded LU (ns per
            // m * klr^1.5).  The cheaper one is used.
            static constexpr double dense_ifc_a = 386.5, dense_ifc_b = 1.222;
            static constexpr double band_ifc_ns = 15.1;
            static cublasStatus_t nrm2(const cublasHandle_t h, const int n, const float* x, float* r) {
                return cublasSnrm2(h, n, x, 1, r);
            }
            static cusolverStatus_t getrf_buf(cusolverDnHandle_t h, const int m, float* A, const int lda, int* lw) {
                return cusolverDnSgetrf_bufferSize(h, m, m, A, lda, lw);
            }
            static cusolverStatus_t getrf(cusolverDnHandle_t h, const int m, float* A, const int lda, float* w, int* ipiv, int* info) {
                return cusolverDnSgetrf(h, m, m, A, lda, w, ipiv, info);
            }
            static cusolverStatus_t getrs(cusolverDnHandle_t h, const int m, const int nrhs, const float* A, const int lda, const int* ipiv, float* B, const int ldb, int* info) {
                return cusolverDnSgetrs(h, CUBLAS_OP_N, m, nrhs, A, lda, ipiv, B, ldb, info);
            }
        };
        template<>
        struct num<double> {
            static constexpr double rtol0 = 1e-12;
            // Above m ~ 3000 the O(m^3) FP64 dense factorization no longer pays
            // for the partitions it enables (measured on an L40).
            static constexpr size_t dense_ifc_max_elems = static_cast<size_t>(3072) * 3072;
            static constexpr double parts_c = 1.15, parts_n = 0.42, parts_klu = 0.5;
            static constexpr double dense_ifc_a = 6.87, dense_ifc_b = 1.910;
            static constexpr double band_ifc_ns = 7.9;
            static cublasStatus_t nrm2(const cublasHandle_t h, const int n, const double* x, double* r) {
                return cublasDnrm2(h, n, x, 1, r);
            }
            static cusolverStatus_t getrf_buf(cusolverDnHandle_t h, const int m, double* A, const int lda, int* lw) {
                return cusolverDnDgetrf_bufferSize(h, m, m, A, lda, lw);
            }
            static cusolverStatus_t getrf(cusolverDnHandle_t h, const int m, double* A, const int lda, double* w, int* ipiv, int* info) {
                return cusolverDnDgetrf(h, m, m, A, lda, w, ipiv, info);
            }
            static cusolverStatus_t getrs(cusolverDnHandle_t h, const int m, const int nrhs, const double* A, const int lda, const int* ipiv, double* B, const int ldb, int* info) {
                return cusolverDnDgetrs(h, CUBLAS_OP_N, m, nrhs, A, lda, ipiv, B, ldb, info);
            }
        };

        // Automatic partition count: c * n^0.42 / klu^0.5.  The per-partition
        // band work falls with p while the interface system (2*klu*(p-1) wide)
        // grows, and this is the power law that reproduced the measured
        // optimum on an L40 over n = 20k..500k and klu = 10..65.  The result is
        // clamped by kMaxParts, the SM count, the rows per partition, and the
        // interface budgets in the solver constructor.  FP64 gets a smaller
        // constant because its interface factorization costs several times more.
        template<class T>
        index_t auto_parts(const index_t n, const index_t klu) {
            if(klu <= 0) return 1;
            const double p = num<T>::parts_c * std::pow(static_cast<double>(n), num<T>::parts_n) / std::pow(static_cast<double>(klu), num<T>::parts_klu);
            return static_cast<index_t>(std::llround(p));
        }

        // Predicted interface factorization time (ns) for the two paths.
        template<class T>
        double band_interface_ns(const index_t m, const index_t klr) {
            return num<T>::band_ifc_ns * static_cast<double>(m) * std::pow(static_cast<double>(klr), 1.5);
        }
        template<class T>
        double dense_interface_ns(const index_t m) {
            return num<T>::dense_ifc_a * std::pow(static_cast<double>(m), num<T>::dense_ifc_b);
        }

        // Dense interface path: only for sizes within dense_ifc_max_elems and
        // when cuSOLVER's O(m^3) LU is predicted to beat the banded LU.
        template<class T>
        bool use_dense_interface(const index_t m, const index_t klr) {
            if(m <= 0 || static_cast<size_t>(m) * static_cast<size_t>(m) > num<T>::dense_ifc_max_elems) return false;
            return dense_interface_ns<T>(m) <= band_interface_ns<T>(m, klr);
        }

        // SM count of the active device; the factor kernels run one partition
        // per CTA, so more partitions than SMs only adds interface work.
        int device_sms() {
            int dev = 0;
            ck(cudaGetDevice(&dev));
            int sms = 0;
            ck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
            return std::max(sms, 1);
        }

        // Launch geometry for the column-wise copies in kernels.cuh: the
        // columns spread over grid.x and the rows over grid.y, eight block-wide
        // chunks per block.
        dim3 copy_grid(const index_t n, const index_t nrhs, const int block) {
            const index_t per = static_cast<index_t>(block) * 8;
            const index_t rows = (n + per - 1) / per;
            return {static_cast<unsigned>(nrhs), static_cast<unsigned>(std::min<index_t>(rows, 65535))};
        }

        // 2-norm of an index_t-long vector.  cuBLAS only counts with int, so
        // long vectors (64-bit builds) are reduced in chunks and combined with
        // hypot; for n <= INT_MAX this is a single cuBLAS call.
        template<class T>
        T nrm2(const cublasHandle_t h, const index_t n, const T* x) {
            T acc = T(0);
            bool first = true;
            for(index_t off = 0; off < n;) {
                const int len = std::min<index_t>(n - off, INT_MAX);
                T part = T(0);
                ckb(num<T>::nrm2(h, len, x + off, &part));
                acc = first ? part : std::hypot(acc, part);
                first = false;
                off += len;
            }
            return acc;
        }

        // Band LU launch parameters.  The pivot row is staged in shared memory
        // when it fits the no-opt-in budget, and wide bands get more warps for
        // the trailing update (one warp per column).
        template<class T>
        size_t lu_shmem(const index_t kl) {
            const size_t need = static_cast<size_t>(kl) * sizeof(T);
            return need <= kSmemBudget ? need : 0;
        }
        int lu_block(const int block, const index_t kl, const index_t ku) {
            // One warp per column in the trailing update: wide bands fill a full
            // block, narrow ones keep the requested target size.
            if(kl + ku > 32) return 1024;
            return std::min(block, 1024);
        }

        // Launch the band solve: one warp per right-hand side column.  The RHS
        // column is staged in shared memory when a warp's copy fits in the
        // 48 KiB dynamic budget that needs no opt-in; otherwise the warp works
        // directly in global memory.  nmax is the largest partition size of the
        // descriptor array, nparts the number of partitions (grid.x).
        template<class T>
        void band_solve(const dk::pdesc<T, index_t>* pd, const index_t lda, const index_t kl, const index_t ku, T* rhs, const index_t ldr, const index_t ncols, const index_t nmax, const int nparts, const int block) {
            if(ncols <= 0 || nparts <= 0) return;
            const unsigned parts = static_cast<unsigned>(nparts);
            if constexpr(kLegacyBandSolve) {
                const auto gc = static_cast<unsigned>((ncols + block - 1) / block);
                dk::band_solve<T, index_t><<<dim3(parts, gc), static_cast<unsigned>(block)>>>(pd, lda, kl, ku, rhs, ldr, ncols);
                return;
            }
            const int wmax = std::max(1, std::min(block, 1024) / 32);
            const size_t per = sizeof(T) * static_cast<size_t>(nmax);
            const size_t fit = (per > 0 && per <= kSmemBudget) ? kSmemBudget / per : 0;
            const int ncol = std::min<int>(wmax, static_cast<int>(std::min<index_t>(ncols, INT_MAX)));
            // Staging pays off only when every warp of the CTA gets its own
            // staged column; trading warps for shared memory costs occupancy,
            // which matters because a CTA per column then needs the full band
            // in L2.
            const int warps = static_cast<int>(std::min<size_t>(fit, static_cast<size_t>(ncol)));
            if(warps == ncol && ncol >= 2) {
                const dim3 grid(parts, static_cast<unsigned>((ncols + warps - 1) / warps));
                dk::band_solve_warp<T, index_t, true><<<grid, static_cast<unsigned>(32 * warps), static_cast<size_t>(warps) * per>>>(pd, lda, kl, ku, rhs, ldr, ncols, nmax);
            }
            else {
                const dim3 grid(parts, static_cast<unsigned>((ncols + ncol - 1) / ncol));
                dk::band_solve_warp<T, index_t, false><<<grid, static_cast<unsigned>(32 * ncol)>>>(pd, lda, kl, ku, rhs, ldr, ncols, nmax);
            }
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
        cusolverDnHandle_t cs = nullptr;
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
        int* d_ipiv = nullptr; // dense interface pivots (cuSOLVER)
        T* d_csw = nullptr;    // dense interface LU workspace
        int* d_info = nullptr;
        index_t m = 0, klr = 0, ldaR = 0;
        index_t nmax = 0;       // largest partition, sizes dynamic shared memory
        bool use_dense = false; // interface path chosen by the constructor
        bool dense_ifc = false; // dense_interface() succeeded

        T* arena = nullptr;
        size_t arena_n = 0;
        index_t arena_nrhs = 0;

        // Factorization workspace: the partition factors, pivots, spikes, the
        // interface band and the spike RHS are carved out of one allocation.
        // Every size is fixed by the constructor, so it is never grown.
        T* fbase = nullptr;
        size_t fcap = 0;  // bytes
        size_t fused = 0; // bytes handed out
        index_t spike_chunk = 0;

        void* take_bytes(size_t bytes) {
            constexpr size_t step = 256;
            bytes = (bytes + step - 1) / step * step;
            if(fused + bytes > fcap) throw std::runtime_error("gspike: factor workspace overflow");
            void* r = reinterpret_cast<char*>(fbase) + fused;
            fused += bytes;
            return r;
        }
        template<class U>
        U* take(size_t count) {
            return static_cast<U*>(take_bytes(count * sizeof(U)));
        }

        ~impl() {
            // The dense interface factors are the only heap buffers that are
            // not part of the workspace.
            if(dense_ifc) {
                if(d_R) cudaFree(d_R);
                if(d_ipiv) cudaFree(d_ipiv);
                if(d_csw) cudaFree(d_csw);
            }
            if(fbase) cudaFree(fbase);
            if(arena) cudaFree(arena);
            if(cs) cusolverDnDestroy(cs);
            if(h) cublasDestroy(h);
        }

        void factor();

        // Interface factorization: the path the constructor predicted to be
        // cheaper (see use_dense_interface()).  The dense path returns false
        // when it was not selected or when cuSOLVER reports a zero pivot, in
        // which case the banded path takes over and reproduces the historical
        // pivot perturbation.
        bool dense_interface();
        void band_interface();

        // Interface solve with the dense factors.
        void dense_ifc_solve(T* rs, const index_t nrhs) const {
            for(index_t c0 = 0; c0 < nrhs;) {
                const int nc = std::min<index_t>(nrhs - c0, INT_MAX);
                cks(num<T>::getrs(cs, static_cast<int>(m), nc, d_R, static_cast<int>(m), d_ipiv, rs + c0 * m, static_cast<int>(m), d_info));
                c0 += nc;
            }
        }

        void alloc_arena(const index_t nrhs) {
            const size_t need = static_cast<size_t>(4) * n * nrhs + static_cast<size_t>(2) * klu * p * nrhs + static_cast<size_t>(m) * nrhs;
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
            const auto gc = static_cast<unsigned>((nrhs + block - 1) / block);
            const dim3 gpar(p, gc);
            const dim3 gred(p > 1 ? p - 1 : 1, gc);

            if(p == 1) {
                band_solve<T>(d_pd, lda_f, kl, ku, X, ldx, nrhs, n, 1, block);
                ck(cudaGetLastError());
                return;
            }
            dk::copy_mat<T, index_t><<<copy_grid(n, nrhs, block), block>>>(n, nrhs, X, ldx, g, n);
            band_solve<T>(d_pd, lda_f, kl, ku, g, n, nrhs, nmax, p, block);
            dk::extract_corners<T, index_t><<<gpar, block>>>(d_pd, g, n, klu, nrhs, grj);
            dk::reduced_rhs<T, index_t><<<gred, block>>>(grj, m, klu, rs, nrhs);
            if(dense_ifc) dense_ifc_solve(rs, nrhs);
            else band_solve<T>(d_rp, ldaR, klr, klr, rs, m, nrhs, m, 1, block);
            dk::correct_rhs<T, index_t><<<gpar, block>>>(d_pd, d_bc, d_cc, rs, klu, m, p, X, ldx, nrhs);
            band_solve<T>(d_pd, lda_f, kl, ku, X, ldx, nrhs, nmax, p, block);
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
            size[i] = base + (i < rem ? 1 : 0);
            row0[i] = (i == 0) ? 0 : row0[i - 1] + size[i - 1];
        }

        // One allocation covers every buffer below, including the spike RHS.
        // The interface path is known, but a failed dense factorization falls
        // back to the band one, so the band storage is always reserved.
        const index_t ncol = 2 * klu;
        if(p > 1) {
            q->spike_chunk = ncol;
            while(q->spike_chunk > 1 && static_cast<size_t>(n) * static_cast<size_t>(q->spike_chunk) * sizeof(T) > (512ULL << 20)) q->spike_chunk /= 2;
        }
        if(!q->fbase) {
            size_t need = 0;
            need += static_cast<size_t>(lda_f) * n * sizeof(T);             // d_ab
            need += static_cast<size_t>(n) * sizeof(index_t);               // d_piv
            need += static_cast<size_t>(p) * sizeof(dk::pdesc<T, index_t>); // d_pd
            need += static_cast<size_t>(p) * 4 * klu * klu * sizeof(T);     // d_sp
            need += static_cast<size_t>(p) * klu * klu * sizeof(T) * 2;     // d_bc, d_cc
            need += static_cast<size_t>(p) * sizeof(T*);                    // d_spp
            need += sizeof(int);                                            // d_info
            need += static_cast<size_t>(n) * static_cast<size_t>(q->spike_chunk) * sizeof(T);
            if(p > 1) {
                need += static_cast<size_t>(q->m) * q->ldaR * sizeof(T); // d_R (band)
                need += static_cast<size_t>(q->m) * sizeof(index_t);     // d_pivR
                need += sizeof(dk::pdesc<T, index_t>);                   // d_rp
            }
            // Every carved block is rounded up to 256 bytes, so leave the
            // padding of the (at most 16) blocks as slack.
            need += 16 * 256;
            ck(cudaMalloc(&q->fbase, need));
            q->fcap = need;
        }
        q->fused = 0;

        q->d_ab = q->take<T>(static_cast<size_t>(lda_f) * n);
        q->d_piv = q->take<index_t>(static_cast<size_t>(n));
        q->d_pd = q->take<dk::pdesc<T, index_t>>(static_cast<size_t>(p));
        q->d_sp = q->take<T>(static_cast<size_t>(p) * 4 * klu * klu);
        q->d_bc = q->take<T>(static_cast<size_t>(p) * klu * klu);
        q->d_cc = q->take<T>(static_cast<size_t>(p) * klu * klu);
        q->d_spp = q->take<T*>(static_cast<size_t>(p));
        q->d_info = q->take<int>(1);
        T* rhs = q->take<T>(static_cast<size_t>(n) * static_cast<size_t>(q->spike_chunk));
        ck(cudaMemset(q->d_info, 0x7f, sizeof(int)));
        ck(cudaMemset(q->d_sp, 0, static_cast<size_t>(p) * 4 * klu * klu * sizeof(T)));

        std::vector<dk::pdesc<T, index_t>> hd(p);
        std::vector<T*> hspp(p);
        for(int i = 0; i < p; ++i) {
            hd[i].ab = q->d_ab + lda_f * row0[i];
            hd[i].piv = q->d_piv + row0[i];
            hd[i].row0 = row0[i];
            hd[i].size = size[i];
            hspp[i] = q->d_sp + i * 4 * klu * klu;
        }
        ck(cudaMemcpy(q->d_pd, hd.data(), p * sizeof(dk::pdesc<T, index_t>), cudaMemcpyHostToDevice));
        ck(cudaMemcpy(q->d_spp, hspp.data(), p * sizeof(T*), cudaMemcpyHostToDevice));

        dk::make_blocks<T, index_t><<<p, block>>>(q->d_pd, q->A, n, q->lda, lda_f, ku, kl, klu, q->d_bc, q->d_cc);
        dk::band_lu<T, index_t><<<p, lu_block(block, kl, ku), lu_shmem<T>(kl)>>>(q->d_pd, lda_f, kl, ku, q->d_info);
        ck(cudaGetLastError());

        if(p == 1) return;

        // spikes: A_ii [C_i | B_i], solved in chunks of columns (2*klu in total).
        // All spike columns are solved at once (one thread each) as long as the
        // temporary buffer stays within budget.
        const auto nb = static_cast<unsigned>((2 * klu * klu + block - 1) / block);
        for(index_t c0 = 0; c0 < ncol; c0 += q->spike_chunk) {
            const index_t nc = std::min<index_t>(q->spike_chunk, ncol - c0);
            dk::spike_rhs<T, index_t><<<dim3(p, static_cast<unsigned>(nc)), block>>>(q->d_pd, q->d_bc, q->d_cc, klu, rhs, n, c0, ncol);
            band_solve<T>(q->d_pd, lda_f, kl, ku, rhs, n, nc, q->nmax, p, block);
            dk::spike_extract<T, index_t><<<dim3(p, nb), block>>>(q->d_pd, rhs, n, klu, c0, nc, q->d_sp);
            ck(cudaGetLastError());
        }
    }

    ////////////////////////////////////////////////////////////////////////////////

    // Dense interface factorization with cuSOLVER.  Used while m^2 stays inside
    // num<T>::dense_ifc_max_elems; above that (or on a zero pivot, where the
    // banded path keeps the historical perturbation behaviour) the caller falls
    // back to band_interface().
    template<class T>
    bool solver<T>::impl::dense_interface() {
        impl* q = this;
        const index_t m = q->m;
        if(!q->use_dense) return false;
        const size_t elems = static_cast<size_t>(m) * static_cast<size_t>(m);

        T* R = nullptr;
        int* ipiv = nullptr;
        T* work = nullptr;
        int wsize = 0;
        ck(cudaMalloc(&R, elems * sizeof(T)));
        ck(cudaMalloc(&ipiv, static_cast<size_t>(m) * sizeof(int)));
        cks(num<T>::getrf_buf(q->cs, static_cast<int>(m), R, static_cast<int>(m), &wsize));
        ck(cudaMalloc(&work, static_cast<size_t>(std::max(wsize, 1)) * sizeof(T)));
        ck(cudaMemset(R, 0, elems * sizeof(T)));
        dk::reduced_diag<T, index_t, true><<<q->p - 1, q->block>>>(q->d_spp, q->p, q->klu, q->klr, m, R);
        ck(cudaGetLastError());
        ck(cudaMemset(q->d_info, 0, sizeof(int)));
        cks(num<T>::getrf(q->cs, static_cast<int>(m), R, static_cast<int>(m), work, ipiv, q->d_info));
        ck(cudaDeviceSynchronize());
        int info = 0;
        ck(cudaMemcpy(&info, q->d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if(info != 0) { // zero pivot in the interface system: let the band path perturb it
            ck(cudaFree(R));
            ck(cudaFree(ipiv));
            ck(cudaFree(work));
            return false;
        }
        q->d_R = R;
        q->d_ipiv = ipiv;
        q->d_csw = work;
        q->dense_ifc = true;
        return true;
    }

    // Banded interface factorization into the private 3*klu-wide band storage;
    // this is also the fallback when the dense system is too large.
    template<class T>
    void solver<T>::impl::band_interface() {
        impl* q = this;
        q->d_R = q->take<T>(static_cast<size_t>(q->m) * q->ldaR);
        q->d_pivR = q->take<index_t>(static_cast<size_t>(q->m));
        q->d_rp = q->take<dk::pdesc<T, index_t>>(1);
        ck(cudaMemset(q->d_R, 0, static_cast<size_t>(q->ldaR) * q->m * sizeof(T)));

        dk::reduced_diag<T, index_t, false><<<q->p - 1, q->block>>>(q->d_spp, q->p, q->klu, q->klr, q->ldaR, q->d_R);
        ck(cudaGetLastError());

        dk::pdesc<T, index_t> rp{};
        rp.ab = q->d_R;
        rp.piv = q->d_pivR;
        rp.row0 = 0;
        rp.size = q->m;
        ck(cudaMemcpy(q->d_rp, &rp, sizeof(rp), cudaMemcpyHostToDevice));
        ck(cudaMemset(q->d_info, 0x7f, sizeof(int)));
        dk::band_lu<T, index_t><<<1, lu_block(q->block, q->klr, q->klr), lu_shmem<T>(q->klr)>>>(q->d_rp, q->ldaR, q->klr, q->klr, q->d_info);
        ck(cudaGetLastError());
        ck(cudaDeviceSynchronize());
        int info = 0;
        ck(cudaMemcpy(&info, q->d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if(info != kInfoNone && q->info0 == 0) {
            // row index in the reduced system; report it offset past the matrix
            q->info0 = (info > INT_MAX - q->n) ? INT_MAX : static_cast<int>(std::min<index_t>(q->n + info, INT_MAX));
        }
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
        const bool autop = p <= 0;
        if(autop) {
            // The power law is fitted to the factor-time optimum.  Do not fall
            // below the historical "at least 16*klu + 32 rows per partition"
            // rule either: larger partitions make the warp band solve re-read
            // the band once per RHS tile, which costs more than the extra
            // partitions save once the RHS count grows.
            const index_t floor_p = (q->klu == 0) ? 1 : std::min<index_t>(kFloorParts, std::max<index_t>(1, n / (16 * q->klu + 32)));
            p = std::max<index_t>(auto_parts<T>(n, q->klu), floor_p);
            p = std::min<index_t>(p, std::min<index_t>(kMaxParts, device_sms()));
        }
        p = std::min(std::max<index_t>(p, 1), pmax);
        // Keep the interface system within the budget of the path that will
        // factor it.  An explicit partition count is only subject to the memory
        // caps; the automatic one also refuses a banded interface whose
        // factorization would eat into the gain from the extra partitions.
        while(p > 1) {
            const index_t m = 2 * q->klu * (p - 1);
            const index_t ldaR = 9 * q->klu + 1;
            const index_t klr = 3 * q->klu;
            const bool band_fits = static_cast<size_t>(m) * ldaR * sizeof(T) <= kBandIfcBytes;
            const bool band_cheap = !autop || band_interface_ns<T>(m, klr) <= kBandIfcMs * 1e6;
            if(use_dense_interface<T>(m, klr) || (band_fits && band_cheap)) break;
            --p;
        }
        q->p = p;
        q->m = 2 * q->klu * (q->p - 1);
        q->klr = 3 * q->klu;
        q->ldaR = 3 * q->klr + 1;
        q->nmax = (n + p - 1) / p;
        q->use_dense = use_dense_interface<T>(q->m, q->klr);

        q->factor();
        ck(cudaDeviceSynchronize());

        int info = 0;
        ck(cudaMemcpy(&info, q->d_info, sizeof(int), cudaMemcpyDeviceToHost));
        q->info0 = (info == kInfoNone) ? 0 : info;

        if(p > 1) {
            // Only the dense interface needs cuSOLVER; its handle costs a
            // couple of milliseconds to create, so do not pay for it on the
            // banded path.
            if(q->use_dense) cks(cusolverDnCreate(&q->cs));
            if(!q->dense_interface()) q->band_interface();
        }

        if(o.verbose) {
            std::printf("[gspike] n=%lld kl=%lld ku=%lld parts=%d lda_f=%lld reduced_dim=%lld iface=%s (%d-bit index)\n", static_cast<long long>(n), static_cast<long long>(kl), static_cast<long long>(ku), q->p, static_cast<long long>(q->lda_f), static_cast<long long>(q->m), q->dense_ifc ? "dense" : (p > 1 ? "band" : "none"), static_cast<int>(8 * sizeof(index_t)));
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

        // The residual (and therefore the saved right-hand side) is only
        // needed for refinement, for stats, or for the verbose line.
        const bool need_res = (st != nullptr) || (q->o.max_refine > 0) || q->o.verbose;
        const dim3 cg = copy_grid(n, nrhs, block);
        if(need_res) dk::copy_mat_dup<T, index_t><<<cg, block>>>(n, B, ldb, x, bsave, n);
        else dk::copy_mat<T, index_t><<<cg, block>>>(n, nrhs, B, ldb, x, n);
        ck(cudaGetLastError());

        q->solve_inplace(x, g, grj, rs, nrhs, n);

        const double tol = (q->o.rtol > 0) ? q->o.rtol : num<T>::rtol0;
        double resn = 0.0, bn = 1.0;
        int iters = 0;

        auto residual = [&]() {
            dk::band_matvec<T, index_t><<<dim3(static_cast<unsigned>(nrhs), static_cast<unsigned>((n + block - 1) / block)), block>>>(
                n, q->kl, q->ku, q->A, q->lda, x, n, res, n, nrhs
            );
            dk::bsub<T, index_t><<<static_cast<unsigned>((tot + block - 1) / block), block>>>(tot, bsave, res);
            ck(cudaGetLastError());
            const T r2 = nrm2<T>(q->h, tot, res);
            const T b2 = nrm2<T>(q->h, tot, bsave);
            resn = static_cast<double>(r2);
            bn = static_cast<double>(b2);
        };

        if(need_res) residual();
        double prev = resn;
        while(iters < q->o.max_refine && resn > tol * bn) {
            q->solve_inplace(res, g, grj, rs, nrhs, n); // res <- A^-1 res
            dk::axpby<T, index_t><<<static_cast<unsigned>((tot + block - 1) / block), block>>>(tot, T(1), res, T(1), x);
            ++iters;
            residual();
            if(!(resn < prev)) break; // no further improvement
            prev = resn;
        }

        dk::copy_mat<T, index_t><<<cg, block>>>(n, nrhs, x, n, B, ldb);
        ck(cudaGetLastError());

        const int info = q->info0; // factorization status only; refinement quality is in stats
        if(st) {
            st->parts = q->p;
            st->refine = iters;
            st->residual = (bn > 0) ? resn / bn : resn;
        }
        if(q->o.verbose)
            std::printf("[gspike] solve nrhs=%lld refine=%d relres=%.3e\n", static_cast<long long>(nrhs), iters, (bn > 0) ? resn / bn : resn);
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
    template<class T>
    bool solver<T>::dense_interface() const { return p_->dense_ifc; }

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
            const size_t tot = static_cast<size_t>(n) * static_cast<size_t>(nrhs);
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
        if(static_cast<size_t>(lda) * n > static_cast<size_t>(std::numeric_limits<index_t>::max()))
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
        ck(cudaMalloc(&q->dA_f, static_cast<size_t>(na) * sizeof(float)));
        dk::cast_copy<float, double, index_t><<<static_cast<unsigned>((na + q->block - 1) / q->block), q->block>>>(na, A, q->dA_f);
        ck(cudaGetLastError());

        opts fo = o;
        fo.max_refine = 0; // solve_direct() only; no float refinement
        q->fs = new solver<float>(n, kl, ku, q->dA_f, lda, fo);

        if(o.verbose)
            std::printf("[gspike] mixed: n=%lld kl=%lld ku=%lld parts=%d float iters=%d\n", static_cast<long long>(n), static_cast<long long>(kl), static_cast<long long>(ku), q->fs->parts(), q->iters);
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
        const auto gc = static_cast<unsigned>((tot + block - 1) / block);
        const auto gr = static_cast<unsigned>((n + block - 1) / block);

        double* x = q->arena;
        double* b = x + tot;
        double* r = b + tot;
        float* rf = q->farena;

        const dim3 cg = copy_grid(n, nrhs, block);
        dk::copy_mat_dup<double, index_t><<<cg, block>>>(n, B, ldb, b, r, n);
        ck(cudaMemset(x, 0, static_cast<size_t>(tot) * sizeof(double)));
        ck(cudaGetLastError());

        const auto bn = nrm2<double>(q->h, tot, b);
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

            dk::band_matvec<double, index_t><<<dim3(static_cast<unsigned>(nrhs), gr), block>>>(n, q->kl, q->ku, q->A, q->lda, x, n, r, n, nrhs);
            dk::bsub<double, index_t><<<gc, block>>>(tot, b, r);
            ck(cudaGetLastError());
            resn = nrm2<double>(q->h, tot, r);
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

        dk::copy_mat<double, index_t><<<cg, block>>>(n, nrhs, x, n, B, ldb);
        ck(cudaGetLastError());

        if(st) {
            st->parts = q->fs->parts();
            st->refine = iters;
            st->residual = (bn > 0) ? resn / bn : resn;
        }
        if(q->o.verbose)
            std::printf("[gspike] mixed solve nrhs=%lld iters=%d relres=%.3e\n", static_cast<long long>(nrhs), iters, (bn > 0) ? resn / bn : resn);
        return q->fs->info();
    }

    index_t mixed_solver::n() const { return p_->n; }
    index_t mixed_solver::kl() const { return p_->kl; }
    index_t mixed_solver::ku() const { return p_->ku; }
    int mixed_solver::parts() const { return p_->fs->parts(); }
    bool mixed_solver::dense_interface() const { return p_->fs->dense_interface(); }

    int spike_gbsv_mixed(const index_t n, const index_t kl, const index_t ku, const index_t nrhs, const double* A, const index_t lda, double* B, const index_t ldb, const opts& o, stats* st) {
        try {
            const mixed_solver s(n, kl, ku, A, lda, o);
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
