// Device kernels for the SPIKE banded solver.
#pragma once

#include <cuda_runtime.h>

namespace gspike::dk {

    // Per-subsystem description (one band LU factorization block).
    template<class T>
    struct pdesc {
        T* ab;    // band factors: (2*kl+ku+1) x size, column major,
                  // a(i,j) at ab[(kl+ku+i-j) + j*lda]
        int* piv; // pivot row (local index) chosen at each column
        int row0; // first global row of the block (RHS buffers indexed by global row)
        int size; // dimension of the block
    };

    // ---------------------------------------------------------------------------
    // Scalar band LU.  Needed only when a column actually needs a row swap:
    // the blocked kernel below defers trailing updates, which does not
    // reproduce this kernel's interleaved pivot/elimination order.  Diagonally
    // dominant bands (the performance cases) never pivot and use the blocked
    // path; the fallback keeps the general case correct.
    // ---------------------------------------------------------------------------
    template<class T>
    __global__ void band_lu_scalar(const pdesc<T>* __restrict__ pd, int lda, int kl, int ku, int* __restrict__ info) {
        const pdesc<T> d = pd[blockIdx.x];
        const int n = d.size;
        const int off = kl + ku;
        const int tid = threadIdx.x;
        const int nt = blockDim.x;
        const int tx = tid & 31;
        const int ty = tid >> 5;
        const int nwarps = (nt + 31) / 32;
        const int kmax = min(kl + ku, n - 1);
        T* __restrict__ ab = d.ab;
        int* __restrict__ piv = d.piv;
        int bad = 0;

        for(int j = 0; j < n; ++j) {
            const int iend = min(j + kl, n - 1);
            T* cj = ab + (long long)j * lda - j + off;

            if(tid < 32) {
                int pv = j;
                T bv = fabs(cj[j]);
                for(int i = j + 1 + tid; i <= iend; i += 32) {
                    const T v = fabs(cj[i]);
                    if(v > bv) {
                        bv = v;
                        pv = i;
                    }
                }
                for(int s = 16; s > 0; s >>= 1) {
                    const T ov = __shfl_down_sync(0xffffffffu, bv, s);
                    const int op = __shfl_down_sync(0xffffffffu, pv, s);
                    if(ov > bv) {
                        bv = ov;
                        pv = op;
                    }
                }
                if(tid == 0) {
                    if(!(bv > T(0))) {
                        if(bad == 0) bad = d.row0 + j + 1;
                        cj[pv] = T(1);
                    }
                    piv[j] = pv;
                }
            }
            __syncthreads();
            const int pv = piv[j];
            const int kend = min(n - 1, j + kmax);

            if(pv != j) {
                for(int k = j + tid; k <= kend; k += nt) {
                    T* ck = ab + (long long)k * lda - k + off;
                    const T t = ck[j];
                    ck[j] = ck[pv];
                    ck[pv] = t;
                }
                __syncthreads();
            }
            const T pval = cj[j];
            for(int i = j + 1 + tid; i <= iend; i += nt) cj[i] /= pval;
            __syncthreads();
            for(int k = j + 1 + ty; k <= kend; k += nwarps) {
                T* ck = ab + (long long)k * lda - k + off;
                const T u = ck[j];
#pragma unroll 16
                for(int i = j + 1 + tx; i <= iend; i += 32) ck[i] -= cj[i] * u;
            }
            __syncthreads();
        }
        if(bad) atomicMin(info, bad);
    }

    // ---------------------------------------------------------------------------
    // Band LU with partial pivoting, blocked over panels of NB columns.
    //
    // Factor one panel per block (NB columns of sequential pivoting), then
    // apply the whole panel to the trailing band with a tiled rank-NB update.
    // A scalar, one-column-at-a-time band LU moves a whole kl x (kl+ku) tile
    // through L1 for a single FMA per element; the blocked update amortizes
    // those loads and stores over NB FMAs and spreads the work over many SMs.
    // ---------------------------------------------------------------------------
    template<class T, int NB>
    __global__ void band_lu_panel(const pdesc<T>* __restrict__ pd, int lda, int kl, int ku, int j0, int nb, int* __restrict__ info, int* __restrict__ pivoted) {
        const pdesc<T> d = pd[blockIdx.x];
        if(j0 >= d.size) return;
        const int n = d.size;
        const int off = kl + ku;
        const int tid = threadIdx.x;
        const int nt = blockDim.x;
        const int tx = tid & 31;
        const int ty = tid >> 5;
        const int nwarps = (nt + 31) / 32;
        const int kmax = min(kl + ku, n - 1); // pivoting fills U up to kl+ku
        T* __restrict__ ab = d.ab;
        int* __restrict__ piv = d.piv;
        int bad = 0;
        const int jend = min(n, j0 + nb);

        for(int j = j0; j < jend; ++j) {
            const int iend = min(j + kl, n - 1);
            T* cj = ab + (long long)j * lda - j + off;

            if(tid < 32) { // warp 0 searches the pivot column in parallel
                int pv = j;
                T bv = fabs(cj[j]);
                for(int i = j + 1 + tid; i <= iend; i += 32) {
                    const T v = fabs(cj[i]);
                    if(v > bv) {
                        bv = v;
                        pv = i;
                    }
                }
                for(int s = 16; s > 0; s >>= 1) {
                    const T ov = __shfl_down_sync(0xffffffffu, bv, s);
                    const int op = __shfl_down_sync(0xffffffffu, pv, s);
                    if(ov > bv) {
                        bv = ov;
                        pv = op;
                    }
                }
                if(tid == 0) {
                    if(pv != j) atomicExch(pivoted, 1);
                    if(!(bv > T(0))) { // exactly singular column: perturb it, remember the first one
                        if(bad == 0) bad = d.row0 + j + 1;
                        cj[pv] = T(1);
                    }
                    piv[j] = pv;
                }
            }
            __syncthreads();

            const int pv = piv[j];
            const int kend = min(n - 1, j + kmax);

            if(pv != j) {
                for(int k = j + tid; k <= kend; k += nt) {
                    T* ck = ab + (long long)k * lda - k + off;
                    const T t = ck[j];
                    ck[j] = ck[pv];
                    ck[pv] = t;
                }
                __syncthreads();
            }

            const T pval = cj[j];
            for(int i = j + 1 + tid; i <= iend; i += nt) cj[i] /= pval;
            __syncthreads();

            const int kp = min(kend, jend - 1); // trailing updates are deferred
            for(int k = j + 1 + ty; k <= kp; k += nwarps) {
                T* ck = ab + (long long)k * lda - k + off;
                const T u = ck[j];
#pragma unroll 16
                for(int i = j + 1 + tx; i <= iend; i += 32) ck[i] -= cj[i] * u;
            }
            // The panel rows of the trailing columns also receive this step's
            // update; keeping it here (rather than solving U12 afterwards)
            // preserves the pivot order of the scalar algorithm, which swaps
            // only the current and trailing columns when it pivots.
            const int iend2 = min(iend, jend - 1);
            if(jend <= kend) {
                for(int k = jend + tid; k <= kend; k += nt) {
                    T* ck = ab + (long long)k * lda - k + off;
                    const T u = ck[j];
                    for(int i = j + 1; i <= iend2; ++i) ck[i] -= cj[i] * u;
                }
            }
            __syncthreads();
        }
        if(bad) atomicMin(info, bad);
    }

    // A22 -= L21 * U12.  Each thread owns a 4x4 micro-tile of the trailing
    // band, so the L and U values staged in shared memory are reused sixteen
    // times before the next inner step.
    template<class T, int NB, int TILE = 64>
    __global__ void band_lu_update(const pdesc<T>* __restrict__ pd, int lda, int kl, int ku, int j0, int nb) {
        const pdesc<T> d = pd[blockIdx.z];
        const int off = kl + ku;
        const int jend = min(d.size, j0 + nb);
        const int i0 = j0 + nb + blockIdx.y * TILE;
        const int k0 = j0 + nb + blockIdx.x * TILE;
        const int iend = min(d.size - 1, jend - 1 + kl);
        const int kend = min(d.size - 1, jend - 1 + kl + ku);
        T* __restrict__ ab = d.ab;
        __shared__ T ls[TILE][NB + 1], us[NB + 1][TILE + 1];
        const int tid = threadIdx.x;

        for(int e = tid; e < TILE * NB; e += 256) { // coalesced in rows
            const int rr = e / TILE, ii = e % TILE, i = i0 + ii, r = j0 + rr;
            ls[ii][rr] = (rr < nb && i <= iend && i - r <= kl && i > r) ? ab[(long long)r * lda + off + i - r] : T(0);
        }
        for(int e = tid; e < TILE * NB; e += 256) { // coalesced in columns
            const int col = e / NB, rr = e % NB, k = k0 + col, r = j0 + rr;
            us[rr][col] = (rr < nb && k <= kend && k - r <= kl + ku && k > r) ? ab[(long long)k * lda + off + r - k] : T(0);
        }
        __syncthreads();

        const int bi = 4 * (tid & 15), bk = 4 * (tid >> 4);
        T a00 = T(0), a01 = T(0), a02 = T(0), a03 = T(0);
        T a10 = T(0), a11 = T(0), a12 = T(0), a13 = T(0);
        T a20 = T(0), a21 = T(0), a22 = T(0), a23 = T(0);
        T a30 = T(0), a31 = T(0), a32 = T(0), a33 = T(0);
#pragma unroll
        for(int rr = 0; rr < NB; ++rr) {
            const T l0 = ls[bi][rr], l1 = ls[bi + 1][rr], l2 = ls[bi + 2][rr], l3 = ls[bi + 3][rr];
            const T u0 = us[rr][bk], u1 = us[rr][bk + 1], u2 = us[rr][bk + 2], u3 = us[rr][bk + 3];
            a00 = fma(l0, u0, a00);
            a01 = fma(l0, u1, a01);
            a02 = fma(l0, u2, a02);
            a03 = fma(l0, u3, a03);
            a10 = fma(l1, u0, a10);
            a11 = fma(l1, u1, a11);
            a12 = fma(l1, u2, a12);
            a13 = fma(l1, u3, a13);
            a20 = fma(l2, u0, a20);
            a21 = fma(l2, u1, a21);
            a22 = fma(l2, u2, a22);
            a23 = fma(l2, u3, a23);
            a30 = fma(l3, u0, a30);
            a31 = fma(l3, u1, a31);
            a32 = fma(l3, u2, a32);
            a33 = fma(l3, u3, a33);
        }
        const int i0t = i0 + bi, k0t = k0 + bk;
#define GSPIKE_ST(DI, DK) \
    if(i0t + DI <= iend && k0t + DK <= kend) ab[(long long)(k0t + DK) * lda + off + (i0t + DI) - (k0t + DK)] -= a##DI##DK
        GSPIKE_ST(0, 0);
        GSPIKE_ST(0, 1);
        GSPIKE_ST(0, 2);
        GSPIKE_ST(0, 3);
        GSPIKE_ST(1, 0);
        GSPIKE_ST(1, 1);
        GSPIKE_ST(1, 2);
        GSPIKE_ST(1, 3);
        GSPIKE_ST(2, 0);
        GSPIKE_ST(2, 1);
        GSPIKE_ST(2, 2);
        GSPIKE_ST(2, 3);
        GSPIKE_ST(3, 0);
        GSPIKE_ST(3, 1);
        GSPIKE_ST(3, 2);
        GSPIKE_ST(3, 3);
#undef GSPIKE_ST
    }

    // ---------------------------------------------------------------------------
    // Solve (LU) X = RHS in place.  One thread per right-hand side column.
    // RHS is indexed by global row: column c, row r -> rhs[d.row0 + r + c*ldr].
    // ---------------------------------------------------------------------------
    template<class T>
    __global__ void band_solve(const pdesc<T>* __restrict__ pd, int lda, int kl, int ku, T* __restrict__ rhs, int ldr, int ncols) {
        const int c = blockIdx.y * blockDim.x + threadIdx.x;
        if(c >= ncols) return;
        const pdesc<T> d = pd[blockIdx.x];
        const int n = d.size;
        const int off = kl + ku;
        T* __restrict__ r = rhs + d.row0 + (long long)c * ldr;

        for(int j = 0; j < n; ++j) { // forward: L y = P b
            const int pv = d.piv[j];
            if(pv != j) {
                const T t = r[j];
                r[j] = r[pv];
                r[pv] = t;
            }
            const T rj = r[j];
            const T* cj = d.ab + (long long)j * lda - j + off;
            const int iend = min(j + kl, n - 1);
            for(int i = j + 1; i <= iend; ++i) r[i] -= cj[i] * rj;
        }
        for(int i = n - 1; i >= 0; --i) { // back: U x = y, column sweep
            const T xi = r[i] / d.ab[off + (long long)i * lda];
            const T* ci = d.ab + off + (long long)i * lda - i;
            const int kend = (i > kl + ku) ? i - (kl + ku) : 0;
            for(int k = kend; k < i; ++k) r[k] -= ci[k] * xi;
            r[i] = xi;
        }
    }

    // ---------------------------------------------------------------------------
    // Solve (LU) X = RHS in place with one warp per right-hand side: the lanes
    // split the band.  The back substitution sweeps columns so the band reads
    // stay coalesced.
    // ---------------------------------------------------------------------------
    template<class T, int W = 1>
    __global__ void band_solve_warp(const pdesc<T>* __restrict__ pd, int lda, int kl, int ku, T* __restrict__ rhs, int ldr, int ncols) {
        constexpr int lanes = W * 32;
        const int c = blockIdx.y * (blockDim.x / lanes) + (threadIdx.x / lanes);
        if(c >= ncols) return;
        const int lane = threadIdx.x % lanes;
        const pdesc<T> d = pd[blockIdx.x];
        const int n = d.size;
        const int off = kl + ku;
        T* __restrict__ r = rhs + d.row0 + (long long)c * ldr;
        const T* __restrict__ ab = d.ab + off;
        if constexpr(W == 1) {
            for(int j = 0; j < n; ++j) { // forward: L y = P b
                const int pv = d.piv[j];
                if(lane == 0 && pv != j) {
                    const T t = r[j];
                    r[j] = r[pv];
                    r[pv] = t;
                }
                __syncwarp();
                const T rj = r[j];
                const T* cj = d.ab + (long long)j * lda - j + off;
                const int iend = min(j + kl, n - 1);
                for(int i = j + 1 + lane; i <= iend; i += 32) r[i] -= cj[i] * rj;
                __syncwarp();
            }
            for(int i = n - 1; i >= 0; --i) { // back: U x = y, column sweep
                const T xi = r[i] / ab[(long long)i * lda];
                const T* ci = d.ab + (long long)i * lda - i + off;
                const int kend = max(0, i - (kl + ku));
                for(int k = i - 1 - lane; k >= kend; k -= 32) r[k] -= ci[k] * xi;
                __syncwarp();
                if(lane == 0) r[i] = xi;
            }
        }
        else {
            for(int j = 0; j < n; ++j) {
                const int pv = d.piv[j];
                if(lane == 0 && pv != j) {
                    const T t = r[j];
                    r[j] = r[pv];
                    r[pv] = t;
                }
                __syncthreads();
                const T rj = r[j];
                const T* cj = d.ab + (long long)j * lda - j + off;
                const int iend = min(j + kl, n - 1);
                for(int i = j + 1 + lane; i <= iend; i += lanes) r[i] -= cj[i] * rj;
                __syncthreads();
            }
            for(int i = n - 1; i >= 0; --i) {
                const T xi = r[i] / ab[(long long)i * lda];
                const T* ci = d.ab + (long long)i * lda - i + off;
                const int kend = max(0, i - (kl + ku));
                for(int k = i - 1 - lane; k >= kend; k -= lanes) r[k] -= ci[k] * xi;
                __syncthreads();
                if(lane == 0) r[i] = xi;
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Solve (LU) X = RHS in place with one block per right-hand side.  Used when
    // there are only a few columns: the long sequential substitution chain
    // leaves a single warp mostly idle, so instead the whole block sweeps the
    // band together and moves through the rows in step.
    // ---------------------------------------------------------------------------
    template<class T>
    __global__ void band_solve_block(const pdesc<T>* __restrict__ pd, int lda, int kl, int ku, T* __restrict__ rhs, int ldr) {
        const int c = blockIdx.y;
        const int t = threadIdx.x;
        const int B = blockDim.x;
        const pdesc<T> d = pd[blockIdx.x];
        const int n = d.size;
        const int off = kl + ku;
        T* __restrict__ r = rhs + d.row0 + (long long)c * ldr;
        const T* __restrict__ ab = d.ab + off;

        for(int j = 0; j < n; ++j) { // forward: L y = P b
            const int pv = d.piv[j];
            if(t == 0 && pv != j) {
                const T tmp = r[j];
                r[j] = r[pv];
                r[pv] = tmp;
            }
            __syncthreads();
            const T rj = r[j];
            const T* cj = d.ab + (long long)j * lda - j + off;
            const int iend = min(j + kl, n - 1);
            for(int i = j + 1 + t; i <= iend; i += B) r[i] -= cj[i] * rj;
            __syncthreads();
        }
        for(int i = n - 1; i >= 0; --i) { // back: U x = y, column sweep
            const T xi = r[i] / ab[(long long)i * lda];
            const T* ci = d.ab + (long long)i * lda - i + off;
            const int kend = max(0, i - (kl + ku));
            for(int k = i - 1 - t; k >= kend; k -= B) r[k] -= ci[k] * xi;
            __syncthreads();
            if(t == 0) r[i] = xi;
        }
    }

    // dst(n x cols, ld ldd) = src(lds x cols)
    template<class T>
    __global__ void copy_mat(int n, int cols, const T* __restrict__ src, int lds, T* __restrict__ dst, int ldd) {
        const int c = blockIdx.x;
        for(int i = threadIdx.x; i < n; i += blockDim.x)
            dst[(long long)c * ldd + i] = src[(long long)c * lds + i];
    }

    template<class To, class From>
    __global__ void convert_mat(int n, int cols, const From* __restrict__ src, int lds, To* __restrict__ dst, int ldd) {
        const int c = blockIdx.x;
        for(int i = threadIdx.x; i < n; i += blockDim.x)
            dst[(long long)c * ldd + i] = (To)src[(long long)c * lds + i];
    }

    // y(n x cols, ld ldy) += a * x(n x cols, ld ldx)
    template<class T>
    __global__ void add_mat(int n, int cols, const T* __restrict__ x, int ldx, T* __restrict__ y, int ldy, T a) {
        const int c = blockIdx.x;
        for(int i = threadIdx.x; i < n; i += blockDim.x) y[(long long)c * ldy + i] += a * x[(long long)c * ldx + i];
    }

    // r = b - A x for banded A (lda = kl+ku+1, offset ku).
    template<class T>
    __global__ void band_residual(int n, int kl, int ku, const T* __restrict__ A, int lda, const T* __restrict__ x, int ldx, const T* __restrict__ b, int ldb, T* __restrict__ r, int ldr) {
        const int c = blockIdx.x;
        const int i = blockIdx.y * blockDim.x + threadIdx.x;
        if(i >= n) return;
        T s = T(0);
        const int j0 = (i > kl) ? i - kl : 0;
        const int j1 = min(n - 1, i + ku);
        for(int j = j0; j <= j1; ++j) s += A[(long long)j * lda + ku + i - j] * x[(long long)c * ldx + j];
        r[(long long)c * ldr + i] = b[(long long)c * ldb + i] - s;
    }

    // ---------------------------------------------------------------------------
    // Extract diagonal + coupling blocks of partition blockIdx.x from A.
    // Interface vectors are klu long, so both coupling blocks are klu x klu:
    //   bc[a + b*klu] : A(row0 + a,        row0 - klu + b)      (previous block)
    //   cc[a + b*klu] : A(row0 + n - klu + a, row0 + n + b)     (next block)
    // ---------------------------------------------------------------------------
    template<class T>
    __global__ void make_blocks(const pdesc<T>* __restrict__ pd, const T* __restrict__ A, int N, int ldaA, int ldaF, int ku, int kl, int klu, T* __restrict__ bc, T* __restrict__ cc) {
        const int i = blockIdx.x;
        const pdesc<T> d = pd[i];
        const int n = d.size;
        const int nserial = kl + ku;
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarp = blockDim.x >> 5;

        // One warp per matrix column: band entries are stored contiguously, so
        // the lanes copy them coalesced instead of one column per thread.
        for(int j = blockIdx.y * nwarp + warp; j < n; j += gridDim.y * nwarp) {
            const T* src = A + (long long)(d.row0 + j) * ldaA;
            T* dst = d.ab + (long long)j * ldaF;
            for(int dd = lane; dd < kl; dd += 32) dst[dd] = T(0);
            for(int dd = lane; dd <= nserial; dd += 32) {
                const int localrow = j + dd - ku;
                dst[kl + dd] = (localrow >= 0 && localrow < n) ? src[dd] : T(0);
            }
        }

        T* bci = bc + (long long)i * klu * klu;
        T* cci = cc + (long long)i * klu * klu;
        for(int e = threadIdx.x; e < klu * klu; e += blockDim.x) {
            bci[e] = T(0);
            cci[e] = T(0);
        }
        __syncthreads();
        for(int e = threadIdx.x; e < klu * klu; e += blockDim.x) {
            const int a = e % klu, b = e / klu;
            const long long gr = d.row0 + a;
            const long long gcm = (long long)d.row0 - klu + b;
            const long long offm = gr - gcm;
            if(gcm >= 0 && offm >= -ku && offm <= kl) bci[(long long)a + (long long)b * klu] = A[gcm * ldaA + ku + offm];
            const long long gp = (long long)d.row0 + n - klu + a;
            const long long gcp = (long long)d.row0 + n + b;
            const long long offp = gp - gcp;
            if(gcp < N && offp >= -ku && offp <= kl) cci[(long long)a + (long long)b * klu] = A[gcp * ldaA + ku + offp];
        }
    }

    // ---------------------------------------------------------------------------
    // Build the coupling right-hand side for a chunk of spike columns (2*klu total):
    //   spike column c in [0,klu)      -> C block (last klu rows of the block)
    //   spike column c in [klu,2*klu)  -> B block (first klu rows of the block)
    // ---------------------------------------------------------------------------
    template<class T>
    __global__ void spike_rhs(const pdesc<T>* __restrict__ pd, const T* __restrict__ bc, const T* __restrict__ cc, int klu, T* __restrict__ rhs, int ldr, int col0, int ncol) {
        const pdesc<T> d = pd[blockIdx.x];
        const int n = d.size;
        const int lc = blockIdx.y; // column inside this chunk
        const int c = col0 + lc;   // global spike column
        T* col = rhs + d.row0 + (long long)lc * ldr;
        for(int r = threadIdx.x; r < n; r += blockDim.x) col[r] = T(0);
        __syncthreads();
        if(c >= ncol) return;

        const T* bci = bc + (long long)blockIdx.x * klu * klu;
        const T* cci = cc + (long long)blockIdx.x * klu * klu;
        if(c < klu) {
            for(int a = threadIdx.x; a < klu; a += blockDim.x)
                col[n - klu + a] = cci[(long long)a + (long long)c * klu];
        }
        else {
            const int j = c - klu;
            for(int a = threadIdx.x; a < klu; a += blockDim.x)
                col[a] = bci[(long long)a + (long long)j * klu];
        }
    }

    // ---------------------------------------------------------------------------
    // Copy the top/bottom klu rows of a solved spike block into Vp / Vm of every
    // partition (sp = p blocks of [Vp | Vm], each 2klu x klu column major).
    //   Vp column j comes from spike column j,  Vm column j from column klu+j.
    // ---------------------------------------------------------------------------
    template<class T>
    __global__ void spike_extract(const pdesc<T>* __restrict__ pd, const T* __restrict__ rhs, int ldr, int klu, int col0, int ncols, T* __restrict__ sp) {
        const int i = blockIdx.x;
        const pdesc<T> d = pd[i];
        const int n = d.size;
        T* vp = sp + (long long)i * 4 * klu * klu;
        T* vm = vp + 2 * klu * klu;

        const long long idx = (long long)blockIdx.y * blockDim.x + threadIdx.x;
        const int j = (int)(idx % klu); // spike column
        const int a = (int)(idx / klu); // spike row
        if(a >= 2 * klu) return;
        const int row = (a < klu) ? a : (n - klu + (a - klu));

        const int lp = j - col0;
        if(lp >= 0 && lp < ncols)
            vp[(long long)a + (long long)j * 2 * klu] = rhs[d.row0 + row + (long long)lp * ldr];
        const int lm = (klu + j) - col0;
        if(lm >= 0 && lm < ncols)
            vm[(long long)a + (long long)j * 2 * klu] = rhs[d.row0 + row + (long long)lm * ldr];
    }

    // ---------------------------------------------------------------------------
    // Assemble the reduced (interface) matrix in band storage.
    // Block i (i = 0..p-2) holds u_i = [ z_i ; w_{i+1} ], 2*klu unknowns:
    //   D_i = [[I, Vp_i^bot], [Vm_{i+1}^top, I]]
    //   L_i = [[Vm_i^bot, 0], [0, 0]]          couples to block i-1
    //   U_i = [[0, 0], [0, Vp_{i+1}^top]]      couples to block i+1
    // Spike columns are 2*klu wide, so the sub-block entries use that stride.
    // ---------------------------------------------------------------------------
    template<class T>
    __global__ void reduced_diag(T** __restrict__ spp, int p, int klu, int klr, int ldaR, T* __restrict__ R) {
        const int i = blockIdx.x;
        const int I0 = 2 * klu * i;
        const int off = klr + klr;
        const int ld = 2 * klu;
        const T* vp_i = spp[i];
        const T* vm_i = spp[i] + 2 * klu * klu;
        const T* vp_n = spp[i + 1];
        const T* vm_n = spp[i + 1] + 2 * klu * klu;

        const auto ridx = [&](int I, int J) -> T& { return R[(long long)J * ldaR + off + (I - J)]; };
        for(long long e = (long long)threadIdx.x; e < (long long)klu * klu; e += blockDim.x) {
            const int a = (int)(e % klu);
            const int b = (int)(e / klu);
            const long long s = (long long)a + (long long)b * ld;
            ridx(I0 + a, I0 + b) = (a == b) ? T(1) : T(0);
            ridx(I0 + klu + a, I0 + klu + b) = (a == b) ? T(1) : T(0);
            // D_i: top-right = Vp_i^bot, bottom-left = Vm_{i+1}^top
            ridx(I0 + a, I0 + klu + b) = vp_i[s + klu];
            ridx(I0 + klu + a, I0 + b) = vm_n[s];
            // L_i: Vm_i^bot (row a, column b of previous block, first half)
            if(i > 0)
                ridx(I0 + a, I0 - 2 * klu + b) = vm_i[s + klu];
            // U_i: Vp_{i+1}^top (second half of block i+1)
            if(i + 1 <= p - 2)
                ridx(I0 + klu + a, I0 + 2 * klu + klu + b) = vp_n[s];
        }
    }

    // ---------------------------------------------------------------------------
    // Reduced right-hand side: rows [0,klu) of block i take the bottom half of
    // grj(i); rows [klu,2klu) take the top half of grj(i+1).
    //   grj: p * (2klu x nrhs), column major with ld 2*klu
    //   rhs: (m x nrhs), column major with ld m
    // ---------------------------------------------------------------------------
    template<class T>
    __global__ void reduced_rhs(const T* __restrict__ grj, int m, int klu, T* __restrict__ rhs, int nrhs) {
        const int i = blockIdx.x;
        const int c = blockIdx.y * blockDim.x + threadIdx.x;
        if(c >= nrhs || i >= m / (2 * klu)) return;
        const long long gcur = (long long)i * 2 * klu * nrhs + (long long)c * 2 * klu;
        const long long gnext = gcur + (long long)2 * klu * nrhs;
        for(int a = 0; a < klu; ++a) {
            rhs[(long long)(2 * klu * i + a) + (long long)c * m] = grj[gcur + klu + a];
            rhs[(long long)(2 * klu * i + klu + a) + (long long)c * m] = grj[gnext + a];
        }
    }

    // ---------------------------------------------------------------------------
    // Apply the interface corrections in place to the partition right-hand sides:
    //   x_i <- f_i - C_i w_{i+1} - B_i z_{i-1}
    // Only the corner rows are touched.
    // ---------------------------------------------------------------------------
    template<class T>
    __global__ void correct_rhs(const pdesc<T>* __restrict__ pd, const T* __restrict__ bc, const T* __restrict__ cc, const T* __restrict__ u, int klu, int m, int p, T* __restrict__ rhs, int ldr, int nrhs) {
        const int i = blockIdx.x;
        const int c = blockIdx.y;
        if(c >= nrhs) return;
        const pdesc<T> d = pd[i];
        const int n = d.size;
        T* col = rhs + d.row0 + (long long)c * ldr;

        const T* bci = bc + (long long)i * klu * klu;
        const T* cci = cc + (long long)i * klu * klu;
        const T* z = (i > 0) ? u + (long long)(2 * klu * (i - 1)) + (long long)c * m : nullptr;
        const T* w = (i < p - 1) ? u + (long long)(2 * klu * i + klu) + (long long)c * m : nullptr;

        // One thread per corner row; each row is an independent dot product.
        for(int a = threadIdx.x; a < klu; a += blockDim.x) {
            if(w) {
                T s = T(0);
                for(int j = 0; j < klu; ++j) s += cci[(long long)a + (long long)j * klu] * w[j];
                col[n - klu + a] -= s;
            }
            if(z) {
                T s = T(0);
                for(int j = 0; j < klu; ++j) s += bci[(long long)a + (long long)j * klu] * z[j];
                col[a] -= s;
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Extract top/bottom klu rows of the local solution g_i into grj.
    // ---------------------------------------------------------------------------
    template<class T>
    __global__ void extract_corners(const pdesc<T>* __restrict__ pd, const T* __restrict__ src, int lds, int klu, int nrhs, T* __restrict__ grj) {
        const int i = blockIdx.x;
        const int c = blockIdx.y * blockDim.x + threadIdx.x;
        if(c >= nrhs) return;
        const pdesc<T> d = pd[i];
        const int n = d.size;
        T* g = grj + (long long)i * 2 * klu * nrhs + (long long)c * 2 * klu;
        for(int a = 0; a < klu; ++a) {
            g[a] = src[d.row0 + a + (long long)c * lds];
            g[klu + a] = src[d.row0 + n - klu + a + (long long)c * lds];
        }
    }

} // namespace gspike::dk
