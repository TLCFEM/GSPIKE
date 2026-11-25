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
    __global__ void band_lu_scalar(const pdesc<T>* __restrict__ pd, const int lda, const int kl, const int ku, int* __restrict__ info) {
        const pdesc<T> d = pd[blockIdx.x];
        const int n = d.size;
        const int off = kl + ku;
        const int tid = threadIdx.x;
        const int nt = blockDim.x;
        const int tx = tid & 31;
        const int ty = tid >> 5;
        const int nwarps = (nt + 31) / 32;
        const int kmax = min(kl + ku, n - 1);
        T* const __restrict__ ab = d.ab;
        int* const __restrict__ piv = d.piv;
        int bad = 0;

        for(int j = 0; j < n; ++j) {
            const int iend = min(j + kl, n - 1);
            T* const cj = ab + (long long)j * lda - j + off;

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
                    T* const ck = ab + (long long)k * lda - k + off;
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
                T* const ck = ab + (long long)k * lda - k + off;
                const T u = ck[j];
#pragma unroll 16
                for(int i = j + 1 + tx; i <= iend; i += 32) ck[i] -= cj[i] * u;
            }
            __syncthreads();
        }
        if(bad) atomicMin(info, bad);
    }

    // ---------------------------------------------------------------------------
    // Band LU with partial pivoting, blocked over panels of NB columns (4 in
    // the current instantiation).
    //
    // Factor one panel per block (NB columns of sequential pivoting), then
    // apply the whole panel to the trailing band with a tiled rank-NB update.
    // A scalar, one-column-at-a-time band LU moves a whole kl x (kl+ku) tile
    // through L1 for a single FMA per element; the blocked update amortizes
    // those loads and stores over NB FMAs and spreads the work over many SMs.
    // Smaller panels were measured to be faster than 16: the panel kernel has
    // many mostly idle threads, and a rank-4 update keeps enough work per tile
    // without the register pressure of wider panels.
    // ---------------------------------------------------------------------------
    template<class T>
    __global__ void band_lu_panel(const pdesc<T>* __restrict__ pd, const int lda, const int kl, const int ku, int j0, const int nb, int* __restrict__ info, int* __restrict__ pivoted) {
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
        T* const __restrict__ ab = d.ab;
        int* const __restrict__ piv = d.piv;
        int bad = 0;
        const int jend = min(n, j0 + nb);

        for(int j = j0; j < jend; ++j) {
            const int iend = min(j + kl, n - 1);
            T* const cj = ab + (long long)j * lda - j + off;

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
                    T* const ck = ab + (long long)k * lda - k + off;
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
                T* const ck = ab + (long long)k * lda - k + off;
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
                    T* const ck = ab + (long long)k * lda - k + off;
                    const T u = ck[j];
                    for(int i = j + 1; i <= iend2; ++i) ck[i] -= cj[i] * u;
                }
            }
            __syncthreads();
        }
        if(bad) atomicMin(info, bad);
    }

    // Only fp32 uses the vector store: vectorized double4 stores were measured
    // no faster, so fp64 keeps the scalar path.
    template<class T>
    __device__ __forceinline__ void store_band4(T* p, T a0, T a1, T a2, T a3) {
        if(sizeof(T) == 4 && (reinterpret_cast<unsigned long long>(p) & 15) == 0) {
            auto* q = reinterpret_cast<float4*>(p);
            float4 v = *q;
            v.x -= a0;
            v.y -= a1;
            v.z -= a2;
            v.w -= a3;
            *q = v;
            return;
        }
        p[0] -= a0;
        p[1] -= a1;
        p[2] -= a2;
        p[3] -= a3;
    }

    // A22 -= L21 * U12.  Each thread owns a 4x4 micro-tile of the trailing
    // band, so the L and U values staged in shared memory are reused across
    // the four rows and four columns of the tile before the next inner step.
    // TILE must stay 64: 256 threads form 16 four-row/four-column micro-tiles.
    template<class T, int NB, int TILE = 64>
    __global__ void band_lu_update(const pdesc<T>* __restrict__ pd, const int lda, const int kl, const int ku, const int j0, const int nb) {
        // The 256-thread 4x4 micro-tile covers exactly 64 rows and columns.
        static_assert(TILE == 64, "band_lu_update requires TILE == 64");
        const pdesc<T> d = pd[blockIdx.z];
        const int off = kl + ku;
        const int jend = min(d.size, j0 + nb);
        const int i0 = j0 + nb + blockIdx.y * TILE;
        const int k0 = j0 + nb + blockIdx.x * TILE;
        const int iend = min(d.size - 1, jend - 1 + kl);
        const int kend = min(d.size - 1, jend - 1 + kl + ku);
        T* const __restrict__ ab = d.ab;
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
        // Four consecutive rows of a band column are contiguous in memory.
        // For fp32, a vector store turns four 16-byte-strided scalar stores
        // (16 L1 sectors per warp) into one coalesced 16-byte-per-lane store;
        // fp64 uses the scalar path.
#define GSPIKE_ST(DK)                                                                     \
    {                                                                                     \
        const int kk = k0t + (DK);                                                        \
        if(kk <= kend) {                                                                  \
            const long long ad = (long long)kk * lda + off + i0t - kk;                    \
            if(i0t + 3 <= iend) dk::store_band4(ab + ad, a0##DK, a1##DK, a2##DK, a3##DK); \
            else {                                                                        \
                if(i0t <= iend) ab[ad] -= a0##DK;                                         \
                if(i0t + 1 <= iend) ab[ad + 1] -= a1##DK;                                 \
                if(i0t + 2 <= iend) ab[ad + 2] -= a2##DK;                                 \
                if(i0t + 3 <= iend) ab[ad + 3] -= a3##DK;                                 \
            }                                                                             \
        }                                                                                 \
    }
        GSPIKE_ST(0);
        GSPIKE_ST(1);
        GSPIKE_ST(2);
        GSPIKE_ST(3);
#undef GSPIKE_ST
    }

    // Precompute 1 / U(ii) once after factorization.  Every RHS then needs a
    // multiply instead of a full-precision division in back substitution; the
    // division is also duplicated across all lanes of a warp/block solve.
    template<class T>
    __global__ void invert_diag(const pdesc<T>* __restrict__ pd, const int lda, const int off, T* __restrict__ inv) {
        const pdesc<T> d = pd[blockIdx.x];
        for(int r = threadIdx.x; r < d.size; r += blockDim.x)
            inv[d.row0 + r] = T(1) / d.ab[off + (long long)r * lda];
    }

    // ---------------------------------------------------------------------------
    // Determinant from the existing LU factors.  SPIKE gives
    //   det(A) = prod_i det(A_ii) * det(R),
    // each LU contributing (-1)^swaps * prod U_kk.  The sign is XOR-reduced;
    // magnitudes are accumulated as a mantissa in [0.5,1) with a separate
    // base-2 exponent, so the product cannot overflow or underflow mid-reduction.
    // ---------------------------------------------------------------------------
    template<class T>
    __device__ __forceinline__ void det_combine(T& a, long long& ea, const T b, const long long eb) {
        a *= b; // mantissas in [0.5,1) -> product in [0.25,1)
        ea += eb;
        if(a < T(0.5)) {
            a *= T(2);
            --ea;
        }
    }

    template<class T>
    __global__ void determinant(const pdesc<T>* __restrict__ pd, const int p,
                                const int lda_f, const int off_f,
                                const pdesc<T>* __restrict__ rp, const int m,
                                const int ldaR, const int offR,
                                T* __restrict__ det_out, int* __restrict__ sign_out) {
        T prod = T(1);
        long long exp2 = 0;
        int flip = 0;

        const auto add = [&](const T v) {
            int e = 0;
            const T mant = frexp(fabs(v), &e);
            det_combine(prod, exp2, mant, e);
            flip ^= (v < T(0));
        };
        for(int pi = 0; pi < p; ++pi) {
            const pdesc<T> d = pd[pi];
            for(int j = threadIdx.x; j < d.size; j += blockDim.x) {
                flip ^= (d.piv[j] != j);
                add(d.ab[off_f + (long long)j * lda_f]); // U(j,j) of this block
            }
        }
        if(m > 0) {
            const pdesc<T> d = rp[0];
            for(int k = threadIdx.x; k < m; k += blockDim.x) {
                flip ^= (d.piv[k] != k);
                add(d.ab[offR + (long long)k * ldaR]); // U(k,k) of the reduced system
            }
        }

        __shared__ T sprod[32];
        __shared__ long long sexp[32];
        __shared__ int sflip[32];
        const int lane = threadIdx.x & 31;
        const int wid = threadIdx.x >> 5;
        const int nwarps = blockDim.x >> 5;
        const int wflip = __popc(__ballot_sync(0xffffffffu, flip)) & 1;
        // Warp reduction of the (mantissa, exponent) pairs.  det_combine keeps
        // the mantissa in [0.5,1), so any pair order is safe.
        for(int s = 16; s > 0; s >>= 1) {
            const T op = __shfl_down_sync(0xffffffffu, prod, s);
            const long long oe = __shfl_down_sync(0xffffffffu, exp2, s);
            if(lane < s) det_combine(prod, exp2, op, oe);
        }
        if(lane == 0) {
            sprod[wid] = prod;
            sexp[wid] = exp2;
            sflip[wid] = wflip;
        }
        __syncthreads();
        for(int s = nwarps / 2; s > 0; s >>= 1) {
            if(wid < s && lane == 0) {
                det_combine(sprod[wid], sexp[wid], sprod[wid + s], sexp[wid + s]);
                sflip[wid] ^= sflip[wid + s];
            }
            __syncthreads();
        }
        if(threadIdx.x == 0) {
            const int s = sflip[0] ? -1 : 1;
            const long long e = sexp[0];
            const int ec = (int)(e > 2000 ? 2000 : (e < -2000 ? -2000 : e));
            const T v = ldexp(sprod[0], ec);
            *sign_out = s;
            *det_out = (s < 0) ? -v : v;
        }
    }

    // ---------------------------------------------------------------------------
    // Blocked triangular solves for the spike-generation stage (many RHS, no
    // pivoting).  A panel of B rows is solved sequentially, then its L/U panel
    // is applied to a tile of C right-hand sides with a register micro-tile.
    // Each band factor is loaded once per RHS tile instead of once per column,
    // which removes most of the L2 traffic of the scalar band solve.  The host
    // launches one pair of kernels per panel; pivoted systems use the scalar
    // path instead.
    // ---------------------------------------------------------------------------
    // One warp per RHS solves the B panel rows sequentially.  Only rows inside
    // the panel are touched here; the band below is handled by the update.
    template<class T, int B>
    __global__ void band_solve_fwd_panel(const pdesc<T>* __restrict__ pd, const int lda, const int kl, const int ku, const int j0, T* __restrict__ rhs, const int ldr, const int ncols, const int ncC) {
        const pdesc<T> d = pd[blockIdx.x];
        const int n = d.size;
        const int jend = min(n, j0 + B);
        const int col = blockIdx.y * 32 + (threadIdx.x >> 5);
        if(col >= ncols || j0 >= n) return;
        // C-spike columns have a nonzero RHS only in the bottom klu rows, so
        // their forward sweep starts there; earlier rows stay exactly zero.
        if(col < ncC && j0 + B <= n - max(kl, ku)) return;
        const int lane = threadIdx.x & 31;
        const int off = kl + ku;
        T* const __restrict__ r = rhs + d.row0 + (long long)col * ldr;
        for(int j = j0; j < jend; ++j) {
            const T rj = r[j];
            const T* const cj = d.ab + (long long)j * lda - j + off;
            const int iend = min(j + kl, jend - 1);
            for(int i = j + 1 + lane; i <= iend; i += 32) r[i] = fma(-cj[i], rj, r[i]);
            __syncwarp();
        }
    }

    // Y(i0:i0+R, :) -= L(i0:i0+R, j0:j0+B) * Y(j0:j0+B, :).
    template<class T, int B, int C, int R>
    __global__ void band_solve_fwd_update(const pdesc<T>* __restrict__ pd, int lda, int kl, int ku, int j0, T* __restrict__ rhs, int ldr, int ncols, int ncC) {
        static_assert(B > 0 && C % 4 == 0 && R % 4 == 0 && B <= R && B <= C, "bad blocked-solve tile sizes");
        static_assert((R / 4) * (C / 4) <= 1024, "blocked-solve tile needs too many threads");
        static_assert((R * B + C * B) * sizeof(T) <= 48 * 1024, "blocked-solve shared memory exceeds 48 KB");
        const pdesc<T> d = pd[blockIdx.z];
        const int n = d.size;
        if(j0 >= n) return;
        const int activeC = j0 + B > n - max(kl, ku);
        const int off = kl + ku;
        const int c0 = blockIdx.x * C;
        const int i0 = j0 + B + blockIdx.y * R;
        const int imax = min(n - 1, j0 + B - 1 + kl);
        if(i0 > imax) return;
        __shared__ T sl[R][B], sy[C][B];
        for(int e = threadIdx.x; e < R * B; e += blockDim.x) {
            const int rr = e / B, b = e % B, i = i0 + rr, j = j0 + b;
            sl[rr][b] = (i <= imax && i - j > 0 && i - j <= kl) ? d.ab[(long long)j * lda + off + i - j] : T(0);
        }
        for(int e = threadIdx.x; e < C * B; e += blockDim.x) {
            const int lc = e / B, b = e % B, col = c0 + lc;
            const bool inactive = (col < ncC && !activeC);
            sy[lc][b] = (col < ncols && j0 + b < n && !inactive) ? rhs[d.row0 + j0 + b + (long long)col * ldr] : T(0);
        }
        __syncthreads();
        const int bi = 4 * (threadIdx.x % (R / 4)), bc = 4 * (threadIdx.x / (R / 4));
        T a00 = 0, a01 = 0, a02 = 0, a03 = 0;
        T a10 = 0, a11 = 0, a12 = 0, a13 = 0;
        T a20 = 0, a21 = 0, a22 = 0, a23 = 0;
        T a30 = 0, a31 = 0, a32 = 0, a33 = 0;
        // Whole C-spike tiles below the active rows are exactly zero.
        if(!activeC && c0 + bc + 3 < ncC) return;
#pragma unroll
        for(int b = 0; b < B; ++b) {
            const T l0 = sl[bi][b], l1 = sl[bi + 1][b], l2 = sl[bi + 2][b], l3 = sl[bi + 3][b];
            const T y0 = sy[bc][b], y1 = sy[bc + 1][b], y2 = sy[bc + 2][b], y3 = sy[bc + 3][b];
            a00 = fma(l0, y0, a00);
            a01 = fma(l0, y1, a01);
            a02 = fma(l0, y2, a02);
            a03 = fma(l0, y3, a03);
            a10 = fma(l1, y0, a10);
            a11 = fma(l1, y1, a11);
            a12 = fma(l1, y2, a12);
            a13 = fma(l1, y3, a13);
            a20 = fma(l2, y0, a20);
            a21 = fma(l2, y1, a21);
            a22 = fma(l2, y2, a22);
            a23 = fma(l2, y3, a23);
            a30 = fma(l3, y0, a30);
            a31 = fma(l3, y1, a31);
            a32 = fma(l3, y2, a32);
            a33 = fma(l3, y3, a33);
        }
#define GSPIKE_FWD_ST(DR, DC)                                    \
    {                                                            \
        const int i = i0 + bi + (DR), col = c0 + bc + (DC);      \
        if(i <= imax && col < ncols && !(col < ncC && !activeC)) \
            rhs[d.row0 + i + (long long)col * ldr] -= a##DR##DC; \
    }
        GSPIKE_FWD_ST(0, 0);
        GSPIKE_FWD_ST(0, 1);
        GSPIKE_FWD_ST(0, 2);
        GSPIKE_FWD_ST(0, 3);
        GSPIKE_FWD_ST(1, 0);
        GSPIKE_FWD_ST(1, 1);
        GSPIKE_FWD_ST(1, 2);
        GSPIKE_FWD_ST(1, 3);
        GSPIKE_FWD_ST(2, 0);
        GSPIKE_FWD_ST(2, 1);
        GSPIKE_FWD_ST(2, 2);
        GSPIKE_FWD_ST(2, 3);
        GSPIKE_FWD_ST(3, 0);
        GSPIKE_FWD_ST(3, 1);
        GSPIKE_FWD_ST(3, 2);
        GSPIKE_FWD_ST(3, 3);
#undef GSPIKE_FWD_ST
    }

    // One warp per RHS solves the B panel rows of U from the bottom up.
    template<class T, int B>
    __global__ void band_solve_bwd_panel(const pdesc<T>* __restrict__ pd, const int lda, const int kl, const int ku, const int i0, const T* __restrict__ diag, T* __restrict__ rhs, const int ldr, const int ncols) {
        const pdesc<T> d = pd[blockIdx.x];
        const int n = d.size;
        const int iend = min(n, i0 + B);
        const int col = blockIdx.y * 32 + (threadIdx.x >> 5);
        if(col >= ncols || i0 >= n) return;
        const int lane = threadIdx.x & 31;
        const int off = kl + ku;
        T* const __restrict__ r = rhs + d.row0 + (long long)col * ldr;
        for(int i = iend - 1; i >= i0; --i) {
            const int kend = min(iend, i + kl + ku + 1);
            T sum = 0;
            for(int k = i + 1 + lane; k < kend; k += 32)
                sum = fma(d.ab[(long long)k * lda + off + i - k], r[k], sum);
            for(int s = 16; s > 0; s >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, s);
            if(lane == 0) r[i] = (r[i] - sum) * diag[d.row0 + i];
            __syncwarp();
        }
    }

    // X(k0:k0+R, :) -= U(k0:k0+R, i0:i0+B) * X(i0:i0+B, :).
    template<class T, int B, int C, int R>
    __global__ void band_solve_bwd_update(const pdesc<T>* __restrict__ pd, int lda, int kl, int ku, int i0, T* __restrict__ rhs, int ldr, int ncols) {
        static_assert(B > 0 && C % 4 == 0 && R % 4 == 0 && B <= R && B <= C, "bad blocked-solve tile sizes");
        static_assert((R / 4) * (C / 4) <= 1024, "blocked-solve tile needs too many threads");
        static_assert((R * B + C * B) * sizeof(T) <= 48 * 1024, "blocked-solve shared memory exceeds 48 KB");
        const pdesc<T> d = pd[blockIdx.z];
        if(i0 <= 0) return;
        const int off = kl + ku;
        const int c0 = blockIdx.x * C;
        const int k1 = i0 - 1 - blockIdx.y * R;
        if(k1 < 0) return;
        const int k0 = k1 - R + 1;
        __shared__ T su[R][B], sx[C][B];
        for(int e = threadIdx.x; e < R * B; e += blockDim.x) {
            const int rr = e / B, b = e % B, k = k0 + rr, j = i0 + b;
            // The host launches panels derived from the largest partition, so
            // j can run past this partition's own size; never read its factors.
            su[rr][b] = (k >= 0 && j < d.size && j - k <= kl + ku) ? d.ab[(long long)j * lda + off + k - j] : T(0);
        }
        for(int e = threadIdx.x; e < C * B; e += blockDim.x) {
            const int lc = e / B, b = e % B, col = c0 + lc;
            sx[lc][b] = (col < ncols && i0 + b < d.size) ? rhs[d.row0 + i0 + b + (long long)col * ldr] : T(0);
        }
        __syncthreads();
        const int bi = 4 * (threadIdx.x % (R / 4)), bc = 4 * (threadIdx.x / (R / 4));
        T a00 = 0, a01 = 0, a02 = 0, a03 = 0;
        T a10 = 0, a11 = 0, a12 = 0, a13 = 0;
        T a20 = 0, a21 = 0, a22 = 0, a23 = 0;
        T a30 = 0, a31 = 0, a32 = 0, a33 = 0;
#pragma unroll
        for(int b = 0; b < B; ++b) {
            const T u0 = su[bi][b], u1 = su[bi + 1][b], u2 = su[bi + 2][b], u3 = su[bi + 3][b];
            const T x0 = sx[bc][b], x1 = sx[bc + 1][b], x2 = sx[bc + 2][b], x3 = sx[bc + 3][b];
            a00 = fma(u0, x0, a00);
            a01 = fma(u0, x1, a01);
            a02 = fma(u0, x2, a02);
            a03 = fma(u0, x3, a03);
            a10 = fma(u1, x0, a10);
            a11 = fma(u1, x1, a11);
            a12 = fma(u1, x2, a12);
            a13 = fma(u1, x3, a13);
            a20 = fma(u2, x0, a20);
            a21 = fma(u2, x1, a21);
            a22 = fma(u2, x2, a22);
            a23 = fma(u2, x3, a23);
            a30 = fma(u3, x0, a30);
            a31 = fma(u3, x1, a31);
            a32 = fma(u3, x2, a32);
            a33 = fma(u3, x3, a33);
        }
#define GSPIKE_BWD_ST(DR, DC)                                                                        \
    {                                                                                                \
        const int k = k0 + bi + (DR), col = c0 + bc + (DC);                                          \
        if(k >= 0 && k < d.size && col < ncols) rhs[d.row0 + k + (long long)col * ldr] -= a##DR##DC; \
    }
        GSPIKE_BWD_ST(0, 0);
        GSPIKE_BWD_ST(0, 1);
        GSPIKE_BWD_ST(0, 2);
        GSPIKE_BWD_ST(0, 3);
        GSPIKE_BWD_ST(1, 0);
        GSPIKE_BWD_ST(1, 1);
        GSPIKE_BWD_ST(1, 2);
        GSPIKE_BWD_ST(1, 3);
        GSPIKE_BWD_ST(2, 0);
        GSPIKE_BWD_ST(2, 1);
        GSPIKE_BWD_ST(2, 2);
        GSPIKE_BWD_ST(2, 3);
        GSPIKE_BWD_ST(3, 0);
        GSPIKE_BWD_ST(3, 1);
        GSPIKE_BWD_ST(3, 2);
        GSPIKE_BWD_ST(3, 3);
#undef GSPIKE_BWD_ST
    }

    // ---------------------------------------------------------------------------
    // Solve (LU) X = RHS in place.  One thread per right-hand side column.
    // RHS is indexed by global row: column c, row r -> rhs[d.row0 + r + c*ldr].
    // ---------------------------------------------------------------------------
    template<class T>
    __global__ void band_solve(const pdesc<T>* __restrict__ pd, const int lda, const int kl, const int ku, const T* __restrict__ diag, T* __restrict__ rhs, const int ldr, const int ncols) {
        const int c = blockIdx.y * blockDim.x + threadIdx.x;
        if(c >= ncols) return;
        const pdesc<T> d = pd[blockIdx.x];
        const int n = d.size;
        const int off = kl + ku;
        T* const __restrict__ r = rhs + d.row0 + (long long)c * ldr;

        for(int j = 0; j < n; ++j) { // forward: L y = P b
            const int pv = d.piv[j];
            if(pv != j) {
                const T t = r[j];
                r[j] = r[pv];
                r[pv] = t;
            }
            const T rj = r[j];
            const T* const cj = d.ab + (long long)j * lda - j + off;
            const int iend = min(j + kl, n - 1);
            for(int i = j + 1; i <= iend; ++i) r[i] -= cj[i] * rj;
        }
        for(int i = n - 1; i >= 0; --i) { // back: U x = y, column sweep
            const T xi = r[i] * diag[d.row0 + i];
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
    __global__ void band_solve_warp(const pdesc<T>* __restrict__ pd, const int lda, const int kl, const int ku, const T* __restrict__ diag, T* __restrict__ rhs, const int ldr, const int ncols) {
        constexpr int lanes = W * 32;
        const int c = blockIdx.y * (blockDim.x / lanes) + (threadIdx.x / lanes);
        const bool active = c < ncols;
        const int lane = threadIdx.x % lanes;
        const pdesc<T> d = pd[blockIdx.x];
        const int n = d.size;
        const int off = kl + ku;
        if constexpr(W == 1) {
            // One warp per column: the early exit is warp-uniform, so the
            // warp-local __syncwarp() calls stay well formed.
            if(!active) return;
            T* const __restrict__ r = rhs + d.row0 + (long long)c * ldr;
            for(int j = 0; j < n; ++j) { // forward: L y = P b
                const int pv = d.piv[j];
                if(pv != j) {
                    if(lane == 0) {
                        const T t = r[j];
                        r[j] = r[pv];
                        r[pv] = t;
                    }
                    __syncwarp(); // pv is uniform across the warp
                }
                const T rj = r[j];
                const T* const cj = d.ab + (long long)j * lda - j + off;
                const int iend = min(j + kl, n - 1);
                for(int i = j + 1 + lane; i <= iend; i += 32) r[i] -= cj[i] * rj;
                __syncwarp();
            }
            for(int i = n - 1; i >= 0; --i) { // back: U x = y, column sweep
                const T xi = r[i] * diag[d.row0 + i];
                const T* ci = d.ab + (long long)i * lda - i + off;
                const int kend = max(0, i - (kl + ku));
                for(int k = i - 1 - lane; k >= kend; k -= 32) r[k] -= ci[k] * xi;
                __syncwarp();
                if(lane == 0) r[i] = xi;
            }
        }
        else {
            // Several warps cooperate on one column.  Inactive groups must
            // still reach every __syncthreads(): the last group of a block is
            // usually only partially filled (ncols is not a multiple of
            // 32/W), and an early return there would deadlock the CTA.
            T* const __restrict__ r = active ? rhs + d.row0 + (long long)c * ldr : rhs;
            for(int j = 0; j < n; ++j) {
                const int pv = d.piv[j];
                if(pv != j) {
                    if(active && lane == 0) {
                        const T t = r[j];
                        r[j] = r[pv];
                        r[pv] = t;
                    }
                    __syncthreads(); // pv is uniform across the block
                }
                if(active) {
                    const T rj = r[j];
                    const T* const cj = d.ab + (long long)j * lda - j + off;
                    const int iend = min(j + kl, n - 1);
                    for(int i = j + 1 + lane; i <= iend; i += lanes) r[i] -= cj[i] * rj;
                }
                __syncthreads();
            }
            for(int i = n - 1; i >= 0; --i) {
                T xi = T(0);
                if(active) {
                    xi = r[i] * diag[d.row0 + i];
                    const T* ci = d.ab + (long long)i * lda - i + off;
                    const int kend = max(0, i - (kl + ku));
                    for(int k = i - 1 - lane; k >= kend; k -= lanes) r[k] -= ci[k] * xi;
                }
                __syncthreads();
                if(active && lane == 0) r[i] = xi;
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
    __global__ void band_solve_block(const pdesc<T>* __restrict__ pd, const int lda, const int kl, const int ku, const T* __restrict__ diag, T* __restrict__ rhs, const int ldr) {
        const int c = blockIdx.y;
        const int t = threadIdx.x;
        const int B = blockDim.x;
        const pdesc<T> d = pd[blockIdx.x];
        const int n = d.size;
        const int off = kl + ku;
        T* const __restrict__ r = rhs + d.row0 + (long long)c * ldr;

        for(int j = 0; j < n; ++j) { // forward: L y = P b
            const int pv = d.piv[j];
            if(pv != j) {
                if(t == 0) {
                    const T tmp = r[j];
                    r[j] = r[pv];
                    r[pv] = tmp;
                }
                __syncthreads(); // piv[j] is uniform across the block
            }
            const T rj = r[j];
            const T* const cj = d.ab + (long long)j * lda - j + off;
            const int iend = min(j + kl, n - 1);
            for(int i = j + 1 + t; i <= iend; i += B) r[i] -= cj[i] * rj;
            __syncthreads();
        }
        for(int i = n - 1; i >= 0; --i) { // back: U x = y, column sweep
            const T xi = r[i] * diag[d.row0 + i];
            const T* ci = d.ab + (long long)i * lda - i + off;
            const int kend = max(0, i - (kl + ku));
            for(int k = i - 1 - t; k >= kend; k -= B) r[k] -= ci[k] * xi;
            __syncthreads();
            if(t == 0) r[i] = xi;
        }
    }

    // dst(n x cols, ld ldd) = src(lds x cols)
    template<class T>
    __global__ void copy_mat(const int n, const T* __restrict__ src, const int lds, T* __restrict__ dst, const int ldd) {
        const int c = blockIdx.x;
        for(int i = threadIdx.x; i < n; i += blockDim.x)
            dst[(long long)c * ldd + i] = src[(long long)c * lds + i];
    }

    template<class To, class From>
    __global__ void convert_mat(const int n, const From* __restrict__ src, const int lds, To* __restrict__ dst, const int ldd) {
        const int c = blockIdx.x;
        for(int i = threadIdx.x; i < n; i += blockDim.x)
            dst[(long long)c * ldd + i] = (To)src[(long long)c * lds + i];
    }

    // y(n x cols, ld ldy) += a * x(n x cols, ld ldx)
    template<class T>
    __global__ void add_mat(const int n, const T* __restrict__ x, const int ldx, T* __restrict__ y, const int ldy, T a) {
        const int c = blockIdx.x;
        for(int i = threadIdx.x; i < n; i += blockDim.x) y[(long long)c * ldy + i] += a * x[(long long)c * ldx + i];
    }

    // r = b - A x for banded A (lda = kl+ku+1, offset ku).
    template<class T>
    __global__ void band_residual(const int n, const int kl, const int ku, const T* __restrict__ A, const int lda, const T* __restrict__ x, const int ldx, const T* __restrict__ b, const int ldb, T* __restrict__ r, const int ldr) {
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
    __global__ void make_blocks(const pdesc<T>* __restrict__ pd, const T* __restrict__ A, const int N, const int ldaA, const int ldaF, const int ku, const int kl, const int klu, T* __restrict__ bc, T* __restrict__ cc) {
        const int i = blockIdx.x;
        const pdesc<T> d = pd[i];
        const int n = d.size;
        const int nserial = kl + ku;
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarp = blockDim.x >> 5;

        // One warp per matrix column: band entries are stored contiguously, so
        // the lanes copy them coalesced instead of one column per thread.
        for(int j = blockIdx.y * nwarp + warp; j < n; j += gridDim.y * nwarp) {
            const T* src = A + (long long)(d.row0 + j) * ldaA;
            T* const dst = d.ab + (long long)j * ldaF;
            for(int dd = lane; dd < kl; dd += 32) dst[dd] = T(0);
            for(int dd = lane; dd <= nserial; dd += 32) {
                const int localrow = j + dd - ku;
                dst[kl + dd] = (localrow >= 0 && localrow < n) ? src[dd] : T(0);
            }
        }

        T* const bci = bc + (long long)i * klu * klu;
        T* const cci = cc + (long long)i * klu * klu;
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
    __global__ void spike_rhs(const pdesc<T>* __restrict__ pd, const T* __restrict__ bc, const T* __restrict__ cc, const int klu, T* __restrict__ rhs, const int ldr, const int col0, const int ncol) {
        const pdesc<T> d = pd[blockIdx.x];
        const int n = d.size;
        const int lc = blockIdx.y; // column inside this chunk
        const int c = col0 + lc;   // global spike column
        T* const col = rhs + d.row0 + (long long)lc * ldr;
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
    __global__ void spike_extract(const pdesc<T>* __restrict__ pd, const T* __restrict__ rhs, const int ldr, const int klu, const int col0, const int ncols, T* __restrict__ sp) {
        const int i = blockIdx.x;
        const pdesc<T> d = pd[i];
        const int n = d.size;
        T* const vp = sp + (long long)i * 4 * klu * klu;
        T* const vm = vp + 2 * klu * klu;

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
    __global__ void reduced_diag(T** __restrict__ spp, const int p, const int klu, const int klr, const int ldaR, T* __restrict__ R) {
        const int i = blockIdx.x;
        const int I0 = 2 * klu * i;
        const int off = klr + klr;
        const int ld = 2 * klu;
        const T* vp_i = spp[i];
        const T* vm_i = spp[i] + 2 * klu * klu;
        const T* vp_n = spp[i + 1];
        const T* vm_n = spp[i + 1] + 2 * klu * klu;

        const auto ridx = [&](const int I, const int J) -> T& { return R[(long long)J * ldaR + off + (I - J)]; };
        for(auto e = (long long)threadIdx.x; e < (long long)klu * klu; e += blockDim.x) {
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
    __global__ void reduced_rhs(const T* __restrict__ grj, const int m, const int klu, T* __restrict__ rhs, const int nrhs) {
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
    __global__ void correct_rhs(const pdesc<T>* __restrict__ pd, const T* __restrict__ bc, const T* __restrict__ cc, const T* __restrict__ u, const int klu, const int m, const int p, T* __restrict__ rhs, const int ldr, const int nrhs) {
        const int i = blockIdx.x;
        const int c = blockIdx.y;
        if(c >= nrhs) return;
        const pdesc<T> d = pd[i];
        const int n = d.size;
        T* const col = rhs + d.row0 + (long long)c * ldr;

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

    // Full-spike correction: x = x1 - Vp_i * w - Vm_i * z, where the spike
    // matrices were saved in full during factorization.  fs is n x 2*klu,
    // column major: column j is the solve against coupling column j.
    template<class T>
    __global__ void full_correct(const pdesc<T>* __restrict__ pd, const T* __restrict__ fs, const T* __restrict__ u, const T* __restrict__ base, int klu, const int m, const int p, const int nglob, T* __restrict__ x, const int ldx, const int nrhs) {
        const int i = blockIdx.x;
        const int c = blockIdx.y;
        if(c >= nrhs) return;
        const pdesc<T> d = pd[i];
        const int n = d.size;
        const T* z = (i > 0) ? u + (long long)(2 * klu * (i - 1)) + (long long)c * m : nullptr;
        const T* w = (i < p - 1) ? u + (long long)(2 * klu * i + klu) + (long long)c * m : nullptr;
        // Caller adds 16 elements of padding for 16-byte vector loads.
        extern __shared__ unsigned char sbuf[];
        T* const sw = reinterpret_cast<T*>(sbuf);
        T* const sz = sw + klu;
        for(int j = threadIdx.x; j < klu; j += blockDim.x) {
            sw[j] = w ? w[j] : T(0);
            sz[j] = z ? z[j] : T(0);
        }
        __syncthreads();
        for(int r = threadIdx.x; r < n; r += blockDim.x) {
            const long long gr = (long long)d.row0 + r;
            T s = T(0);
            if(w) {
                for(int j = 0; j < klu; ++j)
                    s = fma(fs[(long long)j * nglob + gr], sw[j], s);
            }
            if(z) {
                for(int j = 0; j < klu; ++j)
                    s = fma(fs[((long long)klu + j) * nglob + gr], sz[j], s);
            }
            x[gr + (long long)c * ldx] = base[gr + (long long)c * nglob] - s;
        }
    }

    // ---------------------------------------------------------------------------
    // Extract top/bottom klu rows of the local solution g_i into grj.
    // ---------------------------------------------------------------------------
    template<class T>
    __global__ void extract_corners(const pdesc<T>* __restrict__ pd, const T* __restrict__ src, const int lds, int klu, const int nrhs, T* __restrict__ grj) {
        const int i = blockIdx.x;
        const int c = blockIdx.y * blockDim.x + threadIdx.x;
        if(c >= nrhs) return;
        const pdesc<T> d = pd[i];
        const int n = d.size;
        T* const g = grj + (long long)i * 2 * klu * nrhs + (long long)c * 2 * klu;
        for(int a = 0; a < klu; ++a) {
            g[a] = src[d.row0 + a + (long long)c * lds];
            g[klu + a] = src[d.row0 + n - klu + a + (long long)c * lds];
        }
    }

} // namespace gspike::dk
