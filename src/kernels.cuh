// Device kernels for the SPIKE banded solver.
#pragma once

#include <cuda_runtime.h>

namespace gspike {
    namespace dk {

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
        // Band LU with partial pivoting.  One CUDA block per independent subsystem.
        // ---------------------------------------------------------------------------
        template<class T>
        __global__ void band_lu(const pdesc<T>* __restrict__ pd, int lda, int kl, int ku, int* __restrict__ info) {
            const pdesc<T> d = pd[blockIdx.x];
            const int n = d.size;
            const int off = kl + ku;
            const int tid = threadIdx.x;
            const int nt = blockDim.x;
            const int tx = tid & 31;
            const int ty = tid >> 5;
            const int nwarps = (nt + 31) / 32;
            const int kmax = min(kl + ku, n - 1); // pivoting fills the U band up to kl+ku
            T* __restrict__ ab = d.ab;
            int* __restrict__ piv = d.piv;
            int bad = 0;

            for(int j = 0; j < n; ++j) {
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

                for(int k = j + 1 + ty; k <= kend; k += nwarps) {
                    T* ck = ab + (long long)k * lda - k + off;
                    const T u = ck[j];
                    for(int i = j + 1 + tx; i <= iend; i += 32) ck[i] -= cj[i] * u;
                }
                __syncthreads();
            }
            if(bad) atomicMin(info, bad);
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
            for(int i = n - 1; i >= 0; --i) { // back: U x = y
                const T* ab = d.ab + off;
                T s = r[i];
                const int jend = min(n - 1, i + kl + ku);
                for(int j = i + 1; j <= jend; ++j) s -= ab[(long long)j * lda + i - j] * r[j];
                r[i] = s / ab[(long long)i * lda];
            }
        }

        // ---------------------------------------------------------------------------
        // Solve (LU) X = RHS in place with one warp per right-hand side.  The rows
        // of each substitution step are independent, so the lanes split the band
        // and the back substitution reduces the band contribution with a warp
        // shuffle.  This path is used for wide bands, where a single thread per
        // right-hand side leaves the GPU mostly idle.
        // ---------------------------------------------------------------------------
        template<class T>
        __global__ void band_solve_warp(const pdesc<T>* __restrict__ pd, int lda, int kl, int ku, T* __restrict__ rhs, int ldr, int ncols, int warps) {
            const int lane = threadIdx.x & 31;
            const int c = blockIdx.y * warps + (threadIdx.x >> 5);
            if(c >= ncols) return;
            const pdesc<T> d = pd[blockIdx.x];
            const int n = d.size;
            const int off = kl + ku;
            T* __restrict__ r = rhs + d.row0 + (long long)c * ldr;
            const T* __restrict__ ab = d.ab + off;

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
            for(int i = n - 1; i >= 0; --i) { // back: U x = y
                const int jend = min(n - 1, i + kl + ku);
                T s = (lane == 0) ? r[i] : T(0);
                for(int j = i + 1 + lane; j <= jend; j += 32) s -= ab[(long long)j * lda + i - j] * r[j];
                for(int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
                if(lane == 0) r[i] = s / ab[(long long)i * lda];
                __syncwarp();
            }
        }

        // dst(n x cols, ld ldd) = src(lds x cols)
        template<class T>
        __global__ void copy_mat(int n, int cols, const T* __restrict__ src, int lds, T* __restrict__ dst, int ldd) {
            const int c = blockIdx.x;
            for(int i = threadIdx.x; i < n; i += blockDim.x)
                dst[(long long)c * ldd + i] = src[(long long)c * lds + i];
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

            for(int j = threadIdx.x; j < n; j += blockDim.x) {
                const T* src = A + (long long)(d.row0 + j) * ldaA;
                T* dst = d.ab + (long long)j * ldaF;
                for(int dd = 0; dd < kl; ++dd) dst[dd] = T(0);
                for(int dd = 0; dd <= nserial; ++dd) {
                    const int localrow = j + dd - ku; // row inside this diagonal block
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

            for(int e = threadIdx.x; e < klu * klu; e += blockDim.x) { // B_i
                const int a = e % klu, b = e / klu;
                const long long gr = d.row0 + a;
                const long long gc = (long long)d.row0 - klu + b;
                const long long off = gr - gc;
                if(gc < 0 || off < -ku || off > kl) continue;
                bci[(long long)a + (long long)b * klu] = A[gc * ldaA + ku + off];
            }
            for(int e = threadIdx.x; e < klu * klu; e += blockDim.x) { // C_i
                const int a = e % klu, b = e / klu;
                const long long gr = (long long)d.row0 + n - klu + a;
                const long long gc = (long long)d.row0 + n + b;
                const long long off = gr - gc;
                if(gc >= N || off < -ku || off > kl) continue;
                cci[(long long)a + (long long)b * klu] = A[gc * ldaA + ku + off];
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

#define RIDX(I, J) R[(long long)(J) * ldaR + off + ((I) - (J))]
            for(long long e = (long long)threadIdx.x; e < (long long)klu * klu; e += blockDim.x) {
                const int a = (int)(e % klu);
                const int b = (int)(e / klu);
                const long long s = (long long)a + (long long)b * ld;
                RIDX(I0 + a, I0 + b) = (a == b) ? T(1) : T(0);
                RIDX(I0 + klu + a, I0 + klu + b) = (a == b) ? T(1) : T(0);
                // D_i: top-right = Vp_i^bot, bottom-left = Vm_{i+1}^top
                RIDX(I0 + a, I0 + klu + b) = vp_i[s + klu];
                RIDX(I0 + klu + a, I0 + b) = vm_n[s];
                // L_i: Vm_i^bot (row a, column b of previous block, first half)
                if(i > 0)
                    RIDX(I0 + a, I0 - 2 * klu + b) = vm_i[s + klu];
                // U_i: Vp_{i+1}^top (second half of block i+1)
                if(i + 1 <= p - 2)
                    RIDX(I0 + klu + a, I0 + 2 * klu + klu + b) = vp_n[s];
            }
#undef RIDX
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
            const int c = blockIdx.y * blockDim.x + threadIdx.x;
            if(c >= nrhs) return;
            const pdesc<T> d = pd[i];
            const int n = d.size;
            T* col = rhs + d.row0 + (long long)c * ldr;

            const T* bci = bc + (long long)i * klu * klu;
            const T* cci = cc + (long long)i * klu * klu;
            const T* z = (i > 0) ? u + (long long)(2 * klu * (i - 1)) + (long long)c * m : nullptr;
            const T* w = (i < p - 1) ? u + (long long)(2 * klu * i + klu) + (long long)c * m : nullptr;

            if(w) {
                for(int a = 0; a < klu; ++a) {
                    T s = T(0);
                    for(int j = 0; j < klu; ++j) s += cci[(long long)a + (long long)j * klu] * w[j];
                    col[n - klu + a] -= s;
                }
            }
            if(z) {
                for(int a = 0; a < klu; ++a) {
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

    } // namespace dk
} // namespace gspike
