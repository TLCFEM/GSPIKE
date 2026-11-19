// Device kernels for the SPIKE banded solver.
//
// Every kernel is templated on the scalar type T and on the index type I
// (gspike::index_t), which is used for all sizes, strides and offsets so that
// 32-bit and 64-bit builds share the same code.
#pragma once

#include <climits>
#include <cuda_runtime.h>

namespace gspike::dk {

    template<class I>
    __device__ __forceinline__ I imin(const I a, const I b) {
        return a < b ? a : b;
    }
    template<class I>
    __device__ __forceinline__ I imax(const I a, const I b) {
        return a > b ? a : b;
    }

    // Per-subsystem description (one band LU factorization block).
    template<class T, class I>
    struct pdesc {
        T* ab;  // band factors: (2*kl+ku+1) x size, column major,
                // a(i,j) at ab[(kl+ku+i-j) + j*lda]
        I* piv; // pivot row (local index) chosen at each column
        I row0; // first global row of the block (RHS buffers indexed by global row)
        I size; // dimension of the block
    };

    // ---------------------------------------------------------------------------
    // Band LU with partial pivoting.  One CUDA block per independent subsystem.
    //
    // Warp 0 reduces the pivot search, the normalization uses every thread, and
    // the trailing update maps one warp to one column, with the normalized pivot
    // row staged in shared memory (when kl fits) so it is loaded once per step.
    // ---------------------------------------------------------------------------
    template<class T, class I>
    __global__ void band_lu(const pdesc<T, I>* __restrict__ pd, const I lda, const I kl, const I ku, int* __restrict__ info) {
        extern __shared__ __align__(16) unsigned char shraw[];
        const pdesc<T, I> d = pd[blockIdx.x];
        const I n = d.size;
        const I off = kl + ku;
        const int tid = threadIdx.x;
        const int nt = blockDim.x;
        const int lane = tid & 31;
        const int warp = tid >> 5;
        const int nwarp = (int)(nt >> 5);
        const I kmax = imin<I>(kl + ku, n - 1);
        const bool stage = (size_t)kl * sizeof(T) <= (48u << 10);
        T* __restrict__ sh = reinterpret_cast<T*>(shraw);
        T* __restrict__ ab = d.ab;
        I* __restrict__ piv = d.piv;
        int bad = 0;

        for(I j = 0; j < n; ++j) {
            const I iend = imin<I>(j + kl, n - 1);
            T* cj = ab + j * lda - j + off;

            if(warp == 0) { // pivot search, reduced across the warp
                T bv = T(0);
                I pv = j;
                for(I i = j + lane; i <= iend; i += 32) {
                    const T v = fabs(cj[i]);
                    if(v > bv) {
                        bv = v;
                        pv = i;
                    }
                }
                for(int o = 16; o > 0; o >>= 1) {
                    const T ov = __shfl_down_sync(0xffffffffu, bv, o);
                    const I op = __shfl_down_sync(0xffffffffu, pv, o);
                    if(ov > bv) {
                        bv = ov;
                        pv = op;
                    }
                }
                if(lane == 0) {
                    if(!(bv > T(0))) { // exactly singular column: perturb it, remember the first one
                        if(bad == 0) bad = (int)imin<I>(d.row0 + j + 1, (I)INT_MAX);
                        cj[pv] = T(1);
                    }
                    piv[j] = pv;
                }
            }
            __syncthreads();

            const I pv = piv[j];
            const I kend = imin<I>(n - 1, j + kmax);

            if(pv != j) {
                for(I k = j + tid; k <= kend; k += nt) {
                    T* ck = ab + k * lda - k + off;
                    const T t = ck[j];
                    ck[j] = ck[pv];
                    ck[pv] = t;
                }
                __syncthreads();
            }

            const T pval = cj[j];
            for(I i = j + 1 + tid; i <= iend; i += nt) {
                const T t = cj[i] / pval;
                cj[i] = t;
                if(stage) sh[i - j] = t;
            }
            __syncthreads();

            for(I k = j + 1 + warp; k <= kend; k += nwarp) {
                T* ck = ab + k * lda - k + off;
                const T u = ck[j];
                for(I i = j + 1 + lane; i <= iend; i += 32) ck[i] -= (stage ? sh[i - j] : cj[i]) * u;
            }
            __syncthreads();
        }
        if(bad) atomicMin(info, bad);
    }

    // ---------------------------------------------------------------------------
    // Solve (LU) X = RHS in place.  One thread per right-hand side column.
    // RHS is indexed by global row: column c, row r -> rhs[d.row0 + r + c*ldr].
    // ---------------------------------------------------------------------------
    template<class T, class I>
    __global__ void band_solve(const pdesc<T, I>* __restrict__ pd, const I lda, const I kl, const I ku, T* __restrict__ rhs, const I ldr, const I ncols) {
        const I c = (I)blockIdx.y * (I)blockDim.x + (I)threadIdx.x;
        if(c >= ncols) return;
        const pdesc<T, I> d = pd[blockIdx.x];
        const I n = d.size;
        const I off = kl + ku;
        T* __restrict__ r = rhs + d.row0 + c * ldr;

        for(I j = 0; j < n; ++j) { // forward: L y = P b
            const I pv = d.piv[j];
            if(pv != j) {
                const T t = r[j];
                r[j] = r[pv];
                r[pv] = t;
            }
            const T rj = r[j];
            const T* cj = d.ab + j * lda - j + off;
            const I iend = imin<I>(j + kl, n - 1);
            for(I i = j + 1; i <= iend; ++i) r[i] -= cj[i] * rj;
        }
        for(I i = n - 1; i >= 0; --i) { // back: U x = y
            const T* ab = d.ab + off;
            T s = r[i];
            const I jend = imin<I>(n - 1, i + kl + ku);
            for(I j = i + 1; j <= jend; ++j) s -= ab[j * lda + i - j] * r[j];
            r[i] = s / ab[i * lda];
        }
    }

    // ---------------------------------------------------------------------------
    // Warp-cooperative solve (LU) X = RHS in place.  One warp per right-hand side
    // column, several warps per CUDA block.  Same contract as band_solve:
    // column c, row r -> rhs[d.row0 + r + c*ldr].
    //
    // Forward substitution: lane 0 applies the pivot swap, then every lane
    // updates one row of the band, so the factors are read coalesced.  Back
    // substitution: every lane owns one column of the U row, the contributions
    // are reduced with a shuffle, and lane 0 writes the result.
    //
    // With UseSmem the RHS column is staged in shared memory (smem_ld elements
    // per warp, at least n) so that the row recurrence does not depend on L1.
    // ---------------------------------------------------------------------------
    template<class T, class I, bool UseSmem>
    __global__ void band_solve_warp(const pdesc<T, I>* __restrict__ pd, const I lda, const I kl, const I ku, T* __restrict__ rhs, const I ldr, const I ncols, const I smem_ld) {
        extern __shared__ __align__(16) unsigned char smem_raw[];
        const pdesc<T, I> d = pd[blockIdx.x];
        const I n = d.size;
        const I off = kl + ku;
        const int lane = threadIdx.x & 31;
        const I c = (I)blockIdx.y * (I)(blockDim.x >> 5) + (I)(threadIdx.x >> 5);
        if(c >= ncols) return;
        T* __restrict__ rg = rhs + d.row0 + c * ldr;
        T* __restrict__ r = rg;
        if constexpr(UseSmem) {
            r = reinterpret_cast<T*>(smem_raw) + (size_t)(threadIdx.x >> 5) * (size_t)smem_ld;
            for(I i = (I)lane; i < n; i += 32) r[i] = rg[i];
            __syncwarp();
        }
        const T* __restrict__ u = d.ab + off;

        for(I j = 0; j < n; ++j) { // forward: L y = P b
            __syncwarp();          // updates of the previous row are visible
            const I pv = d.piv[j];
            const T old = r[j];
            const T rj = (pv == j) ? old : r[pv];
            if(lane == 0 && pv != j) {
                r[j] = rj;
                r[pv] = old;
            }
            __syncwarp(); // swap visible before the rows are updated
            const T* cj = d.ab + j * lda - j + off;
            const I iend = imin<I>(j + kl, n - 1);
            for(I base = j + 1; base <= iend; base += 32) {
                const I i = base + (I)lane;
                if(i <= iend) r[i] -= cj[i] * rj;
            }
        }

        for(I i = n - 1; i >= 0; --i) { // back: U x = y
            const I jend = imin<I>(n - 1, i + kl + ku);
            T s = (lane == 0) ? r[i] : T(0);
            for(I base = i + 1; base <= jend; base += 32) {
                const I j = base + (I)lane;
                if(j <= jend) s -= u[j * lda + i - j] * r[j];
            }
            for(int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
            if(lane == 0) r[i] = s / u[i * lda];
            __syncwarp();
        }

        if constexpr(UseSmem) {
            __syncwarp();
            for(I i = (I)lane; i < n; i += 32) rg[i] = r[i];
        }
    }

    // ---------------------------------------------------------------------------
    // y = A x for banded A (lda = kl+ku+1, offset ku).  One block per column.
    // ---------------------------------------------------------------------------
    template<class T, class I>
    __global__ void band_matvec(const I n, const I kl, const I ku, const T* __restrict__ A, const I lda, const T* __restrict__ x, const I ldx, T* __restrict__ y, const I ldy, const I nrhs) {
        const I c = blockIdx.x;
        const I i = (I)blockIdx.y * (I)blockDim.x + (I)threadIdx.x;
        if(i >= n) return;
        const T* xc = x + c * ldx;
        T s = T(0);
        const I j0 = imax<I>(I(0), i - kl);
        const I j1 = imin<I>(n - 1, i + ku);
        for(I j = j0; j <= j1; ++j) s += A[j * lda + ku + i - j] * xc[j];
        y[c * ldy + i] = s;
    }

    // ---------------------------------------------------------------------------
    // Small elementwise helpers.
    // ---------------------------------------------------------------------------
    template<class T, class I>
    __global__ void axpby(const I n, T a, const T* __restrict__ x, T b, T* __restrict__ y) {
        for(I i = (I)blockIdx.x * (I)blockDim.x + (I)threadIdx.x; i < n; i += (I)gridDim.x * (I)blockDim.x)
            y[i] = a * x[i] + b * y[i];
    }

    // dst(n x cols, ld ldd) = src(lds x cols).  blockIdx.x is the column and
    // blockIdx.y strides over the rows, so a single column still spreads over
    // many blocks instead of one.
    template<class T, class I>
    __global__ void copy_mat(const I n, const I cols, const T* __restrict__ src, const I lds, T* __restrict__ dst, const I ldd) {
        const I c = blockIdx.x;
        const I step = (I)gridDim.y * (I)blockDim.x;
        for(I i = (I)blockIdx.y * (I)blockDim.x + (I)threadIdx.x; i < n; i += step)
            dst[c * ldd + i] = src[c * lds + i];
    }

    // dst1 = dst2 = src for every column: the working copy and the right-hand
    // side saved for the residual in one pass.
    template<class T, class I>
    __global__ void copy_mat_dup(const I n, const T* __restrict__ src, const I lds, T* __restrict__ dst1, T* __restrict__ dst2, const I ldd) {
        const I c = blockIdx.x;
        const I step = (I)gridDim.y * (I)blockDim.x;
        for(I i = (I)blockIdx.y * (I)blockDim.x + (I)threadIdx.x; i < n; i += step) {
            const T v = src[c * lds + i];
            dst1[c * ldd + i] = v;
            dst2[c * ldd + i] = v;
        }
    }

    // Mixed precision helpers: the same elementwise loop with a change of scalar
    // type, used to move a double right-hand side into single precision and the
    // single-precision correction back.
    template<class To, class From, class I>
    __global__ void cast_copy(const I n, const From* __restrict__ src, To* __restrict__ dst) {
        for(I i = (I)blockIdx.x * (I)blockDim.x + (I)threadIdx.x; i < n; i += (I)gridDim.x * (I)blockDim.x)
            dst[i] = (To)src[i];
    }

    // dst <- dst + alpha * (To)src
    template<class To, class From, class I>
    __global__ void cast_axpy(const I n, const To alpha, const From* __restrict__ src, To* __restrict__ dst) {
        for(I i = (I)blockIdx.x * (I)blockDim.x + (I)threadIdx.x; i < n; i += (I)gridDim.x * (I)blockDim.x)
            dst[i] += alpha * (To)src[i];
    }

    // x <- b - x  (both contiguous, n elements)
    template<class T, class I>
    __global__ void bsub(const I n, const T* __restrict__ b, T* __restrict__ x) {
        for(I i = (I)blockIdx.x * (I)blockDim.x + (I)threadIdx.x; i < n; i += (I)gridDim.x * (I)blockDim.x)
            x[i] = b[i] - x[i];
    }

    // ---------------------------------------------------------------------------
    // Extract diagonal + coupling blocks of partition blockIdx.x from A.
    // Interface vectors are klu long, so both coupling blocks are klu x klu:
    //   bc[a + b*klu] : A(row0 + a,        row0 - klu + b)      (previous block)
    //   cc[a + b*klu] : A(row0 + n - klu + a, row0 + n + b)     (next block)
    // ---------------------------------------------------------------------------
    template<class T, class I>
    __global__ void make_blocks(const pdesc<T, I>* __restrict__ pd, const T* __restrict__ A, const I N, const I ldaA, const I ldaF, const I ku, const I kl, const I klu, T* __restrict__ bc, T* __restrict__ cc) {
        const I i = blockIdx.x;
        const pdesc<T, I> d = pd[i];
        const I n = d.size;
        const I nserial = kl + ku;

        for(I j = threadIdx.x; j < n; j += blockDim.x) {
            const T* src = A + (d.row0 + j) * ldaA;
            T* dst = d.ab + j * ldaF;
            for(I dd = 0; dd < kl; ++dd) dst[dd] = T(0);
            for(I dd = 0; dd <= nserial; ++dd) {
                const I localrow = j + dd - ku; // row inside this diagonal block
                dst[kl + dd] = (localrow >= 0 && localrow < n) ? src[dd] : T(0);
            }
        }

        T* bci = bc + i * klu * klu;
        T* cci = cc + i * klu * klu;
        for(I e = threadIdx.x; e < klu * klu; e += blockDim.x) {
            bci[e] = T(0);
            cci[e] = T(0);
        }
        __syncthreads();

        for(I e = threadIdx.x; e < klu * klu; e += blockDim.x) { // B_i
            const I a = e % klu, b = e / klu;
            const I gr = d.row0 + a;
            const I gc = d.row0 - klu + b;
            const I off = gr - gc;
            if(gc < 0 || off < -ku || off > kl) continue;
            bci[a + b * klu] = A[gc * ldaA + ku + off];
        }
        for(I e = threadIdx.x; e < klu * klu; e += blockDim.x) { // C_i
            const I a = e % klu, b = e / klu;
            const I gr = d.row0 + n - klu + a;
            const I gc = d.row0 + n + b;
            const I off = gr - gc;
            if(gc >= N || off < -ku || off > kl) continue;
            cci[a + b * klu] = A[gc * ldaA + ku + off];
        }
    }

    // ---------------------------------------------------------------------------
    // Build the coupling right-hand side for a chunk of spike columns (2*klu total):
    //   spike column c in [0,klu)      -> C block (last klu rows of the block)
    //   spike column c in [klu,2*klu)  -> B block (first klu rows of the block)
    // ---------------------------------------------------------------------------
    template<class T, class I>
    __global__ void spike_rhs(const pdesc<T, I>* __restrict__ pd, const T* __restrict__ bc, const T* __restrict__ cc, const I klu, T* __restrict__ rhs, const I ldr, const I col0, const I ncol) {
        const pdesc<T, I> d = pd[blockIdx.x];
        const I n = d.size;
        const I lc = blockIdx.y; // column inside this chunk
        const I c = col0 + lc;   // global spike column
        T* col = rhs + d.row0 + lc * ldr;
        for(I r = threadIdx.x; r < n; r += blockDim.x) col[r] = T(0);
        __syncthreads();
        if(c >= ncol) return;

        const T* bci = bc + (I)blockIdx.x * klu * klu;
        const T* cci = cc + (I)blockIdx.x * klu * klu;
        if(c < klu) {
            for(I a = threadIdx.x; a < klu; a += blockDim.x)
                col[n - klu + a] = cci[a + c * klu];
        }
        else {
            const I j = c - klu;
            for(I a = threadIdx.x; a < klu; a += blockDim.x)
                col[a] = bci[a + j * klu];
        }
    }

    // ---------------------------------------------------------------------------
    // Copy the top/bottom klu rows of a solved spike block into Vp / Vm of every
    // partition (sp = p blocks of [Vp | Vm], each 2klu x klu column major).
    //   Vp column j comes from spike column j,  Vm column j from column klu+j.
    // ---------------------------------------------------------------------------
    template<class T, class I>
    __global__ void spike_extract(const pdesc<T, I>* __restrict__ pd, const T* __restrict__ rhs, const I ldr, const I klu, const I col0, const I ncols, T* __restrict__ sp) {
        const I i = blockIdx.x;
        const pdesc<T, I> d = pd[i];
        const I n = d.size;
        T* vp = sp + i * 4 * klu * klu;
        T* vm = vp + 2 * klu * klu;

        const I idx = (I)blockIdx.y * (I)blockDim.x + (I)threadIdx.x;
        const I j = idx % klu; // spike column
        const I a = idx / klu; // spike row
        if(a >= 2 * klu) return;
        const I row = (a < klu) ? a : (n - klu + (a - klu));

        const I lp = j - col0;
        if(lp >= 0 && lp < ncols)
            vp[a + j * 2 * klu] = rhs[d.row0 + row + lp * ldr];
        const I lm = (klu + j) - col0;
        if(lm >= 0 && lm < ncols)
            vm[a + j * 2 * klu] = rhs[d.row0 + row + lm * ldr];
    }

    // ---------------------------------------------------------------------------
    // Assemble the reduced (interface) matrix, in band storage by default and
    // in dense column-major storage (ldaR = m) for the cuSOLVER path.
    // Block i (i = 0..p-2) holds u_i = [ z_i ; w_{i+1} ], 2*klu unknowns:
    //   D_i = [[I, Vp_i^bot], [Vm_{i+1}^top, I]]
    //   L_i = [[Vm_i^bot, 0], [0, 0]]          couples to block i-1
    //   U_i = [[0, 0], [0, Vp_{i+1}^top]]      couples to block i+1
    // Spike columns are 2*klu wide, so the sub-block entries use that stride.
    // ---------------------------------------------------------------------------
    template<class T, class I, bool Dense>
    __global__ void reduced_diag(T** __restrict__ spp, const int p, const I klu, const I klr, const I ldaR, T* __restrict__ R) {
        const I i = blockIdx.x;
        const I I0 = 2 * klu * i;
        const I off = klr + klr;
        const I ld = 2 * klu;
        const T* vp_i = spp[i];
        const T* vm_i = spp[i] + 2 * klu * klu;
        const T* vp_n = spp[i + 1];
        const T* vm_n = spp[i + 1] + 2 * klu * klu;

        // Dense layout: R(row,col) = R[col*ldaR + row] (ldaR = m).  Band layout:
        // R(row,col) = R[col*ldaR + 2*klr + row - col].
        auto at = [&](const I row, const I col) -> T& {
            if constexpr(Dense) return R[col * ldaR + row];
            else return R[col * ldaR + off + (row - col)];
        };
        for(I e = threadIdx.x; e < klu * klu; e += blockDim.x) {
            const I a = e % klu;
            const I b = e / klu;
            const I s = a + b * ld;
            at(I0 + a, I0 + b) = (a == b) ? T(1) : T(0);
            at(I0 + klu + a, I0 + klu + b) = (a == b) ? T(1) : T(0);
            // D_i: top-right = Vp_i^bot, bottom-left = Vm_{i+1}^top
            at(I0 + a, I0 + klu + b) = vp_i[s + klu];
            at(I0 + klu + a, I0 + b) = vm_n[s];
            // L_i: Vm_i^bot (row a, column b of previous block, first half)
            if(i > 0)
                at(I0 + a, I0 - 2 * klu + b) = vm_i[s + klu];
            // U_i: Vp_{i+1}^top (second half of block i+1)
            if(i + 1 <= p - 2)
                at(I0 + klu + a, I0 + 2 * klu + klu + b) = vp_n[s];
        }
    }

    // ---------------------------------------------------------------------------
    // Reduced right-hand side: rows [0,klu) of block i take the bottom half of
    // grj(i); rows [klu,2klu) take the top half of grj(i+1).
    //   grj: p * (2klu x nrhs), column major with ld 2*klu
    //   rhs: (m x nrhs), column major with ld m
    // ---------------------------------------------------------------------------
    template<class T, class I>
    __global__ void reduced_rhs(const T* __restrict__ grj, const I m, const I klu, T* __restrict__ rhs, const I nrhs) {
        const I i = blockIdx.x;
        const I c = (I)blockIdx.y * (I)blockDim.x + (I)threadIdx.x;
        if(c >= nrhs || i >= m / (2 * klu)) return;
        const I gcur = i * 2 * klu * nrhs + c * 2 * klu;
        const I gnext = gcur + 2 * klu * nrhs;
        for(I a = 0; a < klu; ++a) {
            rhs[(2 * klu * i + a) + c * m] = grj[gcur + klu + a];
            rhs[(2 * klu * i + klu + a) + c * m] = grj[gnext + a];
        }
    }

    // ---------------------------------------------------------------------------
    // Apply the interface corrections in place to the partition right-hand sides:
    //   x_i <- f_i - C_i w_{i+1} - B_i z_{i-1}
    // Only the corner rows are touched.
    // ---------------------------------------------------------------------------
    template<class T, class I>
    __global__ void correct_rhs(const pdesc<T, I>* __restrict__ pd, const T* __restrict__ bc, const T* __restrict__ cc, const T* __restrict__ u, const I klu, const I m, const I p, T* __restrict__ rhs, const I ldr, const I nrhs) {
        const I i = blockIdx.x;
        const I c = (I)blockIdx.y * (I)blockDim.x + (I)threadIdx.x;
        if(c >= nrhs) return;
        const pdesc<T, I> d = pd[i];
        const I n = d.size;
        T* col = rhs + d.row0 + c * ldr;

        const T* bci = bc + i * klu * klu;
        const T* cci = cc + i * klu * klu;
        const T* z = (i > 0) ? u + (2 * klu * (i - 1)) + c * m : nullptr;
        const T* w = (i < p - 1) ? u + (2 * klu * i + klu) + c * m : nullptr;

        if(w) {
            for(I a = 0; a < klu; ++a) {
                T s = T(0);
                for(I j = 0; j < klu; ++j) s += cci[a + j * klu] * w[j];
                col[n - klu + a] -= s;
            }
        }
        if(z) {
            for(I a = 0; a < klu; ++a) {
                T s = T(0);
                for(I j = 0; j < klu; ++j) s += bci[a + j * klu] * z[j];
                col[a] -= s;
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Extract top/bottom klu rows of the local solution g_i into grj.
    // ---------------------------------------------------------------------------
    template<class T, class I>
    __global__ void extract_corners(const pdesc<T, I>* __restrict__ pd, const T* __restrict__ src, const I lds, const I klu, const I nrhs, T* __restrict__ grj) {
        const I i = blockIdx.x;
        const I c = (I)blockIdx.y * (I)blockDim.x + (I)threadIdx.x;
        if(c >= nrhs) return;
        const pdesc<T, I> d = pd[i];
        const I n = d.size;
        T* g = grj + i * 2 * klu * nrhs + c * 2 * klu;
        for(I a = 0; a < klu; ++a) {
            g[a] = src[d.row0 + a + c * lds];
            g[klu + a] = src[d.row0 + n - klu + a + c * lds];
        }
    }

} // namespace gspike::dk
