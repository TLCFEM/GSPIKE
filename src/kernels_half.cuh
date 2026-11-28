// Device kernels for the SPIKE banded solver -- CUDA __half instantiation.
//
// The shared kernels (including the half-aware math wrappers) live in
// kernels.cuh.  This header adds only the residual-scaling helpers
// needed by mixed_half_solver.
#pragma once

#include "kernels.cuh"

namespace gspike::dk {

    // Largest absolute entry of each column, used to rescale residuals before
    // handing them to the half solver (half subnormals lose most of their
    // precision, so a small residual must be brought back into normal range).
    static __global__ void max_abs_double(const int n, const double* __restrict__ x, const int ldx, double* __restrict__ out) {
        const int c = blockIdx.x;
        __shared__ double sm[256];
        double m = 0;
        for(int i = threadIdx.x; i < n; i += blockDim.x) m = fmax(m, fabs(x[(long long)c * ldx + i]));
        sm[threadIdx.x] = m;
        __syncthreads();
        for(int s = 128; s > 0; s >>= 1) {
            if(threadIdx.x < s) sm[threadIdx.x] = fmax(sm[threadIdx.x], sm[threadIdx.x + s]);
            __syncthreads();
        }
        if(threadIdx.x == 0) out[c] = sm[0];
    }

    // dst = scale * src, with the conversion to the destination type applied
    // after scaling.  Used for residual/correction scaling around a half solve.
    template<class To, class From>
    static __global__ void convert_scaled_mat(const int n, const From* __restrict__ src, const int lds, To* __restrict__ dst, const int ldd, const double scale) {
        const int c = blockIdx.x;
        for(int i = threadIdx.x; i < n; i += blockDim.x)
            dst[(long long)c * ldd + i] = (To)(scale * (double)src[(long long)c * lds + i]);
    }

} // namespace gspike::dk
