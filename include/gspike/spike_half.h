// gspike -- mixed-precision SPIKE solver using CUDA's native half type.
//
// Parallel public interface to the half-precision implementation in
// spike_half.cu / mixed_half.cu.  The float/double API in spike.h is unchanged.
#pragma once

#include <cuda_fp16.h>
#include <memory>

#include "gspike/spike.h"

namespace gspike {

#if GSPIKE_HAS_CUDA_HALF
    // Convenience name for the explicit solver<__half> specialization
    // implemented in spike_half.cu.
    using half_solver = solver<__half>;
#endif

    // ---------------------------------------------------------------------------
    // Double matrix factorized in CUDA __half.  Each solve starts with a half
    // solve and then performs four double-residual refinement iterations;
    // that is enough to bring a half-precision factorization back to roughly
    // double accuracy on well-conditioned diagonal-dominant bands.
    //
    // A is read only here as well, and must outlive the solver because the
    // double residual is formed against the original matrix on every solve.
    // ---------------------------------------------------------------------------
    class mixed_half_solver {
    public:
        mixed_half_solver(int n, int kl, int ku, const double* A, int lda, const opts& o = opts());
        ~mixed_half_solver();
        mixed_half_solver(const mixed_half_solver&) = delete;
        mixed_half_solver& operator=(const mixed_half_solver&) = delete;

        // B is (ldb x nrhs); overwritten with the solution.  Return value as
        // for solver<T>::solve().
        int solve(int nrhs, double* B, int ldb, stats* st = nullptr) const;

        [[nodiscard]] int n() const;
        [[nodiscard]] int kl() const;
        [[nodiscard]] int ku() const;
        [[nodiscard]] int parts() const;

        // Determinant and sign from the half factorization, returned as double.
        [[nodiscard]] double det() const;
        [[nodiscard]] int det_sign() const;

    private:
        struct impl;
        std::unique_ptr<impl> p_;
    };

} // namespace gspike
