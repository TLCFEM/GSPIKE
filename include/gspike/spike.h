// gspike -- GPU (CUDA) SPIKE banded solver.
//
// Solves A * X = B where A is banded with lower bandwidth kl and upper
// bandwidth ku, provided in LAPACK band storage:
//   A(i,j) is stored at A[lda * j + ku + i - j],  with lda = kl + ku + 1,
//   for i - j in [-ku, kl].  A is treated as read-only.
//
// All pointers (A, B) refer to device memory.  Only float and double are
// instantiated; the header itself is free of CUDA types.
//
// A is captured by pointer, never copied: it must stay alive and unchanged
// while a solver object exists.  CUDA/cuBLAS failures are reported as
// std::runtime_error.
//
// Sizes, strides and offsets use gspike::index_t, which is a 32-bit integer
// by default and a 64-bit integer when the library is configured with
// GSPIKE_INDEX_TYPE=int64 (see gspike/index.h and the README).  Both the
// header and the library must be built with the same setting.
#pragma once

#include "gspike/index.h"

namespace gspike {

    // Tunables.  Defaults are reasonable for every problem.
    struct opts {
        int parts = 0;        // number of SPIKE partitions, 0 => automatic
        int max_refine = 3;   // solver: iterative refinement steps, <=0 disables
        int mixed_iters = 2;  // mixed_solver: float solves (>=1), the first is the
                              // solution and the rest are corrections
        double rtol = 0.0;    // refinement target (relative 2-norm), 0 => eps-based
        int block = 256;      // threads per CUDA block
        bool verbose = false;
    };

    struct stats {
        int parts = 0;         // partitions actually used
        int refine = 0;        // refinement steps performed
        double residual = 0.0; // achieved relative residual (2-norm)
    };

    // ---------------------------------------------------------------------------
    // One-shot factorize + solve:  A * X = B  with A banded (n x n, kl/ku) and
    // B (ldb x nrhs) column major.  B is overwritten with X.
    //
    // info = 0        success
    // info = -1       an argument had an illegal value
    // info = -2/-3/-4 B is null, ldb < n, or n*nrhs exceeds the internal limits
    // info > 0        a zero pivot was met at row info of the factorization; values
    //                 <= n are global rows of A, larger ones belong to the reduced
    //                 interface system.  The pivot was perturbed, so check the
    //                 residual (stats::residual) before trusting the result.
    // ---------------------------------------------------------------------------
    template<class T>
    int spike_gbsv(index_t n, index_t kl, index_t ku, index_t nrhs, const T* A, index_t lda, T* B, index_t ldb, const opts& o = opts(), stats* st = nullptr);

    // ---------------------------------------------------------------------------
    // Split interface: factor once, solve for many right-hand sides.
    //
    // The factorization owns internal scratch space, so solve() calls on the same
    // object must not overlap (one thread at a time).  Different solver objects
    // are independent.
    // ---------------------------------------------------------------------------
    template<class T>
    class solver {
    public:
        solver(index_t n, index_t kl, index_t ku, const T* A, index_t lda, const opts& o = opts());
        ~solver();
        solver(const solver&) = delete;
        solver& operator=(const solver&) = delete;

        // B is (ldb x nrhs); overwritten with the solution.  Return value as above.
        int solve(index_t nrhs, T* B, index_t ldb, stats* st = nullptr) const;

        // One direct solve with the existing factorization: RHS <- A^-1 RHS,
        // where RHS is (ldr x nrhs) column major with n rows used.  No
        // refinement and no copy; this is what a correction step of a
        // mixed-precision iteration needs.  Throws std::invalid_argument on bad
        // arguments.
        void solve_direct(index_t nrhs, T* RHS, index_t ldr) const;

        // Status of the factorization: 0, or the first perturbed pivot row.
        [[nodiscard]] int info() const;

        [[nodiscard]] index_t n() const;
        [[nodiscard]] index_t kl() const;
        [[nodiscard]] index_t ku() const;
        [[nodiscard]] int parts() const;

    private:
        struct impl;
        impl* p_;
    };

    // ---------------------------------------------------------------------------
    // Mixed precision: solve a double system with a float factorization.
    //
    // A is copied to float and factorized once (same SPIKE partitioning and
    // options as solver<float>).  solve() then builds the solution in double
    // precision by iterative refinement: the residual b - A x is formed and the
    // solution is accumulated in double, while every correction is a solve with
    // the float factors.  opts::mixed_iters (default 2) limits the number of
    // float solves: the first produces A_f^-1 b and each further one corrects
    // it.  The factorization is reused, so extra right-hand sides cost about one
    // float solve each.
    //
    // A must stay alive and unchanged; B is overwritten with X.  Return values
    // follow spike_gbsv and stats::refine counts the float solves kept.
    // ---------------------------------------------------------------------------
    class mixed_solver {
    public:
        mixed_solver(index_t n, index_t kl, index_t ku, const double* A, index_t lda, const opts& o = opts());
        ~mixed_solver();
        mixed_solver(const mixed_solver&) = delete;
        mixed_solver& operator=(const mixed_solver&) = delete;

        // B is (ldb x nrhs) double; overwritten with the solution.
        int solve(index_t nrhs, double* B, index_t ldb, stats* st = nullptr) const;

        [[nodiscard]] index_t n() const;
        [[nodiscard]] index_t kl() const;
        [[nodiscard]] index_t ku() const;
        [[nodiscard]] int parts() const;

    private:
        struct impl;
        impl* p_;
    };

    // Factor + solve with mixed_solver in one call.
    int spike_gbsv_mixed(index_t n, index_t kl, index_t ku, index_t nrhs, const double* A, index_t lda, double* B, index_t ldb, const opts& o = opts(), stats* st = nullptr);

    extern template class solver<float>;
    extern template class solver<double>;
    extern template int spike_gbsv<float>(index_t, index_t, index_t, index_t, const float*, index_t, float*, index_t, const opts&, stats*);
    extern template int spike_gbsv<double>(index_t, index_t, index_t, index_t, const double*, index_t, double*, index_t, const opts&, stats*);

} // namespace gspike
