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
#pragma once

#include <cstddef>

namespace gspike {

    // Tunables.  Defaults are reasonable for every problem.
    struct opts {
        int parts = 0;      // number of SPIKE partitions, 0 => automatic
        int max_refine = 3; // iterative refinement steps, <=0 disables
        double rtol = 0.0;  // refinement target (relative 2-norm), 0 => eps-based
        int block = 256;    // threads per CUDA block
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
    int spike_gbsv(int n, int kl, int ku, int nrhs, const T* A, int lda, T* B, int ldb, const opts& o = opts(), stats* st = nullptr);

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
        solver(int n, int kl, int ku, const T* A, int lda, const opts& o = opts());
        ~solver();
        solver(const solver&) = delete;
        solver& operator=(const solver&) = delete;

        // B is (ldb x nrhs); overwritten with the solution.  Return value as above.
        int solve(int nrhs, T* B, int ldb, stats* st = nullptr) const;

        int n() const;
        int kl() const;
        int ku() const;
        int parts() const;

    private:
        struct impl;
        impl* p_;
    };

    extern template class solver<float>;
    extern template class solver<double>;
    extern template int spike_gbsv<float>(int, int, int, int, const float*, int, float*, int, const opts&, stats*);
    extern template int spike_gbsv<double>(int, int, int, int, const double*, int, double*, int, const opts&, stats*);

} // namespace gspike
