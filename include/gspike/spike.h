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
// solver captures A by pointer and never copies it, so A must stay alive and
// unchanged while the object exists.  mixed_solver keeps its own float copy of
// A.  CUDA failures are reported as std::runtime_error.
#pragma once

#include <cstddef>
#include <memory>

namespace gspike {

    // Tunables.  Defaults are reasonable for every problem.
    struct opts {
        int parts = 0;    // number of SPIKE partitions, 0 => automatic
        int block = 1024; // threads per CUDA block
        bool verbose = false;
        // Keep the complete spike solutions (2*klu vectors of length n) so a
        // solve can apply the interface correction with a matvec instead of a
        // second band solve.  Costs 2*klu*n scalars of device memory (~1 GB for
        // double at n=80000, klu=800); off by default.  Independent of accuracy.
        bool full_spikes = false;
    };

    struct stats {
        int parts = 0; // partitions actually used
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
    //                 solution residual before trusting the result.
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

        [[nodiscard]] int n() const;
        [[nodiscard]] int kl() const;
        [[nodiscard]] int ku() const;
        [[nodiscard]] int parts() const;

    private:
        struct impl;
        std::unique_ptr<impl> p_;
    };

    // ---------------------------------------------------------------------------
    // Mixed-precision interface for a double matrix.  The matrix is copied to
    // device float storage and factorized once with solver<float>.  solve()
    // performs two float solves: first for the right-hand side, then for the
    // double-precision residual against the original double matrix.  For well
    // conditioned systems the corrected result has about double precision.
    //
    // A is read once into private float storage during construction, so it only
    // has to stay valid for the constructor call.  (solver<T> keeps its pointer
    // for the lifetime of the object.)
    // ---------------------------------------------------------------------------
    class mixed_solver {
    public:
        mixed_solver(int n, int kl, int ku, const double* A, int lda, const opts& o = opts());
        ~mixed_solver();
        mixed_solver(const mixed_solver&) = delete;
        mixed_solver& operator=(const mixed_solver&) = delete;

        // B is (ldb x nrhs); overwritten with the solution.  Return value as above.
        int solve(int nrhs, double* B, int ldb, stats* st = nullptr) const;

        [[nodiscard]] int n() const;
        [[nodiscard]] int kl() const;
        [[nodiscard]] int ku() const;
        [[nodiscard]] int parts() const;

    private:
        struct impl;
        std::unique_ptr<impl> p_;
    };

    extern template class solver<float>;
    extern template class solver<double>;
    extern template int spike_gbsv<float>(int, int, int, int, const float*, int, float*, int, const opts&, stats*);
    extern template int spike_gbsv<double>(int, int, int, int, const double*, int, double*, int, const opts&, stats*);

} // namespace gspike
