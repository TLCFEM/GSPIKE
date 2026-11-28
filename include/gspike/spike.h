// gspike -- GPU (CUDA) SPIKE banded solver.
//
// Solves A * X = B where A is banded with lower bandwidth kl and upper
// bandwidth ku, provided in LAPACK band storage:
//   A(i,j) is stored at A[lda * j + ku + i - j],  with lda >= kl + ku + 1,
//   for i - j in [-ku, kl].  A is treated as read-only.
//
// All pointers (A, B) refer to device memory.  float and double are accepted
// (supported_scalar), as is std::float16_t when the toolchain provides it;
// the header itself is free of CUDA types.
//
// solver captures A by pointer and never copies it, so A must stay alive and
// unchanged while the object exists.  mixed_solver keeps a float copy for the
// factorization but reads the original double A on every solve to form the
// residual, so A must outlive it as well.  CUDA launch failures are reported
// as std::runtime_error immediately; asynchronous kernel errors surface at
// the next CUDA synchronization.
#pragma once

#include <concepts>
#include <cstdio>
#include <format>
#include <memory>
#include <utility>

// Detect std::float16_t (C++23 <stdfloat>).  GCC and Clang advertise the type
// with __STDCPP_FLOAT16_T__, which is only defined when _Float16 is available.
// The header is included conditionally so this stays valid in C++20 builds
// where <stdfloat> has nothing to declare.
#if defined(__has_include)
#if __has_include(<stdfloat>)
#include <stdfloat>
#endif
#endif
#if defined(__STDCPP_FLOAT16_T__) || (defined(__cpp_lib_stdfloat) && __cpp_lib_stdfloat >= 202106L)
#define GSPIKE_HAS_FLOAT16_T 1
#else
#define GSPIKE_HAS_FLOAT16_T 0
#endif

// CUDA's native half type is the precision actually usable in device code
// (std::float16_t is not supported there).  Forward declared so this header
// stays free of CUDA headers; the implementation lives in spike_half.cu.
struct __half;
#if defined(__CUDACC__)
#define GSPIKE_HAS_CUDA_HALF 1
#else
#define GSPIKE_HAS_CUDA_HALF 0
#endif

namespace gspike {

    // float and double are instantiated in spike.cu, CUDA __half in the
    // parallel spike_half.cu.  std::float16_t is admitted when the toolchain
    // provides it, but NVIDIA does not support it in device code.  Other types
    // are rejected at compile time rather than leaving an undefined reference.
#if GSPIKE_HAS_FLOAT16_T
#define GSPIKE_SUPPORTED_SCALARS std::same_as<T, float> || std::same_as<T, double> || std::same_as<T, std::float16_t>
#else
#define GSPIKE_SUPPORTED_SCALARS std::same_as<T, float> || std::same_as<T, double>
#endif
#if GSPIKE_HAS_CUDA_HALF
    template<class T> concept supported_scalar = GSPIKE_SUPPORTED_SCALARS || std::same_as<T, __half>;
#else
    template<class T> concept supported_scalar = GSPIKE_SUPPORTED_SCALARS;
#endif
#undef GSPIKE_SUPPORTED_SCALARS

    namespace detail {
        // Small formatted-output helpers, shared by the library, benchmarks and tests.
        template<class... Args>
        void print(std::format_string<Args...> fmt, Args&&... args) {
            std::fputs(std::format(fmt, std::forward<Args>(args)...).c_str(), stdout);
        }

        template<class... Args>
        void print_err(std::format_string<Args...> fmt, Args&&... args) {
            std::fputs(std::format(fmt, std::forward<Args>(args)...).c_str(), stderr);
        }
    } // namespace detail

    // Tunables.  Defaults are reasonable for every problem.
    struct opts {
        int parts = 0;    // number of SPIKE partitions, 0 => automatic
        int block = 1024; // threads per CUDA block
        bool verbose = false;
        // Keep the complete spike solutions (2*klu vectors of length n) so a
        // solve can apply the interface correction with a matvec instead of a
        // second band solve.  Costs 2*klu*n scalars of device memory (~1 GB for
        // double at n=80000, klu=800).  Explicitly off by default; it is
        // auto-enabled by auto_full_spikes when memory allows.  Independent
        // of accuracy.
        bool full_spikes = false;
        // Keep full spikes automatically when the extra storage is at most
        // 512 MB and the device has at least three times that much free memory.
        // This trades device memory for a 1.4-3x faster solve; set it to false
        // when memory is tight or a fixed solve path is required.
        bool auto_full_spikes = true;
        // Print per-phase GPU timings to stderr for factorization and solve.
        // Intended for profiling; adds event synchronizations when enabled.
        bool profile = false;
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
    // info > 0        the first zero pivot met during factorization; values
    //                 <= n are global rows of A, larger ones belong to the reduced
    //                 interface system.  The pivot was perturbed, so check the
    //                 solution residual before trusting the result.
    // ---------------------------------------------------------------------------
    template<supported_scalar T>
    int spike_gbsv(int n, int kl, int ku, int nrhs, const T* A, int lda, T* B, int ldb, const opts& o = opts(), stats* st = nullptr);

    // ---------------------------------------------------------------------------
    // Split interface: factor once, solve for many right-hand sides.
    //
    // The factorization owns internal scratch space, so solve() calls on the same
    // object must not overlap (one thread at a time).  Different solver objects
    // are independent.
    // ---------------------------------------------------------------------------
    template<supported_scalar T>
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

        // Determinant of A, computed from the factorization (no extra solve).
        // Construction factorizes eagerly, so this normally reuses the factors;
        // if a solver ever reaches this without them it factorizes first.
        // Overflow/underflow of the final value returns +/-inf or 0; det_sign()
        // remains valid because the sign is accumulated separately.
        [[nodiscard]] T det() const;
        // Sign of det(): -1, 0 (singular or perturbed pivot), or +1.
        [[nodiscard]] int det_sign() const;

    private:
        struct impl;
        std::unique_ptr<impl> p_;
    };

    // Convenience names for the two supported instantiations.
    using float_solver = solver<float>;
    using double_solver = solver<double>;

    // ---------------------------------------------------------------------------
    // Mixed-precision interface for a double matrix.  The matrix is copied to
    // device float storage and factorized once with solver<float>.  solve()
    // performs two float solves: first for the right-hand side, then for the
    // double-precision residual against the original double matrix.  For well
    // conditioned systems the corrected result has about double precision.
    //
    // A is copied to private float storage during construction, but the
    // original double matrix is read again for each residual, so it must stay
    // valid and unchanged for the lifetime of the object (as with solver<T>).
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

        // Determinant and its sign, taken from the underlying float
        // factorization (returned as double).  For near-singular double
        // matrices the sign/magnitude can differ from the double determinant.
        [[nodiscard]] double det() const;
        [[nodiscard]] int det_sign() const;

    private:
        struct impl;
        std::unique_ptr<impl> p_;
    };

    extern template class solver<float>;
    extern template class solver<double>;
    extern template int spike_gbsv<float>(int, int, int, int, const float*, int, float*, int, const opts&, stats*);
    extern template int spike_gbsv<double>(int, int, int, int, const double*, int, double*, int, const opts&, stats*);

} // namespace gspike
