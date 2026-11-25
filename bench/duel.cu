// Direct-solver head-to-head on the same banded matrix:
//   * gspike mixed precision (float factor + double residual correction)
//   * gspike double precision (double factor + solve)
//   * Intel MKL general band LU (dgbtrf + dgbtrs, double)
//   * original Fortran SPIKE-SMP CPU solver (double)
// Optional target, built when MKL, Fortran, LAPACK and OpenMP are available.
// Factorization is timed once, solve is averaged over five solves with the same
// factorization, and the residual is checked in double precision for every
// method.  Only direct solvers are compared: no iterative methods.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <gspike/spike.h>
#include <memory>
#include <mkl/mkl.h>
#include <random>
#include <vector>

#include "spike_ref.h"

namespace {
    constexpr int kReps = 5;

    void ck(const cudaError_t e, const char* what) {
        if(e != cudaSuccess) {
            std::fprintf(stderr, "spike_duel: %s: %s\n", what, cudaGetErrorString(e));
            std::exit(1);
        }
    }

    void check_info(const int info, const char* what) {
        if(info != 0) {
            std::fprintf(stderr, "spike_duel: %s failed (info=%d)\n", what, info);
            std::exit(1);
        }
    }

    // Identical generator to spike_bench so both benchmarks see the same bands.
    template<class T>
    void make_band(const int n, const int klu, const bool dominant, std::vector<T>& a, std::vector<double>& b, const int nrhs, const int lda) {
        a.assign((size_t)lda * n, T(0));
        b.assign((size_t)n * nrhs, 0.0);
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        for(int j = 0; j < n; ++j) {
            double rs = 0;
            for(int i = std::max(0, j - klu); i <= std::min(n - 1, j + klu); ++i) {
                if(i == j) continue;
                const double v = u(gen) / (1.0 + std::abs(i - j));
                a[(size_t)j * lda + klu + i - j] = (T)v;
                rs += std::abs(v);
            }
            a[(size_t)j * lda + klu] = (T)(dominant ? rs + 1.0 : u(gen) * 0.5);
        }
        for(auto& v : b) v = u(gen);
    }

    // Relative residual ||b - A x|| / ||b|| in double precision.
    template<class T>
    double residual(const int n, const int klu, const int lda, const std::vector<T>& a, const T* x, const int ldx, const std::vector<double>& b, const int nrhs) {
        double rn = 0, bn = 0;
        for(int c = 0; c < nrhs; ++c)
            for(int i = 0; i < n; ++i) {
                double s = 0;
                for(int j = std::max(0, i - klu); j <= std::min(n - 1, i + klu); ++j)
                    s += (double)a[(size_t)j * lda + klu + i - j] * (double)x[(size_t)c * ldx + j];
                const double r = b[(size_t)c * n + i] - s;
                rn += r * r;
                bn += b[(size_t)c * n + i] * b[(size_t)c * n + i];
            }
        return std::sqrt(rn / std::max(bn, 1e-300));
    }

    struct result {
        int parts = 0;
        double factor_ms = 0, solve_ms = 0, relres = 0;

        [[nodiscard]] double total_ms() const { return factor_ms + solve_ms; }
    };

    // Shared GPU timing path: make(device A, lda) constructs the factorized
    // solver, so solver<double> and mixed_solver share this code.
    template<class Factory>
    result run_gpu(const int n, const int klu, const int nrhs, const std::vector<double>& a, const std::vector<double>& b, const int lda, Factory make) {
        result t;
        double* da = nullptr;
        ck(cudaMalloc(&da, a.size() * sizeof(double)), "malloc A");
        ck(cudaMemcpy(da, a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice), "copy A");

        // Warm the allocator so timed construction does not pay for it.
        { auto warm = make(da, lda); }

        cudaEvent_t e0, e1;
        ck(cudaEventCreate(&e0), "event");
        ck(cudaEventCreate(&e1), "event");
        ck(cudaEventRecord(e0), "record");
        auto s = make(da, lda);
        ck(cudaEventRecord(e1), "record");
        ck(cudaEventSynchronize(e1), "sync");
        float ms = 0;
        ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
        t.factor_ms = ms;
        t.parts = s->parts();

        double* db = nullptr;
        ck(cudaMalloc(&db, b.size() * sizeof(double)), "malloc B");
        ck(cudaMemcpy(db, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice), "copy B");
        s->solve(nrhs, db, n); // warm up
        double total = 0;
        for(int r = 0; r < kReps; ++r) {
            ck(cudaMemcpy(db, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice), "reset B");
            ck(cudaEventRecord(e0), "record");
            s->solve(nrhs, db, n);
            ck(cudaEventRecord(e1), "record");
            ck(cudaEventSynchronize(e1), "sync");
            ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
            total += ms;
        }
        t.solve_ms = total / kReps;

        ck(cudaMemcpy(db, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice), "reset B");
        s->solve(nrhs, db, n);
        std::vector<double> x(b.size());
        ck(cudaMemcpy(x.data(), db, x.size() * sizeof(double), cudaMemcpyDeviceToHost), "copy X");
        t.relres = residual(n, klu, lda, a, x.data(), n, b, nrhs);

        cudaFree(da);
        cudaFree(db);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        return t;
    }

    // MKL double-precision general band LU: dgbtrf + dgbtrs.
    result run_mkl(const int n, const int klu, const int nrhs, const std::vector<double>& a, const std::vector<double>& b, const int lda) {
        result t;
        const int ldab = 3 * klu + 1, off = 2 * klu;
        std::vector<double> ab((size_t)ldab * n, 0.0), rhs = b;
        for(int j = 0; j < n; ++j)
            for(int i = std::max(0, j - klu); i <= std::min(n - 1, j + klu); ++i)
                ab[(size_t)j * ldab + off + i - j] = a[(size_t)j * lda + klu + i - j];
        std::vector<int> ipiv(n);
        int info = 0;
        const char trans = 'N';

        // Warm up the MKL thread pool and first-touch the work arrays.
        {
            auto abw = ab, rhsw = rhs;
            auto ipivw = ipiv;
            dgbtrf_(&n, &n, &klu, &klu, abw.data(), &ldab, ipivw.data(), &info);
            check_info(info, "MKL dgbtrf (warmup)");
            dgbtrs_(&trans, &n, &klu, &klu, &nrhs, abw.data(), &ldab, ipivw.data(), rhsw.data(), &n, &info);
            check_info(info, "MKL dgbtrs (warmup)");
        }

        const auto start = std::chrono::steady_clock::now();
        dgbtrf_(&n, &n, &klu, &klu, ab.data(), &ldab, ipiv.data(), &info);
        const auto fac = std::chrono::steady_clock::now();
        check_info(info, "MKL dgbtrf");
        t.factor_ms = std::chrono::duration<double, std::milli>(fac - start).count();

        {
            auto rhsw = rhs;
            dgbtrs_(&trans, &n, &klu, &klu, &nrhs, ab.data(), &ldab, ipiv.data(), rhsw.data(), &n, &info);
            check_info(info, "MKL dgbtrs (warmup solve)");
        }
        std::vector<double> x = rhs;
        double total = 0;
        for(int r = 0; r < kReps; ++r) {
            x = rhs;
            const auto s0 = std::chrono::steady_clock::now();
            dgbtrs_(&trans, &n, &klu, &klu, &nrhs, ab.data(), &ldab, ipiv.data(), x.data(), &n, &info);
            const auto s1 = std::chrono::steady_clock::now();
            total += std::chrono::duration<double, std::milli>(s1 - s0).count();
        }
        check_info(info, "MKL dgbtrs");
        t.solve_ms = total / kReps;
        t.relres = residual(n, klu, lda, a, x.data(), n, b, nrhs);
        return t;
    }

    // Original Fortran SPIKE-SMP solver; the wrapper reports factor time.
    result run_fortran(const int n, const int klu, const int nrhs, const std::vector<double>& a, const std::vector<double>& b, const int lda, const int nthreads) {
        result t;
        int parts = 0, info = 0;
        double factor_ms = 0;
        const int id = spike_ref_create(n, klu, klu, a.data(), lda, nthreads, &parts, &factor_ms, &info);
        if(id < 0 || info != 0) {
            std::fprintf(stderr, "spike_duel: Fortran create failed (id=%d info=%d)\n", id, info);
            std::exit(1);
        }
        std::vector<double> x = b;
        double solve_ms = 0;
        int sinfo = spike_ref_solve(id, nrhs, x.data(), n, &solve_ms); // warm up
        double total = 0;
        for(int r = 0; r < kReps; ++r) {
            std::copy(b.begin(), b.end(), x.begin());
            sinfo = spike_ref_solve(id, nrhs, x.data(), n, &solve_ms);
            total += solve_ms;
        }
        spike_ref_destroy(id);
        if(sinfo != 0) {
            std::fprintf(stderr, "spike_duel: Fortran solve failed (info=%d)\n", sinfo);
            std::exit(1);
        }
        t.parts = parts;
        t.factor_ms = factor_ms;
        t.solve_ms = total / kReps;
        t.relres = residual(n, klu, lda, a, x.data(), n, b, nrhs);
        return t;
    }

    void run_case(const int n, const int klu, const int nrhs, const bool dominant, const int threads) {
        const int lda = 2 * klu + 1;
        std::vector<double> a, b;
        make_band(n, klu, dominant, a, b, nrhs, lda);

        const auto gm = run_gpu(n, klu, nrhs, a, b, lda, [&](const double* da, int ld) {
            return std::make_unique<gspike::mixed_solver>(n, klu, klu, da, ld, gspike::opts{});
        });
        const auto gd = run_gpu(n, klu, nrhs, a, b, lda, [&](const double* da, int ld) {
            return std::make_unique<gspike::solver<double>>(n, klu, klu, da, ld, gspike::opts{});
        });
        const auto md = run_mkl(n, klu, nrhs, a, b, lda);
        const auto fp = run_fortran(n, klu, nrhs, a, b, lda, threads);

        std::printf("n=%d klu=%d nrhs=%d dominant=%d threads=%d\n", n, klu, nrhs, dominant ? 1 : 0, threads);
        std::printf("%-18s %6s %10s %10s %10s %12s\n", "method", "parts", "factor_ms", "solve_ms", "total_ms", "relres");
        const auto row = [](const char* name, const result& r) {
            std::printf("%-18s %6d %10.2f %10.2f %10.2f %12.2e\n", name, r.parts, r.factor_ms, r.solve_ms, r.total_ms(), r.relres);
        };
        row("gspike mixed", gm);
        row("gspike double", gd);
        row("MKL double band", md);
        row("Fortran SPIKE", fp);
        std::printf("\n");
    }
} // namespace

int main(int argc, char** argv) {
    // This binary also links the Fortran reference (and therefore libgomp).
    // With MKL's default Intel threading layer that combination gives wrong
    // dgbtrf/dgbtrs results with more than one thread; the GNU layer shares
    // libgomp with the Fortran code and is correct.
    mkl_set_threading_layer(MKL_THREADING_GNU);
    const char* env = std::getenv("DUEL_CPU_THREADS");
    const int threads = env ? std::atoi(env) : 8;
    mkl_set_num_threads(threads);

    if(argc > 1) {
        const int n = std::atoi(argv[1]);
        const int klu = argc > 2 ? std::atoi(argv[2]) : n / 100;
        const int nrhs = argc > 3 ? std::atoi(argv[3]) : 1;
        const bool dominant = argc < 5 || std::atoi(argv[4]) != 0;
        if(n < 1 || klu < 0 || nrhs < 1) {
            std::printf("usage: %s [n klu nrhs [dominant]]   (DUEL_CPU_THREADS sets both MKL and OpenMP threads)\n", argv[0]);
            return 2;
        }
        run_case(n, klu, nrhs, dominant, threads);
        return 0;
    }

    // Same sweep as spike_bench: bands around 1% of n, few right-hand sides.
    run_case(10000, 100, 1, true, threads);
    run_case(25000, 250, 1, true, threads);
    run_case(50000, 500, 2, true, threads);
    run_case(80000, 800, 1, true, threads);
    return 0;
}
