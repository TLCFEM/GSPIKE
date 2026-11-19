// Benchmark for gspike.  Factor time is measured once with CUDA events, solve
// time is the average over several solves with the same factorization.
//
// Alongside the timings the benchmark reports algorithmic GFLOPS.  The
// operation count is derived on the host from the public API (n, kl, ku, the
// partition count and the refinement steps actually performed) and the
// formulas below; nothing is instrumented inside the library.
//
//   spike_bench                          # default sweep (double), incl. nrhs = 1..20
//   spike_bench --type float             # default sweep (float), incl. nrhs = 1..20
//   spike_bench --mixed                  # default sweep (float factors, double solution)
//   spike_bench --nrhs-sweep             # only the nrhs = 1..20 solve-scaling table
//   spike_bench [options] n klu nrhs [parts]    # one problem, parts=0 => auto
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <gspike/spike.h>
#include <random>
#include <vector>

namespace {

    // -------------------------------------------------------------------------
    // Operation counts.
    //
    // The formulas mirror src/kernels.cuh: one division is one flop, one
    // multiply-add is two, comparisons/pivot search and data movement are not
    // counted.  n is the size of the (sub)system, kl/ku its bandwidths.
    // -------------------------------------------------------------------------

    // dk::band_lu: per column j, (iend-j) divisions and
    // (kend-j)*(iend-j) multiply-adds, with iend = min(j+kl, n-1) and
    // kend = min(n-1, j+kl+ku).
    double lu_flops(const long long n, const long long kl, const long long ku) {
        const long long kmax = std::min(kl + ku, n - 1);
        double f = 0.0;
        for(long long j = 0; j < n; ++j) {
            const long long iend = std::min(j + kl, n - 1);
            const long long kend = std::min(n - 1, j + kmax);
            f += static_cast<double>(iend - j) + 2.0 * static_cast<double>(kend - j) * static_cast<double>(iend - j);
        }
        return f;
    }

    // dk::band_solve for one right-hand side: one division per row, kl
    // multiply-adds forward and kl+ku backward.
    double solve_flops_block(const long long n, const long long kl, const long long ku) {
        auto f = static_cast<double>(n);
        for(long long j = 0; j < n; ++j) f += 2.0 * static_cast<double>(std::min(kl, n - 1 - j));
        for(long long i = 0; i < n; ++i) f += 2.0 * static_cast<double>(std::min(kl + ku, n - 1 - i));
        return f;
    }

    // Union of the partition sizes used by solver<>::factor().
    double solve_flops_all_blocks(const long long n, const long long kl, const long long ku, const int p) {
        double f = 0.0;
        for(int i = 0; i < p; ++i) {
            const long long ni = n / p + (i < n % p ? 1 : 0);
            f += solve_flops_block(ni, kl, ku);
        }
        return f;
    }

    // dk::band_matvec: one multiply-add per band entry, per right-hand side.
    double matvec_flops(const long long n, const long long kl, const long long ku, const long long nrhs) {
        double f = 0.0;
        for(long long i = 0; i < n; ++i) {
            const long long j0 = std::max(0LL, i - kl);
            const long long j1 = std::min(n - 1, i + ku);
            f += 2.0 * static_cast<double>(j1 - j0 + 1);
        }
        return f * static_cast<double>(nrhs);
    }

    // Factorization: band LU of every partition, 2*klu spike solves, and the
    // reduced interface LU.
    double factor_flops(const long long n, const long long kl, const long long ku, const int p) {
        const long long klu = std::max(kl, ku);
        double f = 0.0;
        for(int i = 0; i < p; ++i) {
            const long long ni = n / p + (i < n % p ? 1 : 0);
            f += lu_flops(ni, kl, ku);
        }
        f += 2.0 * static_cast<double>(klu) * solve_flops_all_blocks(n, kl, ku, p);
        if(p > 1) {
            const long long m = 2 * klu * (p - 1), klr = 3 * klu;
            f += lu_flops(m, klr, klr);
        }
        return f;
    }

    // One solve with nrhs right-hand sides, including the refinement steps
    // actually performed: each refinement adds a solve, an axpby update and a
    // new residual (matvec, bsub, two 2-norms).
    double solve_flops(const long long n, const long long kl, const long long ku, const int p, const long long nrhs, const int refine) {
        const long long klu = std::max(kl, ku);
        const long long m = 2 * klu * (p - 1), klr = 3 * klu;
        const double one = (p == 1) ? static_cast<double>(nrhs) * solve_flops_block(n, kl, ku) : static_cast<double>(nrhs) * (2.0 * solve_flops_all_blocks(n, kl, ku, p) + solve_flops_block(m, klr, klr) + 4.0 * static_cast<double>(p - 1) * static_cast<double>(klu) * static_cast<double>(klu));
        const double residual = matvec_flops(n, kl, ku, nrhs) + static_cast<double>(n) * static_cast<double>(nrhs) + 4.0 * static_cast<double>(n) * static_cast<double>(nrhs);
        const double axpby = 3.0 * static_cast<double>(n) * static_cast<double>(nrhs);
        return (1 + refine) * one + refine * axpby + (1 + refine) * residual;
    }

    // -------------------------------------------------------------------------

    // Diagonally dominant random band matrix; |a_ij| decays with the distance
    // from the diagonal so that the band is not dominated by a few entries.
    template<class T>
    void make_band(const int n, const int klu, std::vector<T>& a, std::vector<T>& b, const int nrhs, const int lda) {
        a.assign(static_cast<size_t>(lda) * n, T(0));
        b.resize(static_cast<size_t>(n) * nrhs);
        std::mt19937 gen(42);
        std::uniform_real_distribution u(-1.0, 1.0);
        for(int j = 0; j < n; ++j) {
            double rs = 0;
            for(int i = std::max(0, j - klu); i <= std::min(n - 1, j + klu); ++i) {
                if(i == j) continue;
                const double v = u(gen) / (1.0 + std::abs(i - j));
                a[static_cast<size_t>(j) * lda + klu + i - j] = static_cast<T>(v);
                rs += std::abs(v);
            }
            a[static_cast<size_t>(j) * lda + klu] = static_cast<T>(rs + 1.0);
        }
        for(auto& v : b) v = static_cast<T>(u(gen));
    }

    void print_header() {
        std::printf("%-6s %8s %4s %4s %5s %3s %9s %8s %9s %9s %8s %9s %10s\n", "type", "n", "klu", "nrhs", "parts", "ref", "fact_ms", "fact_G", "fact_GF/s", "solve_ms", "solve_G", "solve_GF/s", "relres");
    }

    // Header for the 1..20 right-hand-side sweep, where a factorization is
    // reused and only the per-solve cost changes with nrhs.
    void print_solve_header() {
        std::printf("%-6s %8s %4s %4s %5s %3s %9s %8s %9s %10s\n", "type", "n", "klu", "nrhs", "parts", "ref", "solve_ms", "us/rhs", "solve_GF/s", "relres");
    }

    template<class T>
    void run(const char* tag, const int n, const int klu, const int nrhs, const int parts, const int reps) {
        const int kl = klu, ku = klu, lda = kl + ku + 1;
        std::vector<T> a, b;
        make_band<T>(n, klu, a, b, nrhs, lda);

        T *da = nullptr, *db = nullptr;
        cudaMalloc(&da, a.size() * sizeof(T));
        cudaMalloc(&db, b.size() * sizeof(T));
        cudaMemcpy(da, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice);

        gspike::opts o;
        o.parts = parts;
        gspike::stats st;

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);

        cudaEventRecord(e0, nullptr);
        gspike::solver<T> s(n, kl, ku, da, lda, o);
        cudaEventRecord(e1, nullptr);
        cudaEventSynchronize(e1);
        float factor_ms = 0.0f;
        cudaEventElapsedTime(&factor_ms, e0, e1);

        cudaMemcpy(db, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice);
        s.solve(nrhs, db, n, &st); // warm up

        float solve_ms = 0.0f;
        int info = 0;
        for(int r = 0; r < reps; ++r) { // each solve timed on its own; the B reset is outside
            cudaMemcpy(db, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice);
            cudaEventRecord(e0, nullptr);
            info = s.solve(nrhs, db, n, &st);
            cudaEventRecord(e1, nullptr);
            cudaEventSynchronize(e1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, e0, e1);
            solve_ms += ms;
        }
        solve_ms /= static_cast<float>(reps);

        const double ff = factor_flops(n, kl, ku, s.parts());
        const double sf = solve_flops(n, kl, ku, s.parts(), nrhs, st.refine);

        std::printf("%-6s %8d %4d %4d %5d %3d %9.2f %8.3f %9.2f %9.2f %8.3f %9.2f %10.2e\n", tag, n, klu, nrhs, s.parts(), st.refine, factor_ms, ff / 1e9, ff / (static_cast<double>(factor_ms) * 1e6), solve_ms, sf / 1e9, sf / (static_cast<double>(solve_ms) * 1e6), st.residual);
        if(info != 0) std::printf("       warning: solve returned info=%d\n", info);

        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaFree(da);
        cudaFree(db);
    }

    template<class T>
    void sweep(const char* tag, const int reps) {
        print_header();
        run<T>(tag, 20000, 10, 1, 0, reps);
        run<T>(tag, 100000, 20, 8, 0, reps);
        run<T>(tag, 200000, 50, 4, 0, reps);
        run<T>(tag, 200000, 50, 4, 1, reps);   // single partition = plain band LU
        run<T>(tag, 500000, 65, 4, 0, reps);   // factor throughput plateau
        run<T>(tag, 80000, 15, 1024, 0, reps); // solve throughput plateau (many RHS)
    }

    // Mixed precision: the same problems with a float factorization and
    // opts::mixed_iters float solves per solve.  Flops are counted with the
    // same model, with the float iterations playing the role of refinement.
    void run_mixed(const char* tag, const int n, const int klu, const int nrhs, const int parts, const int reps, const int mix_iters) {
        const int kl = klu, ku = klu, lda = kl + ku + 1;
        std::vector<double> a, b;
        make_band<double>(n, klu, a, b, nrhs, lda);

        double* da = nullptr;
        double* db = nullptr;
        cudaMalloc(&da, a.size() * sizeof(double));
        cudaMalloc(&db, b.size() * sizeof(double));
        cudaMemcpy(da, a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice);

        gspike::opts o;
        o.parts = parts;
        o.mixed_iters = mix_iters;
        gspike::stats st;

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);

        cudaEventRecord(e0, nullptr);
        const gspike::mixed_solver s(n, kl, ku, da, lda, o);
        cudaEventRecord(e1, nullptr);
        cudaEventSynchronize(e1);
        float factor_ms = 0.0f;
        cudaEventElapsedTime(&factor_ms, e0, e1);

        cudaMemcpy(db, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice);
        s.solve(nrhs, db, n, &st); // warm up

        float solve_ms = 0.0f;
        int info = 0;
        for(int r = 0; r < reps; ++r) {
            cudaMemcpy(db, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice);
            cudaEventRecord(e0, nullptr);
            info = s.solve(nrhs, db, n, &st);
            cudaEventRecord(e1, nullptr);
            cudaEventSynchronize(e1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, e0, e1);
            solve_ms += ms;
        }
        solve_ms /= static_cast<float>(reps);

        const double ff = factor_flops(n, kl, ku, s.parts());
        // After the first float solve, each kept iteration is a correction and
        // costs like one refinement step of the double solver.
        const double sf = solve_flops(n, kl, ku, s.parts(), nrhs, std::max(0, st.refine - 1));

        std::printf("%-6s %8d %4d %4d %5d %3d %9.2f %8.3f %9.2f %9.2f %8.3f %9.2f %10.2e\n", tag, n, klu, nrhs, s.parts(), st.refine, factor_ms, ff / 1e9, ff / (static_cast<double>(factor_ms) * 1e6), solve_ms, sf / 1e9, sf / (static_cast<double>(solve_ms) * 1e6), st.residual);
        if(info != 0) std::printf("       warning: mixed solve returned info=%d\n", info);

        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaFree(da);
        cudaFree(db);
    }

    void mixed_sweep(const char* tag, const int reps) {
        print_header();
        run_mixed(tag, 20000, 10, 1, 0, reps, 2);
        run_mixed(tag, 100000, 20, 8, 0, reps, 2);
        run_mixed(tag, 200000, 50, 4, 0, reps, 2);
        run_mixed(tag, 200000, 50, 4, 1, reps, 2);
        run_mixed(tag, 500000, 65, 4, 0, reps, 2);
        run_mixed(tag, 80000, 15, 1024, 0, reps, 2);
    }

    // -------------------------------------------------------------------------
    // Solve scaling for 1..20 right-hand sides.
    //
    // One factorization of a fixed problem is made once and every row reuses
    // it, so the table isolates the cost of a solve with few right-hand sides:
    // it is dominated by kernel latency and by the serial row recurrences until
    // enough columns are available to fill the GPU, after which it grows
    // linearly in nrhs.  us/rhs is the marginal cost of one right-hand side.
    // -------------------------------------------------------------------------
    constexpr int solve_sweep_n = 100000;
    constexpr int solve_sweep_klu = 20;
    constexpr int solve_sweep_nrhs = 20;

    template<class T>
    void solve_sweep(const char* tag, const int reps) {
        constexpr int n = solve_sweep_n, klu = solve_sweep_klu, kl = klu, ku = klu, lda = kl + ku + 1;
        std::vector<T> a, b;
        make_band<T>(n, klu, a, b, solve_sweep_nrhs, lda);

        T *da = nullptr, *db = nullptr;
        cudaMalloc(&da, a.size() * sizeof(T));
        cudaMalloc(&db, b.size() * sizeof(T));
        cudaMemcpy(da, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice);

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);

        cudaEventRecord(e0, nullptr);
        gspike::solver<T> s(n, kl, ku, da, lda);
        cudaEventRecord(e1, nullptr);
        cudaEventSynchronize(e1);
        float factor_ms = 0.0f;
        cudaEventElapsedTime(&factor_ms, e0, e1);
        const double ff = factor_flops(n, kl, ku, s.parts());

        std::printf("# solve scaling, one factorization reused: %s n=%d klu=%d parts=%d factor=%.2f ms (%.2f GF/s)\n", tag, n, klu, s.parts(), factor_ms, ff / (static_cast<double>(factor_ms) * 1e6));
        print_solve_header();

        for(int nrhs = 1; nrhs <= solve_sweep_nrhs; ++nrhs) {
            const size_t nb = static_cast<size_t>(n) * nrhs;
            cudaMemcpy(db, b.data(), nb * sizeof(T), cudaMemcpyHostToDevice);
            gspike::stats st;
            s.solve(nrhs, db, n, &st); // warm up

            float solve_ms = 0.0f;
            int info = 0;
            for(int r = 0; r < reps; ++r) {
                cudaMemcpy(db, b.data(), nb * sizeof(T), cudaMemcpyHostToDevice);
                cudaEventRecord(e0, nullptr);
                info = s.solve(nrhs, db, n, &st);
                cudaEventRecord(e1, nullptr);
                cudaEventSynchronize(e1);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, e0, e1);
                solve_ms += ms;
            }
            solve_ms /= static_cast<float>(reps);

            const double sf = solve_flops(n, kl, ku, s.parts(), nrhs, st.refine);
            std::printf("%-6s %8d %4d %4d %5d %3d %9.2f %8.0f %9.2f %10.2e\n", tag, n, klu, nrhs, s.parts(), st.refine, solve_ms, 1000.0 * solve_ms / nrhs, sf / (static_cast<double>(solve_ms) * 1e6), st.residual);
            if(info != 0) std::printf("       warning: solve returned info=%d\n", info);
        }

        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaFree(da);
        cudaFree(db);
    }

    void solve_sweep_mixed(const char* tag, const int reps) {
        constexpr int n = solve_sweep_n, klu = solve_sweep_klu, kl = klu, ku = klu, lda = kl + ku + 1;
        std::vector<double> a, b;
        make_band<double>(n, klu, a, b, solve_sweep_nrhs, lda);

        double* da = nullptr;
        double* db = nullptr;
        cudaMalloc(&da, a.size() * sizeof(double));
        cudaMalloc(&db, b.size() * sizeof(double));
        cudaMemcpy(da, a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice);

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);

        cudaEventRecord(e0, nullptr);
        const gspike::mixed_solver s(n, kl, ku, da, lda);
        cudaEventRecord(e1, nullptr);
        cudaEventSynchronize(e1);
        float factor_ms = 0.0f;
        cudaEventElapsedTime(&factor_ms, e0, e1);
        const double ff = factor_flops(n, kl, ku, s.parts());

        std::printf("# solve scaling, one factorization reused: %s n=%d klu=%d parts=%d factor=%.2f ms (%.2f GF/s)\n", tag, n, klu, s.parts(), factor_ms, ff / (static_cast<double>(factor_ms) * 1e6));
        print_solve_header();

        for(int nrhs = 1; nrhs <= solve_sweep_nrhs; ++nrhs) {
            const size_t nb = static_cast<size_t>(n) * nrhs;
            cudaMemcpy(db, b.data(), nb * sizeof(double), cudaMemcpyHostToDevice);
            gspike::stats st;
            s.solve(nrhs, db, n, &st); // warm up

            float solve_ms = 0.0f;
            int info = 0;
            for(int r = 0; r < reps; ++r) {
                cudaMemcpy(db, b.data(), nb * sizeof(double), cudaMemcpyHostToDevice);
                cudaEventRecord(e0, nullptr);
                info = s.solve(nrhs, db, n, &st);
                cudaEventRecord(e1, nullptr);
                cudaEventSynchronize(e1);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, e0, e1);
                solve_ms += ms;
            }
            solve_ms /= static_cast<float>(reps);

            // Each kept float solve after the first plays the role of a
            // refinement step in the flop model.
            const double sf = solve_flops(n, kl, ku, s.parts(), nrhs, std::max(0, st.refine - 1));
            std::printf("%-6s %8d %4d %4d %5d %3d %9.2f %8.0f %9.2f %10.2e\n", tag, n, klu, nrhs, s.parts(), st.refine, solve_ms, 1000.0 * solve_ms / nrhs, sf / (static_cast<double>(solve_ms) * 1e6), st.residual);
            if(info != 0) std::printf("       warning: mixed solve returned info=%d\n", info);
        }

        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaFree(da);
        cudaFree(db);
    }

    void run_solve_sweep(const char* tag, const bool use_float, const bool use_mixed, const int reps) {
        if(use_mixed)
            solve_sweep_mixed(tag, reps);
        else if(use_float)
            solve_sweep<float>(tag, reps);
        else
            solve_sweep<double>(tag, reps);
    }

} // namespace

int main(const int argc, char** argv) {
    bool use_float = false, use_mixed = false, only_solve_sweep = false;
    int reps = 5;
    int pos = 1;
    std::printf("# gspike: %d-bit indexing\n", static_cast<int>(8 * sizeof(gspike::index_t)));
    for(; pos < argc; ++pos) {
        if(std::strcmp(argv[pos], "--type") == 0 && pos + 1 < argc) {
            use_float = std::strcmp(argv[++pos], "float") == 0;
        }
        else if(std::strcmp(argv[pos], "--mixed") == 0) {
            use_mixed = true;
        }
        else if(std::strcmp(argv[pos], "--nrhs-sweep") == 0) {
            only_solve_sweep = true;
        }
        else if(std::strcmp(argv[pos], "--reps") == 0 && pos + 1 < argc) {
            reps = std::atoi(argv[++pos]);
            if(reps < 1) {
                std::printf("--reps must be positive\n");
                return 2;
            }
        }
        else if(argv[pos][0] == '-') {
            std::printf("usage: %s [--type float|double] [--mixed] [--nrhs-sweep] [--reps N] [n klu nrhs [parts]]\n", argv[0]);
            return 2;
        }
        else {
            break;
        }
    }
    if(use_mixed && use_float) {
        std::printf("--mixed needs the double problem (drop --type float)\n");
        return 2;
    }
    const char* tag = use_mixed ? "mixed" : (use_float ? "float" : "double");

    if(pos + 2 < argc) {
        const int n = std::atoi(argv[pos]);
        const int klu = std::atoi(argv[pos + 1]);
        const int nrhs = std::atoi(argv[pos + 2]);
        const int parts = (pos + 3 < argc) ? std::atoi(argv[pos + 3]) : 0;
        print_header();
        if(use_mixed)
            run_mixed(tag, n, klu, nrhs, parts, reps, 2);
        else if(use_float)
            run<float>(tag, n, klu, nrhs, parts, reps);
        else
            run<double>(tag, n, klu, nrhs, parts, reps);
        return 0;
    }

    if(only_solve_sweep) {
        run_solve_sweep(tag, use_float, use_mixed, reps);
        return 0;
    }

    if(use_mixed)
        mixed_sweep(tag, reps);
    else if(use_float)
        sweep<float>(tag, reps);
    else
        sweep<double>(tag, reps);

    std::printf("\n");
    run_solve_sweep(tag, use_float, use_mixed, reps);
    return 0;
}
