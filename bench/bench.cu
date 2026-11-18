// Benchmark for gspike.  Factor time is measured once, solve time is the
// average over several solves with the same factorization.
//
//   spike_bench                          # default sweep (double)
//   spike_bench --type float             # default sweep (float)
//   spike_bench n klu nrhs [parts]       # one problem, parts=0 => automatic
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <gspike/spike.h>
#include <random>
#include <vector>

namespace {

    double millis(const std::chrono::high_resolution_clock::time_point a, const std::chrono::high_resolution_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    }

    // Diagonally dominant random band matrix; |a_ij| decays with the distance
    // from the diagonal so that the band is not dominated by a few entries.
    template<class T>
    void make_band(const int n, const int klu, std::vector<T>& a, std::vector<T>& b, const int nrhs, const int lda) {
        a.assign((size_t)lda * n, T(0));
        b.resize((size_t)n * nrhs);
        std::mt19937 gen(42);
        std::uniform_real_distribution u(-1.0, 1.0);
        for(int j = 0; j < n; ++j) {
            double rs = 0;
            for(int i = std::max(0, j - klu); i <= std::min(n - 1, j + klu); ++i) {
                if(i == j) continue;
                const double v = u(gen) / (1.0 + std::abs(i - j));
                a[(size_t)j * lda + klu + i - j] = (T)v;
                rs += std::abs(v);
            }
            a[(size_t)j * lda + klu] = (T)(rs + 1.0);
        }
        for(auto& v : b) v = (T)u(gen);
    }

    template<class T>
    void run(const char* tag, int n, const int klu, int nrhs, const int parts) {
        const int kl = klu, ku = klu, lda = kl + ku + 1, reps = 5;
        std::vector<T> a, b;
        make_band<T>(n, klu, a, b, nrhs, lda);

        T *da = nullptr, *db = nullptr;
        cudaMalloc(&da, a.size() * sizeof(T));
        cudaMalloc(&db, b.size() * sizeof(T));
        cudaMemcpy(da, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice);

        gspike::opts o;
        o.parts = parts;
        gspike::stats st;

        const auto t0 = std::chrono::high_resolution_clock::now();
        gspike::solver<T> s(n, kl, ku, da, lda, o);
        cudaDeviceSynchronize();
        const auto t1 = std::chrono::high_resolution_clock::now();

        cudaMemcpy(db, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice);
        s.solve(nrhs, db, n, &st); // warm up

        const auto t2 = std::chrono::high_resolution_clock::now();
        for(int r = 0; r < reps; ++r) {
            cudaMemcpy(db, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice);
            s.solve(nrhs, db, n, &st);
        }
        cudaDeviceSynchronize();
        const auto t3 = std::chrono::high_resolution_clock::now();

        std::printf("%-6s %8d %5d %5d %6d %9.2f %9.2f %10.2e\n", tag, n, klu, nrhs, s.parts(), millis(t0, t1), millis(t2, t3) / reps, st.residual);

        cudaFree(da);
        cudaFree(db);
    }

    template<class T>
    void sweep(const char* tag) {
        std::printf("%-6s %8s %5s %5s %6s %9s %9s %10s\n", "type", "n", "klu", "nrhs", "parts", "factor_ms", "solve_ms", "relres");
        run<T>(tag, 20000, 10, 1, 0);
        run<T>(tag, 100000, 20, 1, 0);
        run<T>(tag, 100000, 20, 8, 0);
        run<T>(tag, 200000, 50, 4, 0);
        run<T>(tag, 200000, 50, 4, 1); // single partition = plain band LU
    }

} // namespace

int main(const int argc, char** argv) {
    bool use_float = false;
    int pos = 1;
    for(; pos < argc; ++pos) {
        if(std::strcmp(argv[pos], "--type") == 0 && pos + 1 < argc) {
            use_float = std::strcmp(argv[++pos], "float") == 0;
        }
        else if(argv[pos][0] == '-') {
            std::printf("usage: %s [--type float|double] [n klu nrhs [parts]]\n", argv[0]);
            return 2;
        }
        else {
            break;
        }
    }
    const char* tag = use_float ? "float" : "double";

    if(pos + 2 < argc) {
        const int n = atoi(argv[pos]);
        const int klu = atoi(argv[pos + 1]);
        const int nrhs = atoi(argv[pos + 2]);
        const int parts = (pos + 3 < argc) ? atoi(argv[pos + 3]) : 0;
        std::printf("%-6s %8s %5s %5s %6s %9s %9s %10s\n", "type", "n", "klu", "nrhs", "parts", "factor_ms", "solve_ms", "relres");
        if(use_float)
            run<float>(tag, n, klu, nrhs, parts);
        else
            run<double>(tag, n, klu, nrhs, parts);
        return 0;
    }

    if(use_float)
        sweep<float>(tag);
    else
        sweep<double>(tag);
    return 0;
}
