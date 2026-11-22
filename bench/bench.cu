// Benchmark for gspike.  Factor time is measured once, solve time is the
// average over several solves with the same factorization.
//
//   spike_bench                          # default sweep (double)
//   spike_bench --type float             # default sweep (float)
//   spike_bench n klu nrhs [parts]       # one problem, parts=0 => automatic
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <gspike/spike.h>
#include <random>
#include <vector>

namespace {

    double millis(std::chrono::high_resolution_clock::time_point a, std::chrono::high_resolution_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    }

    // Diagonally dominant random band matrix; |a_ij| decays with the distance
    // from the diagonal so that the band is not dominated by a few entries.
    template<class T>
    void make_band(int n, int klu, std::vector<T>& a, std::vector<T>& b, int nrhs, int lda) {
        a.assign((size_t)lda * n, T(0));
        b.resize((size_t)n * nrhs);
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
            a[(size_t)j * lda + klu] = (T)(rs + 1.0);
        }
        for(auto& v : b) v = (T)u(gen);
    }

    // Relative residual ||b - A x|| / ||b|| in double precision.
    template<class T>
    double relres(int n, int kl, int ku, int lda, int nrhs, const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& x) {
        double rn = 0, bn = 0;
        for(int c = 0; c < nrhs; ++c) {
            for(int i = 0; i < n; ++i) {
                double s = 0;
                for(int j = std::max(0, i - kl); j <= std::min(n - 1, i + ku); ++j)
                    s += (double)a[(size_t)j * lda + ku + i - j] * (double)x[(size_t)c * n + j];
                const double r = (double)b[(size_t)c * n + i] - s;
                rn += r * r;
                bn += (double)b[(size_t)c * n + i] * (double)b[(size_t)c * n + i];
            }
        }
        return std::sqrt(rn / std::max(bn, 1e-300));
    }

    template<class T>
    void run(const char* tag, int n, int klu, int nrhs, int parts) {
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

        auto t0 = std::chrono::high_resolution_clock::now();
        gspike::solver<T> s(n, kl, ku, da, lda, o);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();

        cudaMemcpy(db, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice);
        s.solve(nrhs, db, n, &st); // warm up

        auto t2 = std::chrono::high_resolution_clock::now();
        for(int r = 0; r < reps; ++r) {
            cudaMemcpy(db, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice);
            s.solve(nrhs, db, n, &st);
        }
        cudaDeviceSynchronize();
        auto t3 = std::chrono::high_resolution_clock::now();

        std::vector<T> x(b.size());
        cudaMemcpy(x.data(), db, x.size() * sizeof(T), cudaMemcpyDeviceToHost);
        const double rr = relres<T>(n, kl, ku, lda, nrhs, a, b, x);
        std::printf("%-6s %8d %5d %5d %6d %9.2f %9.2f %10.2e\n", tag, n, klu, nrhs, s.parts(), millis(t0, t1), millis(t2, t3) / reps, rr);

        cudaFree(da);
        cudaFree(db);
    }

    void run_mixed(int n, int klu, int nrhs, int parts) {
        const int kl = klu, ku = klu, lda = kl + ku + 1, reps = 5;
        std::vector<double> a, b;
        make_band<double>(n, klu, a, b, nrhs, lda);

        double *da = nullptr, *db = nullptr;
        cudaMalloc(&da, a.size() * sizeof(double));
        cudaMalloc(&db, b.size() * sizeof(double));
        cudaMemcpy(da, a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice);

        gspike::opts o;
        o.parts = parts;

        auto t0 = std::chrono::high_resolution_clock::now();
        gspike::mixed_solver s(n, kl, ku, da, lda, o);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();

        cudaMemcpy(db, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice);
        s.solve(nrhs, db, n); // warm up

        auto t2 = std::chrono::high_resolution_clock::now();
        for(int r = 0; r < reps; ++r) {
            cudaMemcpy(db, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice);
            s.solve(nrhs, db, n);
        }
        cudaDeviceSynchronize();
        auto t3 = std::chrono::high_resolution_clock::now();

        std::vector<double> x(b.size());
        cudaMemcpy(x.data(), db, x.size() * sizeof(double), cudaMemcpyDeviceToHost);
        const double rr = relres<double>(n, kl, ku, lda, nrhs, a, b, x);
        std::printf("%-6s %8d %5d %5d %6d %9.2f %9.2f %10.2e\n", "mixed", n, klu, nrhs, s.parts(), millis(t0, t1), millis(t2, t3) / reps, rr);

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

int main(int argc, char** argv) {
    enum class mode { fp32,
                      fp64,
                      mixed } m = mode::fp64;
    int pos = 1;
    for(; pos < argc; ++pos) {
        if(std::strcmp(argv[pos], "--type") == 0 && pos + 1 < argc) {
            const char* t = argv[++pos];
            if(std::strcmp(t, "float") == 0) m = mode::fp32;
            else if(std::strcmp(t, "double") == 0) m = mode::fp64;
            else if(std::strcmp(t, "mixed") == 0) m = mode::mixed;
            else {
                std::printf("usage: %s [--type float|double|mixed] [n klu nrhs [parts]]\n", argv[0]);
                return 2;
            }
        }
        else if(argv[pos][0] == '-') {
            std::printf("usage: %s [--type float|double|mixed] [n klu nrhs [parts]]\n", argv[0]);
            return 2;
        }
        else {
            break;
        }
    }

    if(pos + 2 < argc) {
        const int n = atoi(argv[pos]);
        const int klu = atoi(argv[pos + 1]);
        const int nrhs = atoi(argv[pos + 2]);
        const int parts = (pos + 3 < argc) ? atoi(argv[pos + 3]) : 0;
        std::printf("%-6s %8s %5s %5s %6s %9s %9s %10s\n", "type", "n", "klu", "nrhs", "parts", "factor_ms", "solve_ms", "relres");
        if(m == mode::fp32) run<float>("float", n, klu, nrhs, parts);
        else if(m == mode::fp64) run<double>("double", n, klu, nrhs, parts);
        else run_mixed(n, klu, nrhs, parts);
        return 0;
    }

    if(m == mode::fp32) sweep<float>("float");
    else if(m == mode::fp64) sweep<double>("double");
    else {
        std::printf("%-6s %8s %5s %5s %6s %9s %9s %10s\n", "type", "n", "klu", "nrhs", "parts", "factor_ms", "solve_ms", "relres");
        run_mixed(20000, 10, 1, 0);
        run_mixed(100000, 20, 1, 0);
        run_mixed(100000, 20, 8, 0);
        run_mixed(200000, 50, 4, 0);
    }
    return 0;
}
