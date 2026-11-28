// Benchmark for gspike.  Factorization is timed once, solve time is the average
// over several solves with the same factorization.  The GFLOP/s columns are
// derived from analytic operation counts (band LU, triangular solves, spike
// work, and the mixed-precision residual); the solvers are not instrumented.
//
//   spike_bench                          # default sweep (double)
//   spike_bench --type float             # default sweep (float)
//   spike_bench --type float16           # default sweep (CUDA half)
//   spike_bench --type mixed16           # default sweep (half + 4 double refinements)
//   spike_bench --type mixed             # default sweep (mixed)
//   spike_bench --type cpu               # default sweep (original Fortran)
//   spike_bench --compare                # CPU double vs GPU mixed sweep
//   spike_bench n klu nrhs [parts]       # one problem, parts=0 => automatic
//   spike_bench --compare n klu nrhs     # one CPU/GPU comparison
//   spike_bench --cpu-threads N ...      # OpenMP threads for the reference

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <gspike/spike.h>
#include <gspike/spike_half.h>
#include <memory>
#include <random>
#include <vector>

#ifdef GSPIKE_HAVE_REF
#include "spike_ref.h"
#endif

namespace {
    bool g_full_spikes = false;
    bool g_no_full_spikes = false;
    bool g_profile = false;

    enum class mode {
        fp16,
        fp32,
        fp64,
        mixed,
        mixed16,
        cpu,
        compare
    };

    void ck(const cudaError_t e) {
        if(e != cudaSuccess) {
            gspike::detail::print_err("spike_bench: {}\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }

    class timer {
    public:
        timer() {
            ck(cudaEventCreate(&start_));
            ck(cudaEventCreate(&stop_));
        }

        ~timer() {
            cudaEventDestroy(start_);
            cudaEventDestroy(stop_);
        }

        timer(const timer&) = delete;
        timer& operator=(const timer&) = delete;

        void start() const { ck(cudaEventRecord(start_)); }

        [[nodiscard]] double stop() const {
            ck(cudaEventRecord(stop_));
            ck(cudaEventSynchronize(stop_));
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, start_, stop_));
            return ms;
        }

    private:
        cudaEvent_t start_{}, stop_{};
    };

    template<class T>
    class dev_buffer {
    public:
        explicit dev_buffer(const size_t count) { ck(cudaMalloc(&p_, std::max<size_t>(count, 1) * sizeof(T))); }

        ~dev_buffer() {
            if(p_) cudaFree(p_);
        }

        dev_buffer(const dev_buffer&) = delete;
        dev_buffer& operator=(const dev_buffer&) = delete;

        [[nodiscard]] T* get() const { return p_; }

    private:
        T* p_ = nullptr;
    };

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

    // Relative residual ||b - A x|| / ||b|| in double precision.
    template<class T>
    double relres(const int n, const int kl, const int ku, const int lda, const int nrhs, const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& x) {
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

    std::vector<int> partition_sizes(const int n, const int p) {
        std::vector<int> size(p);
        const int base = n / p, rem = n % p;
        for(int i = 0; i < p; ++i) size[i] = base + (i < rem ? 1 : 0);
        return size;
    }

    // Arithmetic of the band LU kernel: forward elimination over the kl+ku
    // superdiagonals that pivoting can fill, one division and one FMA per entry.
    double band_lu_flops(const int s, const int kl, const int ku) {
        double f = 0;
        for(int j = 0; j < s; ++j) {
            const int rem = s - 1 - j;
            const int rows = std::min(rem, kl);
            const int cols = std::min(rem, kl + ku);
            f += rows + 2.0 * rows * cols;
        }
        return f;
    }

    // Arithmetic of one band triangular solve: forward and back substitution
    // plus the s divisions of the back solve.
    double band_solve_flops(const int s, const int kl, const int ku) {
        double f = s;
        for(int i = 0; i < s; ++i) {
            const int rem = s - 1 - i;
            f += 2.0 * std::min(rem, kl) + 2.0 * std::min(rem, kl + ku);
        }
        return f;
    }

    double band_nnz(const int n, const int kl, const int ku) {
        double nnz = 0;
        for(int i = 0; i < n; ++i) nnz += std::min(n - 1, i + ku) - std::max(0, i - kl) + 1;
        return nnz;
    }

    // SPIKE factorization: block LU, 2*klu spike solves per partition and the
    // reduced interface LU.
    double factor_flops(const int n, const int kl, const int ku, const int p) {
        const int klu = std::max(kl, ku);
        const auto size = partition_sizes(n, p);
        double f = 0;
        for(const int s : size) f += band_lu_flops(s, kl, ku);
        if(p > 1) {
            for(const int s : size) f += 2.0 * klu * band_solve_flops(s, kl, ku);
            f += band_lu_flops(2 * klu * (p - 1), 3 * klu, 3 * klu);
        }
        return f;
    }

    // SPIKE solve: two block solves, the reduced solve and the corner updates.
    double solve_flops(const int n, const int kl, const int ku, const int nrhs, const int p) {
        const int klu = std::max(kl, ku);
        if(p <= 1) return nrhs * band_solve_flops(n, kl, ku);

        const auto size = partition_sizes(n, p);
        double f = 0;
        for(const int s : size) f += 2.0 * nrhs * band_solve_flops(s, kl, ku);
        f += nrhs * band_solve_flops(2 * klu * (p - 1), 3 * klu, 3 * klu);
        f += nrhs * 4.0 * klu * klu * (p - 1);
        return f;
    }

    // Refined solve: one low-precision solve plus 'refinements' double
    // residuals and correction solves.  The float mixed path uses one pass,
    // the half mixed path uses four.
    double mixed_solve_flops(const int n, const int kl, const int ku, const int nrhs, const int p, const int refinements) {
        return (1.0 + refinements) * solve_flops(n, kl, ku, nrhs, p) +
               (double)refinements * nrhs * (2.0 * band_nnz(n, kl, ku) + 3.0 * n);
    }

    struct measurement {
        int parts = 0;
        double factor_ms = 0, solve_ms = 0, relres = 0;
        double factor_flops = 0, solve_flops = 0;
        double mem_MB = 0; // device memory retained by the solver
    };

    [[nodiscard]] double total_ms(const measurement& m) { return m.factor_ms + m.solve_ms; }

    void print_header() {
        gspike::detail::print("{:<6} {:>8} {:>5} {:>5} {:>6} {:>10} {:>10} {:>10} {:>12} {:>12} {:>9} {:>10}\n", "type", "n", "klu", "nrhs", "parts", "factor_ms", "solve_ms", "total_ms", "factor_GFLOP/s", "solve_GFLOP/s", "mem_MB", "relres");
    }

    void print_row(const char* tag, const int n, const int klu, const int nrhs, const measurement& m) {
        const double fm = std::max(m.factor_ms, 1e-9);
        const double sm = std::max(m.solve_ms, 1e-9);
        gspike::detail::print("{:<6} {:>8} {:>5} {:>5} {:>6} {:>10.2f} {:>10.2f} {:>10.2f} {:>12.2f} {:>12.2f} {:>9.0f} {:>10.2e}\n", tag, n, klu, nrhs, m.parts, m.factor_ms, m.solve_ms, total_ms(m), m.factor_flops / (fm * 1e6), m.solve_flops / (sm * 1e6), m.mem_MB, m.relres);
    }

    // Shared GPU timing path.  make(device A, lda, opts) constructs the
    // factorized solver, so solver<T> and mixed_solver share this code.
    template<class T, class Factory>
    measurement measure(int n, const int kl, const int ku, int nrhs, const int parts, const int refinements, Factory make) {
        const int klu = std::max(kl, ku), lda = kl + ku + 1, reps = 5;
        std::vector<T> a, b;
        make_band<T>(n, klu, a, b, nrhs, lda);

        dev_buffer<T> da(a.size()), db(b.size());
        ck(cudaMemcpy(da.get(), a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice));

        gspike::opts o;
        o.parts = parts;
        o.full_spikes = g_full_spikes;
        o.auto_full_spikes = !g_no_full_spikes;
        o.profile = g_profile;

        size_t free0 = 0, total_mem = 0;
        ck(cudaMemGetInfo(&free0, &total_mem));

        // Warm up the allocator so the timed construction is comparable to the
        // reference, whose work arrays are allocated before timing starts.
        {
            auto warm = make(da.get(), lda, o);
        }

        const timer t;
        t.start();
        auto s = make(da.get(), lda, o);
        const double factor_ms = t.stop();
        size_t free_after_factor = 0;
        ck(cudaMemGetInfo(&free_after_factor, &total_mem));

        ck(cudaMemcpy(db.get(), b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice));
        s->solve(nrhs, db.get(), n); // warm up

        double solve_ms = 0;
        for(int r = 0; r < reps; ++r) {
            ck(cudaMemcpy(db.get(), b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice));
            t.start();
            s->solve(nrhs, db.get(), n);
            solve_ms += t.stop();
        }
        solve_ms /= reps;
        size_t free_after_solve = 0;
        ck(cudaMemGetInfo(&free_after_solve, &total_mem));

        ck(cudaMemcpy(db.get(), b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice));
        s->solve(nrhs, db.get(), n);
        std::vector<T> x(b.size());
        ck(cudaMemcpy(x.data(), db.get(), x.size() * sizeof(T), cudaMemcpyDeviceToHost));

        measurement m;
        m.parts = s->parts();
        m.factor_ms = factor_ms;
        m.solve_ms = solve_ms;
        m.relres = relres<T>(n, kl, ku, lda, nrhs, a, b, x);
        m.factor_flops = factor_flops(n, kl, ku, m.parts);
        m.solve_flops = refinements ? mixed_solve_flops(n, kl, ku, nrhs, m.parts, refinements) : solve_flops(n, kl, ku, nrhs, m.parts);
        const size_t free_low = std::min(free_after_factor, free_after_solve);
        m.mem_MB = free0 > free_low ? (double)(free0 - free_low) / 1048576.0 : 0.0;
        return m;
    }

    void run_gpu(const mode kind, int n, int klu, const int nrhs, const int parts) {
        const char* tag = (kind == mode::fp16) ? "float16" : (kind == mode::fp32) ? "float" :
                                                         (kind == mode::fp64)     ? "double" :
                                                         (kind == mode::mixed16)  ? "mixed16" :
                                                                                    "mixed";
        measurement m;
        if(kind == mode::fp16) {
            m = measure<__half>(n, klu, klu, nrhs, parts, 0, [&](const __half* a, int lda, const gspike::opts& o) {
                return std::make_unique<gspike::solver<__half>>(n, klu, klu, a, lda, o);
            });
        }
        else if(kind == mode::fp32) {
            m = measure<float>(n, klu, klu, nrhs, parts, 0, [&](const float* a, int lda, const gspike::opts& o) {
                return std::make_unique<gspike::solver<float>>(n, klu, klu, a, lda, o);
            });
        }
        else if(kind == mode::fp64) {
            m = measure<double>(n, klu, klu, nrhs, parts, 0, [&](const double* a, int lda, const gspike::opts& o) {
                return std::make_unique<gspike::solver<double>>(n, klu, klu, a, lda, o);
            });
        }
        else if(kind == mode::mixed16) {
            m = measure<double>(n, klu, klu, nrhs, parts, 4, [&](const double* a, int lda, const gspike::opts& o) {
                return std::make_unique<gspike::mixed_half_solver>(n, klu, klu, a, lda, o);
            });
        }
        else {
            m = measure<double>(n, klu, klu, nrhs, parts, 1, [&](const double* a, int lda, const gspike::opts& o) {
                return std::make_unique<gspike::mixed_solver>(n, klu, klu, a, lda, o);
            });
        }
        print_row(tag, n, klu, nrhs, m);
    }

#ifdef GSPIKE_HAVE_REF
    measurement measure_cpu(const int n, const int kl, const int ku, const int nrhs, const int nthreads) {
        const int klu = std::max(kl, ku), lda = kl + ku + 1, reps = 5;
        std::vector<double> a, b;
        make_band<double>(n, klu, a, b, nrhs, lda);
        std::vector<double> x = b;

        int parts = 0, info = 0;
        double factor_ms = 0;
        const int id = spike_ref_create(n, kl, ku, a.data(), lda, nthreads, &parts, &factor_ms, &info);
        if(id < 0 || info != 0) {
            gspike::detail::print_err("spike_bench: reference create failed (id={} info={})\n", id, info);
            std::exit(1);
        }

        double solve_ms = 0;
        int sinfo = spike_ref_solve(id, nrhs, x.data(), n, &solve_ms); // warm up
        double total = 0;
        for(int r = 0; r < reps; ++r) {
            std::ranges::copy(b, x.begin());
            sinfo = spike_ref_solve(id, nrhs, x.data(), n, &solve_ms);
            total += solve_ms;
        }
        spike_ref_destroy(id);
        if(sinfo != 0) {
            gspike::detail::print_err("spike_bench: reference solve failed (info={})\n", sinfo);
            std::exit(1);
        }

        measurement m;
        m.parts = parts;
        m.factor_ms = factor_ms;
        m.solve_ms = total / reps;
        m.relres = relres<double>(n, kl, ku, lda, nrhs, a, b, x);
        m.factor_flops = factor_flops(n, kl, ku, parts);
        m.solve_flops = solve_flops(n, kl, ku, nrhs, parts);
        return m;
    }
#endif

    void run_cpu(const int n, const int klu, const int nrhs, const int cpu_threads) {
#ifdef GSPIKE_HAVE_REF
        print_row("cpu", n, klu, nrhs, measure_cpu(n, klu, klu, nrhs, cpu_threads));
#else
        (void)n;
        (void)klu;
        (void)nrhs;
        (void)cpu_threads;
        gspike::detail::print_err("spike_bench: CPU reference unavailable (Fortran, LAPACK and OpenMP required)\n");
        std::exit(2);
#endif
    }

    void print_compare_header() {
        gspike::detail::print("{:>8} {:>5} {:>5} {:>6} {:>6} {:>10} {:>10} {:>7} {:>10} {:>10} {:>7} {:>10} {:>10} {:>7} {:>12} {:>12} {:>10} {:>10} {:>8}\n", "n", "klu", "nrhs", "cparts", "gparts", "cpu_fac_ms", "gpu_fac_ms", "fac_x", "cpu_sol_ms", "gpu_sol_ms", "sol_x", "cpu_tot_ms", "gpu_tot_ms", "tot_x", "cpu_GFLOP/s", "gpu_GFLOP/s", "cpu_relres", "gpu_relres", "gpu_mem_MB");
    }

    // CPU reference (double) against the GPU mixed solver, which factorizes in
    // float and corrects with a double residual.  This is the useful GPU path
    // for double-precision problems.
    void run_compare(int n, int klu, const int nrhs, const int gpu_parts, const int cpu_threads) {
#ifdef GSPIKE_HAVE_REF
        const measurement g = measure<double>(n, klu, klu, nrhs, gpu_parts, 1, [&](const double* a, int lda, const gspike::opts& o) {
            return std::make_unique<gspike::mixed_solver>(n, klu, klu, a, lda, o);
        });
        const measurement c = measure_cpu(n, klu, klu, nrhs, cpu_threads);
        const double fx = c.factor_ms / std::max(g.factor_ms, 1e-9);
        const double sx = c.solve_ms / std::max(g.solve_ms, 1e-9);
        const double tx = total_ms(c) / std::max(total_ms(g), 1e-9);
        gspike::detail::print("{:>8} {:>5} {:>5} {:>6} {:>6} {:>10.2f} {:>10.2f} {:>7.2f} {:>10.2f} {:>10.2f} {:>7.2f} {:>10.2f} {:>10.2f} {:>7.2f} {:>12.2f} {:>12.2f} {:>10.2e} {:>10.2e} {:>8.0f}\n", n, klu, nrhs, c.parts, g.parts, c.factor_ms, g.factor_ms, fx, c.solve_ms, g.solve_ms, sx, total_ms(c), total_ms(g), tx, c.factor_flops / (std::max(c.factor_ms, 1e-9) * 1e6), g.factor_flops / (std::max(g.factor_ms, 1e-9) * 1e6), c.relres, g.relres, g.mem_MB);
#else
        (void)n;
        (void)klu;
        (void)nrhs;
        (void)gpu_parts;
        (void)cpu_threads;
        gspike::detail::print_err("spike_bench: --compare requires the CPU reference (Fortran, LAPACK and OpenMP)\n");
        std::exit(2);
#endif
    }

    // Default sweeps use bands around 1% of n with a few right-hand sides.  The
    // largest case (n=80000, klu=800) peaks near 2.5 GB of device memory, so the
    // whole sweep fits comfortably in 4 GB.
    int pct_klu(const int n) { return std::max(1, (n + 50) / 100); }

    void sweep(const mode kind, const int cpu_threads) {
        if(kind == mode::compare) {
            print_compare_header();
            run_compare(10000, pct_klu(10000), 1, 0, cpu_threads);
            run_compare(25000, pct_klu(25000), 1, 0, cpu_threads);
            run_compare(50000, pct_klu(50000), 2, 0, cpu_threads);
            run_compare(80000, pct_klu(80000), 1, 0, cpu_threads);
            return;
        }

        print_header();
        run_gpu(kind, 10000, pct_klu(10000), 1, 0);
        run_gpu(kind, 25000, pct_klu(25000), 1, 0);
        run_gpu(kind, 50000, pct_klu(50000), 2, 0);
        run_gpu(kind, 80000, pct_klu(80000), 1, 0);
        if(kind != mode::mixed)
            run_gpu(kind, 25000, pct_klu(25000), 1, 1); // single partition = plain band LU
    }

    void sweep_cpu(const int cpu_threads) {
        print_header();
        run_cpu(10000, pct_klu(10000), 1, cpu_threads);
        run_cpu(25000, pct_klu(25000), 1, cpu_threads);
        run_cpu(50000, pct_klu(50000), 2, cpu_threads);
        run_cpu(80000, pct_klu(80000), 1, cpu_threads);
    }

    bool parse_type(const char* s, mode& m) {
        if(std::strcmp(s, "float16") == 0) m = mode::fp16;
        else if(std::strcmp(s, "float") == 0) m = mode::fp32;
        else if(std::strcmp(s, "double") == 0) m = mode::fp64;
        else if(std::strcmp(s, "mixed") == 0) m = mode::mixed;
        else if(std::strcmp(s, "mixed16") == 0) m = mode::mixed16;
        else if(std::strcmp(s, "cpu") == 0) m = mode::cpu;
        else return false;
        return true;
    }

} // namespace

int main(const int argc, char** argv) {
    auto kind = mode::fp64;
    // Eight threads balance the SPIKE partitions best on the 24-core EPYC
    // used for the reference numbers; sixteen leave the recursive reduced
    // system with too little work per worker.
    int cpu_threads = 8;
    int pos = 1;
    for(; pos < argc; ++pos) {
        if(std::strcmp(argv[pos], "--type") == 0 && pos + 1 < argc) {
            if(!parse_type(argv[++pos], kind)) {
                gspike::detail::print("usage: {} [--type float16|mixed16|float|double|mixed|cpu] [--compare] [--full-spikes|--no-full-spikes] [--profile] [--cpu-threads N] [n klu nrhs [parts]]\n", argv[0]);
                return 2;
            }
        }
        else if(std::strcmp(argv[pos], "--compare") == 0) {
            kind = mode::compare;
        }
        else if(std::strcmp(argv[pos], "--full-spikes") == 0) {
            g_full_spikes = true;
        }
        else if(std::strcmp(argv[pos], "--no-full-spikes") == 0) {
            g_no_full_spikes = true;
        }
        else if(std::strcmp(argv[pos], "--profile") == 0) {
            g_profile = true;
        }
        else if(std::strcmp(argv[pos], "--cpu-threads") == 0 && pos + 1 < argc) {
            cpu_threads = std::atoi(argv[++pos]);
        }
        else if(argv[pos][0] == '-') {
            gspike::detail::print("usage: {} [--type float16|mixed16|float|double|mixed|cpu] [--compare] [--full-spikes|--no-full-spikes] [--profile] [--cpu-threads N] [n klu nrhs [parts]]\n", argv[0]);
            return 2;
        }
        else {
            break;
        }
    }

    if(kind != mode::cpu) {
        int dev = 0;
        cudaDeviceProp prop{};
        if(cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
            gspike::detail::print("# GPU: {} (cc {}.{}, {} SMs, {:.0f} MB L2)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.l2CacheSize / 1048576.0);
        }
    }

    if(pos + 2 < argc) {
        const int n = std::atoi(argv[pos]);
        const int klu = std::atoi(argv[pos + 1]);
        const int nrhs = std::atoi(argv[pos + 2]);
        const int parts = (pos + 3 < argc) ? std::atoi(argv[pos + 3]) : 0;
        if(kind == mode::compare) {
            print_compare_header();
            run_compare(n, klu, nrhs, parts, cpu_threads);
        }
        else {
            print_header();
            if(kind == mode::cpu)
                run_cpu(n, klu, nrhs, cpu_threads);
            else
                run_gpu(kind, n, klu, nrhs, parts);
        }
        return 0;
    }

    if(kind == mode::cpu)
        sweep_cpu(cpu_threads);
    else
        sweep(kind, cpu_threads);
    return 0;
}
