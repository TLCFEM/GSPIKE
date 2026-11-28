// Correctness tests for gspike::spike_gbsv.
//
// Every solve is checked against a dense host reference (small problems) and
// against the relative residual ||b - A x|| / ||b||.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <gspike/spike.h>
#include <gspike/spike_half.h>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

    // The public API accepts the two instantiated scalar types, plus float16
    // when the toolchain provides std::float16_t.
    static_assert(gspike::supported_scalar<float> && gspike::supported_scalar<double>);
    static_assert(!gspike::supported_scalar<int> && !gspike::supported_scalar<long double>);
#if GSPIKE_HAS_FLOAT16_T
    static_assert(gspike::supported_scalar<std::float16_t>);
#endif
#if GSPIKE_HAS_CUDA_HALF
    static_assert(gspike::supported_scalar<__half>);
#endif
    static_assert(std::same_as<gspike::float_solver, gspike::solver<float>>);
    static_assert(std::same_as<gspike::double_solver, gspike::solver<double>>);
#if GSPIKE_HAS_CUDA_HALF
    static_assert(std::same_as<gspike::half_solver, gspike::solver<__half>>);
#endif

    int g_fail = 0;

    void check(const bool ok, const std::string& what) {
        gspike::detail::print("  [{}] {}\n", ok ? " ok " : "FAIL", what);
        if(!ok) ++g_fail;
    }

    // A failed CUDA runtime call must fail the test; the simulator would
    // otherwise print a warning and let the process exit successfully.
    inline void check_cuda(const cudaError_t e, const char* what) {
        if(e != cudaSuccess) {
            gspike::detail::print("  [FAIL] CUDA {}: {}\n", what, cudaGetErrorString(e));
            ++g_fail;
        }
    }
#define CUDA_CALL(expr) check_cuda((expr), #expr)

    // ---- host band storage: A(i,j) at ab[j*lda + ku + i-j], lda = kl+ku+1 ----
    struct band {
        int n = 0, kl = 0, ku = 0;
        std::vector<double> a;
        [[nodiscard]] int lda() const { return kl + ku + 1; }
        double& at(const int i, const int j) { return a[(long long)j * lda() + ku + i - j]; }
        [[nodiscard]] double at(const int i, const int j) const { return a[(long long)j * lda() + ku + i - j]; }
    };

    band make_random(const int n, const int kl, const int ku, const bool dominant, const unsigned seed, const double decay = 1.0) {
        band m;
        m.n = n;
        m.kl = kl;
        m.ku = ku;
        m.a.assign((size_t)m.lda() * n, 0.0);
        std::mt19937 gen(seed);
        std::uniform_real_distribution u(-1.0, 1.0);
        for(int j = 0; j < n; ++j) {
            double rowsum = 0.0;
            for(int i = std::max(0, j - ku); i <= std::min(n - 1, j + kl); ++i) {
                if(i == j) continue;
                const double v = u(gen) * std::pow(decay, std::abs(i - j));
                m.at(i, j) = v;
                rowsum += std::abs(v);
            }
            m.at(j, j) = dominant ? rowsum + 1.0 + u(gen) * 0.1 : u(gen) * 0.5;
        }
        return m;
    }

    // dense LU with partial pivoting (host reference)
    std::vector<double> dense_solve(const band& A, const std::vector<double>& b, const int nrhs) {
        const int n = A.n;
        std::vector<double> M((size_t)n * n), x = b;
        for(int j = 0; j < n; ++j)
            for(int i = std::max(0, j - A.ku); i <= std::min(n - 1, j + A.kl); ++i)
                M[(size_t)j * n + i] = A.at(i, j);
        for(int j = 0; j < n; ++j) {
            int p = j;
            for(int i = j + 1; i < n; ++i)
                if(std::abs(M[(size_t)j * n + i]) > std::abs(M[(size_t)j * n + p])) p = i;
            if(p != j) {
                for(int c = 0; c < n; ++c) std::swap(M[(size_t)c * n + j], M[(size_t)c * n + p]);
                for(int c = 0; c < nrhs; ++c) std::swap(x[(size_t)c * n + j], x[(size_t)c * n + p]);
            }
            const double d = M[(size_t)j * n + j];
            for(int i = j + 1; i < n; ++i) {
                const double l = M[(size_t)j * n + i] / d;
                if(l == 0.0) continue;
                M[(size_t)j * n + i] = l;
                for(int c = j + 1; c < n; ++c) M[(size_t)c * n + i] -= l * M[(size_t)c * n + j];
                for(int c = 0; c < nrhs; ++c) x[(size_t)c * n + i] -= l * x[(size_t)c * n + j];
            }
        }
        for(int j = n - 1; j >= 0; --j) {
            const double d = M[(size_t)j * n + j];
            for(int c = 0; c < nrhs; ++c) {
                double s = x[(size_t)c * n + j];
                for(int k = j + 1; k < n; ++k) s -= M[(size_t)k * n + j] * x[(size_t)c * n + k];
                x[(size_t)c * n + j] = s / d;
            }
        }
        return x;
    }

    // Dense LU determinant, reported as (sign, log|det|) to avoid overflow.
    std::pair<int, double> dense_det(const band& A) {
        const int n = A.n;
        std::vector<double> M((size_t)n * n);
        for(int j = 0; j < n; ++j)
            for(int i = std::max(0, j - A.ku); i <= std::min(n - 1, j + A.kl); ++i)
                M[(size_t)j * n + i] = A.at(i, j);
        int sgn = 1;
        double logabs = 0;
        for(int j = 0; j < n; ++j) {
            int p = j;
            for(int i = j + 1; i < n; ++i)
                if(std::abs(M[(size_t)j * n + i]) > std::abs(M[(size_t)j * n + p])) p = i;
            if(p != j) {
                for(int c = 0; c < n; ++c) std::swap(M[(size_t)c * n + j], M[(size_t)c * n + p]);
                sgn = -sgn;
            }
            const double d = M[(size_t)j * n + j];
            if(d == 0) return {0, 0};
            logabs += std::log(std::abs(d));
            if(d < 0) sgn = -sgn;
            for(int i = j + 1; i < n; ++i) {
                const double l = M[(size_t)j * n + i] / d;
                for(int c = j + 1; c < n; ++c) M[(size_t)c * n + i] -= l * M[(size_t)c * n + j];
            }
        }
        return {sgn, logabs};
    }

    template<class T>
    struct dev {
        T* p = nullptr;
        ~dev() {
            if(p) CUDA_CALL(cudaFree(p));
        }
        void alloc(const size_t n) { CUDA_CALL(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T))); }
    };

    template<class T>
    double residual(const band& A, const std::vector<T>& x, const std::vector<T>& b, const int nrhs) {
        double rn = 0, bn = 0;
        for(int c = 0; c < nrhs; ++c) {
            for(int i = 0; i < A.n; ++i) {
                double s = 0;
                for(int j = std::max(0, i - A.kl); j <= std::min(A.n - 1, i + A.ku); ++j)
                    s += A.at(i, j) * (double)x[(size_t)c * A.n + j];
                const double r = (double)b[(size_t)c * A.n + i] - s;
                rn += r * r;
                bn += (double)b[(size_t)c * A.n + i] * (double)b[(size_t)c * A.n + i];
            }
        }
        return std::sqrt(rn / std::max(bn, 1e-300));
    }

    template<class T>
    void run_case(const std::string& name, const band& A, const int nrhs, const unsigned seed, const bool reference, const int parts = 0, const double err_tol = -1.0, const double res_tol = -1.0) {
        const int n = A.n, lda = A.lda(), ldb = n;
        std::vector<T> h((size_t)lda * n), b((size_t)ldb * nrhs);
        for(size_t i = 0; i < h.size(); ++i) h[i] = (T)A.a[i];
        std::mt19937 gen(seed);
        std::uniform_real_distribution u(-1.0, 1.0);
        for(auto& v : b) v = (T)u(gen);

        dev<T> da, db;
        da.alloc(h.size());
        db.alloc(b.size());
        CUDA_CALL(cudaMemcpy(da.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(db.p, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice));

        gspike::opts o;
        o.parts = parts;
        gspike::stats st;
        const int info = gspike::spike_gbsv<T>(n, A.kl, A.ku, nrhs, da.p, lda, db.p, ldb, o, &st);

        std::vector<T> x(b.size());
        CUDA_CALL(cudaMemcpy(x.data(), db.p, x.size() * sizeof(T), cudaMemcpyDeviceToHost));

        std::vector<T> back((size_t)lda * n);
        CUDA_CALL(cudaMemcpy(back.data(), da.p, back.size() * sizeof(T), cudaMemcpyDeviceToHost));
        const bool a_ok = (back == h);

        const double res = residual<T>(A, x, b, nrhs);
        const double eps = (sizeof(T) == 4) ? 1e-5 : 1e-11;
        const double tol = (err_tol > 0) ? err_tol : eps;

        std::string s = name + " (n=" + std::to_string(n) + " kl=" + std::to_string(A.kl) +
                        " ku=" + std::to_string(A.ku) + " nrhs=" + std::to_string(nrhs) +
                        " parts=" + std::to_string(st.parts) +
                        ")";
        gspike::detail::print("{}  info={} relres={:.2e}\n", s, info, res);
        check(info == 0, s + ": info");
        const double rtol = res_tol > 0 ? res_tol : (sizeof(T) == 4 ? 2e-4 : 2e-10);
        check(res < rtol, s + ": residual");
        check(a_ok, s + ": A untouched");

        if(reference && n <= 700) {
            std::vector<double> bd(b.size()), xd;
            for(size_t i = 0; i < b.size(); ++i) bd[i] = (double)b[i];
            xd = dense_solve(A, bd, nrhs);
            double num = 0, den = 0;
            for(size_t i = 0; i < xd.size(); ++i) {
                num = std::max(num, std::abs((double)x[i] - xd[i]));
                den = std::max(den, std::abs(xd[i]));
            }
            const double err = num / std::max(den, 1e-300);
            gspike::detail::print("       error vs dense = {:.2e} (tol {:.1e})\n", err, tol);
            check(err < tol, s + ": matches dense solve");
        }
    }

    // The warp-per-column solve needs a wide band and >=512 chains.  Its last
    // block is usually only partly filled; those inactive warps must still
    // reach every __syncthreads() or the CTA deadlocks.  Use a sampled residual
    // to keep these many-RHS cases cheap.
    template<class T>
    void warp_solve_case(const char* name, const int n, const int klu, const int nrhs, const int parts) {
        const band A = make_random(n, klu, klu, true, 77);
        const int lda = A.lda();
        std::vector<T> h((size_t)lda * n);
        for(size_t i = 0; i < h.size(); ++i) h[i] = (T)A.a[i];
        std::vector<double> b((size_t)n * nrhs);
        std::mt19937 gen(78);
        std::uniform_real_distribution u(-1.0, 1.0);
        for(auto& v : b) v = u(gen);
        std::vector<T> bh(b.size());
        for(size_t i = 0; i < b.size(); ++i) bh[i] = (T)b[i];

        dev<T> da;
        dev<T> db;
        da.alloc(h.size());
        db.alloc(bh.size());
        CUDA_CALL(cudaMemcpy(da.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(db.p, bh.data(), bh.size() * sizeof(T), cudaMemcpyHostToDevice));
        gspike::opts o;
        o.parts = parts;
        const int info = gspike::spike_gbsv<T>(n, klu, klu, nrhs, da.p, lda, db.p, n, o, nullptr);
        std::vector<T> x(bh.size());
        CUDA_CALL(cudaMemcpy(x.data(), db.p, x.size() * sizeof(T), cudaMemcpyDeviceToHost));

        double worst = 0, scale = 0;
        const int cstep = std::max(1, nrhs / 16), istep = std::max(1, n / 32);
        for(int c = 0; c < nrhs; c += cstep)
            for(int i = 0; i < n; i += istep) {
                double s = 0;
                for(int j = std::max(0, i - klu); j <= std::min(n - 1, i + klu); ++j)
                    s += (double)A.at(i, j) * (double)x[(size_t)c * n + j];
                worst = std::max(worst, std::abs(b[(size_t)c * n + i] - s));
            }
        for(double v : b) scale = std::max(scale, std::abs(v));
        const double rel = worst / std::max(scale, 1e-300);
        gspike::detail::print("  {} (n={} klu={} nrhs={} parts={}): info={} sampled relres={:.2e}\n", name, n, klu, nrhs, parts, info, rel);
        check(info == 0 && std::isfinite(rel) && rel < (sizeof(T) == 4 ? 1e-3 : 1e-10), std::string(name) + ": warp solve");
    }

    // A single partition with more than 65535*8 rows exercises the capped
    // make_blocks grid.  The band is tiny, so this covers the launch
    // configuration without a multi-GB allocation.
    void large_partition_case() {
        constexpr int n = 550000; // ceil(n/8) = 68750 > 65535
        constexpr int kl = 1, ku = 1, lda = kl + ku + 1;
        std::vector<float> h((size_t)lda * n, 0.0f), b(n);
        for(int j = 0; j < n; ++j) {
            h[(size_t)j * lda + 1] = 4.0f;
            if(j > 0) h[(size_t)j * lda + 0] = -0.25f;
            if(j + 1 < n) h[(size_t)j * lda + 2] = -0.25f;
            b[j] = (float)((j % 17) - 8) / 8.0f;
        }
        dev<float> da, db;
        da.alloc(h.size());
        db.alloc(b.size());
        CUDA_CALL(cudaMemcpy(da.p, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(db.p, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));

        gspike::opts o;
        o.parts = 1;
        o.auto_full_spikes = false;
        const int info = gspike::spike_gbsv<float>(n, kl, ku, 1, da.p, lda, db.p, n, o, nullptr);
        std::vector<float> x(b.size());
        CUDA_CALL(cudaMemcpy(x.data(), db.p, x.size() * sizeof(float), cudaMemcpyDeviceToHost));

        double worst = 0, scale = 0;
        for(int i = 0; i < n; ++i) {
            double s = 0;
            for(int j = std::max(0, i - kl); j <= std::min(n - 1, i + ku); ++j)
                s += (double)h[(size_t)j * lda + ku + i - j] * (double)x[j];
            worst = std::max(worst, std::abs((double)b[i] - s));
            scale = std::max(scale, std::abs((double)b[i]));
        }
        const double rel = worst / std::max(scale, 1e-300);
        gspike::detail::print("large partition (n={} parts=1): info={} relres={:.2e}\n", n, info, rel);
        check(info == 0 && std::isfinite(rel) && rel < 1e-4, "partition above CUDA grid.y limit");
    }

    template<class T>
    void suite(const char* tag) {
        gspike::detail::print("=== {} ===\n", tag);

        { // tridiagonal SPD
            const band A = make_random(97, 1, 1, true, 1);
            run_case<T>("tridiagonal", A, 1, 2, true);
        }
        { // small, several RHS: the direct factorization must be enough
            const band A = make_random(61, 3, 3, true, 3);
            run_case<T>("dd banded", A, 5, 4, true);
        }
        { // non symmetric bandwidths, forced SPIKE partitions
            const band A = make_random(213, 2, 6, true, 5);
            run_case<T>("kl<ku", A, 3, 6, true, 4);
            const band B = make_random(211, 7, 2, true, 7);
            run_case<T>("kl>ku", B, 2, 8, true, 4);
        }
        { // many partitions on a small matrix (forced)
            const band A = make_random(400, 4, 4, true, 9);
            run_case<T>("forced 16 parts", A, 4, 10, true, 16);
            run_case<T>("single partition", A, 4, 10, true, 1);
        }
        { // larger, residual check only
            const band A = make_random(3000, 6, 6, true, 11);
            run_case<T>("large", A, 2, 12, false);
        }
        { // wide bands use the blocked spike solve; the sparse C-spike forward
          // sweep must agree with the full sweep, including asymmetric bands.
            const band A = make_random(600, 60, 60, true, 31);
            run_case<T>("wide band", A, 2, 31, true, 4);
            const band B = make_random(2000, 80, 55, true, 32);
            run_case<T>("wide kl>ku", B, 2, 32, false, 4);
            // One-sided bands: the forward (or backward) sweep has zero width
            // and must not produce a zero-sized launch.
            const band U = make_random(2000, 0, 80, true, 33);
            run_case<T>("wide upper only", U, 1, 33, false, 4);
            const band L = make_random(2000, 80, 0, true, 34);
            run_case<T>("wide lower only", L, 1, 34, false, 4);
        }
        { // wide bands with many RHS: W=2 and W=4 warp solves, including
          // partially filled terminal blocks (517 % 16 = 5, 523 % 8 = 3).
            warp_solve_case<T>("warp W=2 tail", 1200, 200, 517, 2);
            warp_solve_case<T>("warp W=4 tail", 1600, 375, 523, 2);
        }
        { // non diagonally dominant: needs pivoting.  In float the attainable
          // accuracy is limited by eps * cond(A), so the tolerance is loose.
            const band A = make_random(150, 3, 3, false, 13, 0.9);
            run_case<T>("non dominant", A, 2, 14, true, 0, sizeof(T) == 4 ? 5e-3 : 1e-7);
            gspike::opts o;
            o.full_spikes = true; // pivoted fallback + saved full spikes
            const int n = A.n, lda = A.lda(), nrhs = 2;
            std::vector<T> h((size_t)lda * n);
            for(size_t i = 0; i < h.size(); ++i) h[i] = (T)A.a[i];
            std::mt19937 gen(39);
            std::uniform_real_distribution u(-1.0, 1.0);
            std::vector<T> b((size_t)n * nrhs);
            for(auto& v : b) v = (T)u(gen);
            dev<T> da, db;
            da.alloc(h.size());
            db.alloc(b.size());
            CUDA_CALL(cudaMemcpy(da.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(db.p, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice));
            const int info = gspike::spike_gbsv<T>(n, A.kl, A.ku, nrhs, da.p, lda, db.p, n, o, nullptr);
            std::vector<T> x(b.size());
            CUDA_CALL(cudaMemcpy(x.data(), db.p, x.size() * sizeof(T), cudaMemcpyDeviceToHost));
            check(info == 0 && residual<T>(A, x, b, nrhs) < (sizeof(T) == 4 ? 5e-3 : 1e-7), "full spikes with pivoting");
        }
        { // opts::profile exercises the event-timed path without changing results
            const band A = make_random(97, 2, 2, true, 40);
            const int n = A.n, lda = A.lda();
            std::vector<T> h((size_t)lda * n);
            for(size_t i = 0; i < h.size(); ++i) h[i] = (T)A.a[i];
            std::vector<T> b(n, (T)1);
            dev<T> da, db;
            da.alloc(h.size());
            db.alloc(b.size());
            CUDA_CALL(cudaMemcpy(da.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(db.p, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice));
            gspike::opts o;
            o.profile = true;
            const int info = gspike::spike_gbsv<T>(n, 2, 2, 1, da.p, lda, db.p, n, o, nullptr);
            std::vector<T> x(b.size());
            CUDA_CALL(cudaMemcpy(x.data(), db.p, x.size() * sizeof(T), cudaMemcpyDeviceToHost));
            check(info == 0 && residual<T>(A, x, b, 1) < (sizeof(T) == 4 ? 2e-4 : 2e-10), "opts::profile leaves results unchanged");
        }
        { // degenerate band shapes
            const band D = make_random(37, 0, 0, true, 19);
            run_case<T>("diagonal", D, 2, 20, true);
            const band S = make_random(1, 0, 0, true, 21);
            run_case<T>("1x1", S, 1, 22, true);
        }
        { // determinant and its sign are read from the factorization
            for(int parts : {0, 1, 3}) {
                const band A = make_random(24, 2, 2, true, 700 + parts);
                const auto ref = dense_det(A);
                const int n = A.n, lda = A.lda();
                std::vector<T> h((size_t)lda * n);
                for(size_t i = 0; i < h.size(); ++i) h[i] = (T)A.a[i];
                dev<T> da;
                da.alloc(h.size());
                CUDA_CALL(cudaMemcpy(da.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
                gspike::opts o;
                o.parts = parts;
                gspike::solver<T> s(n, A.kl, A.ku, da.p, lda, o);
                const T d = s.det();
                const int sg = s.det_sign();
                const double got_log = std::log(std::abs((double)d));
                const double err = std::abs(got_log - ref.second) / std::max(1.0, std::abs(ref.second));
                gspike::detail::print("  det (parts={}): sign={}/{} logabs={:.6f}/{:.6f}\n", s.parts(), sg, ref.first, got_log, ref.second);
                check(std::isfinite((double)d) && sg == ref.first && err < (sizeof(T) == 4 ? 1e-4 : 1e-10), "det/det_sign matches dense LU");
            }
            { // a perturbed pivot makes the determinant undefined -> zero
                const int n = 12, kl = 1, ku = 1, lda = kl + ku + 1;
                std::vector<T> h((size_t)lda * n, T(0));
                for(int j = 0; j < n; ++j) h[(size_t)j * lda + ku] = (j == 5) ? T(0) : T(1);
                dev<T> da;
                da.alloc(h.size());
                CUDA_CALL(cudaMemcpy(da.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
                gspike::solver<T> s(n, kl, ku, da.p, lda);
                check(s.det() == T(0) && s.det_sign() == 0, "singular matrix: det and det_sign are zero");
            }
        }
        { // reuse: factor once, two different RHS
            const band A = make_random(500, 5, 4, true, 15);
            const int n = A.n, lda = A.lda(), nrhs = 3;
            std::vector<T> h((size_t)lda * n);
            for(size_t i = 0; i < h.size(); ++i) h[i] = (T)A.a[i];
            dev<T> da;
            da.alloc(h.size());
            CUDA_CALL(cudaMemcpy(da.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
            gspike::solver<T> s(n, A.kl, A.ku, da.p, lda);
            std::mt19937 gen(16);
            std::uniform_real_distribution u(-1.0, 1.0);
            std::vector<T> b0((size_t)n * nrhs);
            for(auto& v : b0) v = (T)u(gen);
            dev<T> db;
            db.alloc(b0.size());
            bool ok = true;
            for(int rep = 0; rep < 2; ++rep) {
                CUDA_CALL(cudaMemcpy(db.p, b0.data(), b0.size() * sizeof(T), cudaMemcpyHostToDevice));
                const int info = s.solve(nrhs, db.p, n);
                std::vector<T> x(b0.size());
                CUDA_CALL(cudaMemcpy(x.data(), db.p, x.size() * sizeof(T), cudaMemcpyDeviceToHost));
                const double r = residual<T>(A, x, b0, nrhs);
                ok = ok && info == 0 && r < (sizeof(T) == 4 ? 2e-4 : 2e-10);
            }
            check(ok, "solver reuse (parts=" + std::to_string(s.parts()) + ")");
        }
        { // full-spike correction: same answer with the option on and off
            const band A = make_random(420, 6, 5, true, 41);
            const int n = A.n, lda = A.lda(), nrhs = 3;
            std::vector<T> h((size_t)lda * n);
            for(size_t i = 0; i < h.size(); ++i) h[i] = (T)A.a[i];
            std::mt19937 gen(42);
            std::uniform_real_distribution u(-1.0, 1.0);
            std::vector<T> b((size_t)n * nrhs);
            for(auto& v : b) v = (T)u(gen);
            dev<T> da, dx0, dx1;
            da.alloc(h.size());
            dx0.alloc(b.size());
            dx1.alloc(b.size());
            CUDA_CALL(cudaMemcpy(da.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(dx0.p, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(dx1.p, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice));
            gspike::opts o0, o1;
            o0.auto_full_spikes = false; // compare the plain two-solve path
            o1.full_spikes = true;
            gspike::solver<T> s0(n, A.kl, A.ku, da.p, lda, o0);
            gspike::solver<T> s1(n, A.kl, A.ku, da.p, lda, o1);
            const int i0 = s0.solve(nrhs, dx0.p, n);
            const int i1 = s1.solve(nrhs, dx1.p, n);
            std::vector<T> x0(b.size()), x1(b.size());
            CUDA_CALL(cudaMemcpy(x0.data(), dx0.p, x0.size() * sizeof(T), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaMemcpy(x1.data(), dx1.p, x1.size() * sizeof(T), cudaMemcpyDeviceToHost));
            double diff = 0, scale = 0;
            for(size_t i = 0; i < b.size(); ++i) {
                diff = std::max(diff, std::abs((double)x1[i] - (double)x0[i]));
                scale = std::max(scale, std::abs((double)x0[i]));
            }
            const double res = residual<T>(A, x1, b, nrhs);
            const double tol = (sizeof(T) == 4) ? 1e-4 : 1e-10;
            check(i0 == 0 && i1 == 0 && res < (sizeof(T) == 4 ? 2e-4 : 2e-10) && diff < scale * tol, "full-spike correction matches standard solve");
        }
        { // randomized fuzz: shapes, bandwidths, right-hand sides and partitions
            std::mt19937 gen(23);
            std::uniform_int_distribution<int> n_dist(2, 260), band_dist(0, 7), rhs_dist(1, 5), part_dist(0, 8);
            for(int t = 0; t < 12; ++t) {
                const int n = n_dist(gen), kl = band_dist(gen), ku = band_dist(gen);
                const bool dominant = (t % 4) != 3;
                const int parts = std::min(part_dist(gen), std::max(1, n / std::max(1, 2 * std::max(kl, ku) + 1)));
                const band A = make_random(n, kl, ku, dominant, 100 + t, dominant ? 1.0 : 0.92);
                const double tol = dominant ? -1.0 : (sizeof(T) == 4 ? 5e-3 : 1e-7);
                run_case<T>("fuzz " + std::to_string(t), A, rhs_dist(gen), 200 + t, true, parts, tol);
            }
        }
        { // invalid arguments
            const band A = make_random(10, 1, 1, true, 17);
            std::vector<T> h((size_t)A.lda() * A.n), b(10);
            for(size_t i = 0; i < h.size(); ++i) h[i] = (T)A.a[i];
            dev<T> da, db;
            da.alloc(h.size());
            db.alloc(b.size());
            CUDA_CALL(cudaMemcpy(da.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
            check(gspike::spike_gbsv<T>(10, 1, 1, 1, da.p, 1, db.p, 10) < 0, "bad lda rejected");
            check(gspike::spike_gbsv<T>(10, 1, 1, 1, da.p, A.lda(), db.p, 5) < 0, "bad ldb rejected");
            check(gspike::spike_gbsv<T>(10, 1, 1, 1, nullptr, A.lda(), db.p, 10) < 0, "null A rejected");
            check(gspike::spike_gbsv<T>(10, 1, 1, 0, da.p, A.lda(), db.p, 10) == 0, "zero right-hand sides accepted");
            gspike::solver<T> s(10, 1, 1, da.p, A.lda());
            check(s.solve(-1, db.p, 10) < 0, "solver: negative nrhs rejected");
            check(s.solve(1, nullptr, 10) < 0, "solver: null B rejected");
            check(s.solve(1, db.p, 9) < 0, "solver: bad ldb rejected");
            check(s.solve(0, db.p, 10) == 0, "solver: zero right-hand sides accepted");
            gspike::opts capped;
            capped.parts = 128; // clamped to kMaxParts / the partition limit
            gspike::stats st;
            check(gspike::spike_gbsv<T>(10, 1, 1, 1, da.p, A.lda(), db.p, 10, capped, &st) == 0 && st.parts <= 3, "excess parts clamped");
        }
    }

    void mixed_suite() {
        gspike::detail::print("=== mixed (float factorization, double correction) ===\n");

        const band A = make_random(420, 5, 4, true, 31);
        const int n = A.n, lda = A.lda(), nrhs = 3;
        std::vector<double> h((size_t)lda * n), b((size_t)n * nrhs);
        for(size_t i = 0; i < h.size(); ++i) h[i] = A.a[i];
        std::mt19937 gen(32);
        std::uniform_real_distribution u(-1.0, 1.0);
        for(auto& v : b) v = u(gen);

        dev<double> da, db;
        da.alloc(h.size());
        db.alloc(b.size());
        CUDA_CALL(cudaMemcpy(da.p, h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice));

        gspike::mixed_solver s(n, A.kl, A.ku, da.p, lda);
        bool ok = true;
        for(int rep = 0; rep < 2; ++rep) { // reuse the factorization
            CUDA_CALL(cudaMemcpy(db.p, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice));
            gspike::stats st;
            const int info = s.solve(nrhs, db.p, n, &st);
            std::vector<double> x(b.size());
            CUDA_CALL(cudaMemcpy(x.data(), db.p, x.size() * sizeof(double), cudaMemcpyDeviceToHost));

            const double res = residual<double>(A, x, b, nrhs);
            const std::vector<double> xd = dense_solve(A, b, nrhs);
            double num = 0, den = 0;
            for(size_t i = 0; i < xd.size(); ++i) {
                num = std::max(num, std::abs(x[i] - xd[i]));
                den = std::max(den, std::abs(xd[i]));
            }
            const double err = num / std::max(den, 1e-300);
            gspike::detail::print("  mixed reuse {}: info={} parts={} relres={:.2e} err={:.2e}\n", rep, info, st.parts, res, err);
            ok = ok && info == 0 && st.parts == s.parts() && res < 1e-12 && err < 1e-11;
        }
        check(ok, "mixed solve reaches double precision and reuses the factorization");

        std::vector<double> back((size_t)lda * n);
        CUDA_CALL(cudaMemcpy(back.data(), da.p, back.size() * sizeof(double), cudaMemcpyDeviceToHost));
        check(back == h, "mixed: A untouched");

        { // mixed det/det_sign come from the float factorization
            const band M = make_random(28, 2, 2, true, 71);
            const auto ref = dense_det(M);
            const int nn = M.n, ldm = M.lda();
            std::vector<double> ah((size_t)ldm * nn);
            for(size_t i = 0; i < ah.size(); ++i) ah[i] = M.a[i];
            dev<double> dA;
            dA.alloc(ah.size());
            CUDA_CALL(cudaMemcpy(dA.p, ah.data(), ah.size() * sizeof(double), cudaMemcpyHostToDevice));
            gspike::mixed_solver ms(nn, M.kl, M.ku, dA.p, ldm);
            const double d = ms.det();
            const double err = std::abs(std::log(std::abs(d)) - ref.second) / std::max(1.0, std::abs(ref.second));
            gspike::detail::print("  mixed det: sign={}/{} logabs={:.6f}/{:.6f}\n", ms.det_sign(), ref.first, std::log(std::abs(d)), ref.second);
            check(std::isfinite(d) && ms.det_sign() == ref.first && err < 1e-3, "mixed det/det_sign matches dense LU");
        }

        { // mixed sweep over bandwidth shapes and forced partition counts
            bool ok2 = true;
            const int shapes[][3] = {{256, 2, 4}, {300, 5, 2}, {200, 0, 6}};
            for(int t = 0; t < 3; ++t) {
                const int nn = shapes[t][0], kl = shapes[t][1], ku = shapes[t][2];
                const band M = make_random(nn, kl, ku, true, 60 + t);
                const int ldam = M.lda(), nr = 3;
                std::vector<double> ah((size_t)ldam * nn), bh((size_t)nn * nr);
                for(size_t i = 0; i < ah.size(); ++i) ah[i] = M.a[i];
                for(auto& v : bh) v = u(gen);
                dev<double> dA, dB;
                dA.alloc(ah.size());
                dB.alloc(bh.size());
                CUDA_CALL(cudaMemcpy(dA.p, ah.data(), ah.size() * sizeof(double), cudaMemcpyHostToDevice));
                CUDA_CALL(cudaMemcpy(dB.p, bh.data(), bh.size() * sizeof(double), cudaMemcpyHostToDevice));
                gspike::opts mo;
                mo.parts = t == 0 ? 3 : 0;
                gspike::mixed_solver ms(nn, kl, ku, dA.p, ldam, mo);
                const int mi = ms.solve(nr, dB.p, nn);
                std::vector<double> mx(bh.size());
                CUDA_CALL(cudaMemcpy(mx.data(), dB.p, mx.size() * sizeof(double), cudaMemcpyDeviceToHost));
                const double mr = residual<double>(M, mx, bh, nr);
                gspike::detail::print("  mixed shape {} (n={} kl={} ku={} parts={}): info={} relres={:.2e}\n", t, nn, kl, ku, ms.parts(), mi, mr);
                ok2 = ok2 && mi == 0 && mr < 1e-11;
            }
            check(ok2, "mixed: bandwidth and partition sweep");
        }

        check(s.solve(1, nullptr, n) < 0, "mixed: null B rejected");
        check(s.solve(1, db.p, n - 1) < 0, "mixed: bad ldb rejected");
        check(s.solve(0, db.p, n) == 0, "mixed: zero right-hand sides accepted");
        bool threw = false;
        try {
            gspike::mixed_solver bad(n, A.kl, A.ku, da.p, lda - 1);
        }
        catch(const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "mixed: bad lda rejected");
    }

    void mixed_half_suite() {
        gspike::detail::print("=== mixed16 (__half factorization, 4 double refinements) ===\n");

        const band A = make_random(420, 5, 4, true, 31);
        const int n = A.n, lda = A.lda(), nrhs = 3;
        std::vector<double> h((size_t)lda * n), b((size_t)n * nrhs);
        for(size_t i = 0; i < h.size(); ++i) h[i] = A.a[i];
        std::mt19937 gen(32);
        std::uniform_real_distribution u(-1.0, 1.0);
        for(auto& v : b) v = u(gen);

        dev<double> da, db;
        da.alloc(h.size());
        db.alloc(b.size());
        CUDA_CALL(cudaMemcpy(da.p, h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice));

        gspike::mixed_half_solver s(n, A.kl, A.ku, da.p, lda);
        bool ok = true;
        for(int rep = 0; rep < 2; ++rep) { // reuse the factorization
            CUDA_CALL(cudaMemcpy(db.p, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice));
            gspike::stats st;
            const int info = s.solve(nrhs, db.p, n, &st);
            std::vector<double> x(b.size());
            CUDA_CALL(cudaMemcpy(x.data(), db.p, x.size() * sizeof(double), cudaMemcpyDeviceToHost));

            const double res = residual<double>(A, x, b, nrhs);
            gspike::detail::print("  mixed16 reuse {}: info={} parts={} relres={:.2e}\n", rep, info, st.parts, res);
            ok = ok && info == 0 && st.parts == s.parts() && res < 1e-13;
        }
        check(ok, "mixed16: four refinements recover double precision");

        std::vector<double> back((size_t)lda * n);
        CUDA_CALL(cudaMemcpy(back.data(), da.p, back.size() * sizeof(double), cudaMemcpyDeviceToHost));
        check(back == h, "mixed16: A untouched");

        check(s.solve(1, nullptr, n) < 0, "mixed16: null B rejected");
        check(s.solve(1, db.p, n - 1) < 0, "mixed16: bad ldb rejected");
        check(s.solve(0, db.p, n) == 0, "mixed16: zero right-hand sides accepted");
        bool threw = false;
        try {
            gspike::mixed_half_solver bad(n, A.kl, A.ku, da.p, lda - 1);
        }
        catch(const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "mixed16: bad lda rejected");
    }

    void half_suite() {
        gspike::detail::print("=== half (__half factorization) ===\n");

        // Half factors reach ~1e-3 relative residuals; float tolerances do not
        // apply.  The one-shot API covers tridiagonal, forced partitions,
        // plain band LU, wide bands and the pivoted fallback.
        run_case<__half>("half tridiagonal", make_random(97, 1, 1, true, 101), 1, 102, false, 0, -1.0, 1e-2);
        run_case<__half>("half forced parts", make_random(400, 4, 4, true, 103), 4, 104, false, 4, -1.0, 1e-2);
        run_case<__half>("half single part", make_random(400, 4, 4, true, 103), 4, 105, false, 1, -1.0, 1e-2);
        run_case<__half>("half wide band", make_random(600, 60, 60, true, 106), 2, 107, false, 4, -1.0, 1e-2);
        run_case<__half>("half non dominant", make_random(150, 3, 3, false, 108, 0.9), 2, 109, false, 0, -1.0, 1e-1);

        // Reuse the factorization and check the determinant API.
        {
            const band A = make_random(12, 3, 2, true, 110);
            const int n = A.n, lda = A.lda(), nrhs = 2;
            std::vector<__half> ah((size_t)lda * n);
            for(size_t i = 0; i < ah.size(); ++i) ah[i] = (__half)A.a[i];
            dev<__half> da, db;
            da.alloc(ah.size());
            db.alloc((size_t)n * nrhs);
            CUDA_CALL(cudaMemcpy(da.p, ah.data(), ah.size() * sizeof(__half), cudaMemcpyHostToDevice));

            gspike::half_solver s(n, A.kl, A.ku, da.p, lda);
            // The magnitude of a half determinant can overflow (>65504); the
            // sign is accumulated separately and must still match dense LU.
            const auto ref = dense_det(A);
            gspike::detail::print("  half det={:.3e} sign={}/{} (overflow is expected in half)\n", (double)s.det(), s.det_sign(), ref.first);
            bool ok = s.det_sign() == ref.first;
            std::mt19937 gen(111);
            std::uniform_real_distribution u(-1.0, 1.0);
            for(int rep = 0; rep < 2; ++rep) {
                std::vector<__half> b((size_t)n * nrhs);
                for(auto& v : b) v = (__half)u(gen);
                CUDA_CALL(cudaMemcpy(db.p, b.data(), b.size() * sizeof(__half), cudaMemcpyHostToDevice));
                const int info = s.solve(nrhs, db.p, n);
                std::vector<__half> x(b.size());
                CUDA_CALL(cudaMemcpy(x.data(), db.p, x.size() * sizeof(__half), cudaMemcpyDeviceToHost));
                std::vector<double> bd(b.size()), xd(x.size());
                for(size_t i = 0; i < b.size(); ++i) {
                    bd[i] = (double)b[i];
                    xd[i] = (double)x[i];
                }
                const double res = residual<double>(A, xd, bd, nrhs);
                gspike::detail::print("  half reuse {}: info={} parts={} relres={:.2e}\n", rep, info, s.parts(), res);
                ok = ok && info == 0 && res < 1e-2;
            }
            check(ok, "half: solver reuse, determinant and residual");
        }
    }

} // namespace

int main() {
    suite<float>("float");
    large_partition_case();
    suite<double>("double");
    mixed_suite();
    mixed_half_suite();
    half_suite();
    gspike::detail::print("\n{} ({} failure{})\n", g_fail == 0 ? "ALL TESTS PASSED" : "FAILURES", g_fail, g_fail == 1 ? "" : "s");
    return g_fail == 0 ? 0 : 1;
}
