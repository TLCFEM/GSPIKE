// Correctness tests for gspike::spike_gbsv.
//
// Every solve is checked against a dense host reference (small problems) and
// against the relative residual ||b - A x|| / ||b||.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <gspike/index.h>
#include <gspike/spike.h>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

    int g_fail = 0;

    void check(const bool ok, const std::string& what) {
        std::printf("  [%s] %s\n", ok ? " ok " : "FAIL", what.c_str());
        if(!ok) ++g_fail;
    }

    // ---- host band storage: A(i,j) at ab[j*lda + ku + i-j], lda = kl+ku+1 ----
    struct band {
        int n = 0, kl = 0, ku = 0;
        std::vector<double> a;
        [[nodiscard]] int lda() const { return kl + ku + 1; }
        double& at(const int i, const int j) { return a[static_cast<long long>(j) * lda() + ku + i - j]; }
        [[nodiscard]] double at(const int i, const int j) const { return a[static_cast<long long>(j) * lda() + ku + i - j]; }
    };

    band make_random(const int n, const int kl, const int ku, const bool dominant, const unsigned seed, const double decay = 1.0) {
        band m;
        m.n = n;
        m.kl = kl;
        m.ku = ku;
        m.a.assign(static_cast<size_t>(m.lda()) * n, 0.0);
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
        std::vector<double> M(static_cast<size_t>(n) * n), x = b;
        for(int j = 0; j < n; ++j)
            for(int i = std::max(0, j - A.ku); i <= std::min(n - 1, j + A.kl); ++i)
                M[static_cast<size_t>(j) * n + i] = A.at(i, j);
        for(int j = 0; j < n; ++j) {
            int p = j;
            for(int i = j + 1; i < n; ++i)
                if(std::abs(M[static_cast<size_t>(j) * n + i]) > std::abs(M[static_cast<size_t>(j) * n + p])) p = i;
            if(p != j) {
                for(int c = 0; c < n; ++c) std::swap(M[static_cast<size_t>(c) * n + j], M[static_cast<size_t>(c) * n + p]);
                for(int c = 0; c < nrhs; ++c) std::swap(x[static_cast<size_t>(c) * n + j], x[static_cast<size_t>(c) * n + p]);
            }
            const double d = M[static_cast<size_t>(j) * n + j];
            for(int i = j + 1; i < n; ++i) {
                const double l = M[static_cast<size_t>(j) * n + i] / d;
                if(l == 0.0) continue;
                M[static_cast<size_t>(j) * n + i] = l;
                for(int c = j + 1; c < n; ++c) M[static_cast<size_t>(c) * n + i] -= l * M[static_cast<size_t>(c) * n + j];
                for(int c = 0; c < nrhs; ++c) x[static_cast<size_t>(c) * n + i] -= l * x[static_cast<size_t>(c) * n + j];
            }
        }
        for(int j = n - 1; j >= 0; --j) {
            const double d = M[static_cast<size_t>(j) * n + j];
            for(int c = 0; c < nrhs; ++c) {
                double s = x[static_cast<size_t>(c) * n + j];
                for(int k = j + 1; k < n; ++k) s -= M[static_cast<size_t>(k) * n + j] * x[static_cast<size_t>(c) * n + k];
                x[static_cast<size_t>(c) * n + j] = s / d;
            }
        }
        return x;
    }

    template<class T>
    struct dev {
        T* p = nullptr;
        ~dev() {
            if(p) cudaFree(p);
        }
        void alloc(const size_t n) { cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)); }
    };

    template<class T>
    double residual(const band& A, const std::vector<T>& x, const std::vector<T>& b, const int nrhs) {
        double rn = 0, bn = 0;
        for(int c = 0; c < nrhs; ++c) {
            for(int i = 0; i < A.n; ++i) {
                double s = 0;
                for(int j = std::max(0, i - A.kl); j <= std::min(A.n - 1, i + A.ku); ++j)
                    s += A.at(i, j) * static_cast<double>(x[static_cast<size_t>(c) * A.n + j]);
                const double r = static_cast<double>(b[static_cast<size_t>(c) * A.n + i]) - s;
                rn += r * r;
                bn += static_cast<double>(b[static_cast<size_t>(c) * A.n + i]) * static_cast<double>(b[static_cast<size_t>(c) * A.n + i]);
            }
        }
        return std::sqrt(rn / std::max(bn, 1e-300));
    }

    template<class T>
    void run_case(const std::string& name, const band& A, const int nrhs, const unsigned seed, const bool reference, const int parts = 0, const double err_tol = -1.0, const int refine = 3) {
        const int n = A.n, lda = A.lda(), ldb = n;
        std::vector<T> h(static_cast<size_t>(lda) * n), b(static_cast<size_t>(ldb) * nrhs);
        for(size_t i = 0; i < h.size(); ++i) h[i] = static_cast<T>(A.a[i]);
        std::mt19937 gen(seed);
        std::uniform_real_distribution u(-1.0, 1.0);
        for(auto& v : b) v = static_cast<T>(u(gen));

        dev<T> da, db;
        da.alloc(h.size());
        db.alloc(b.size());
        cudaMemcpy(da.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(db.p, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice);

        gspike::opts o;
        o.parts = parts;
        o.max_refine = refine;
        gspike::stats st;
        const int info = gspike::spike_gbsv<T>(n, A.kl, A.ku, nrhs, da.p, lda, db.p, ldb, o, &st);

        std::vector<T> x(b.size());
        cudaMemcpy(x.data(), db.p, x.size() * sizeof(T), cudaMemcpyDeviceToHost);

        std::vector<T> back(static_cast<size_t>(lda) * n);
        cudaMemcpy(back.data(), da.p, back.size() * sizeof(T), cudaMemcpyDeviceToHost);
        const bool a_ok = (back == h);

        const double res = residual<T>(A, x, b, nrhs);
        const double eps = (sizeof(T) == 4) ? 1e-5 : 1e-11;
        const double tol = (err_tol > 0) ? err_tol : eps;

        const std::string s = name + " (n=" + std::to_string(n) + " kl=" + std::to_string(A.kl) +
                              " ku=" + std::to_string(A.ku) + " nrhs=" + std::to_string(nrhs) +
                              " parts=" + std::to_string(st.parts) + " refine=" + std::to_string(st.refine) +
                              ")";
        std::printf("%s  info=%d relres=%.2e\n", s.c_str(), info, st.residual);
        check(info == 0, s + ": info");
        check(res < (sizeof(T) == 4 ? 2e-4 : 2e-10), s + ": residual");
        check(a_ok, s + ": A untouched");

        if(reference && n <= 700) {
            std::vector<double> bd(b.size());
            for(size_t i = 0; i < b.size(); ++i) bd[i] = static_cast<double>(b[i]);
            const std::vector<double> xd = dense_solve(A, bd, nrhs);
            double num = 0, den = 0;
            for(size_t i = 0; i < xd.size(); ++i) {
                num = std::max(num, std::abs(static_cast<double>(x[i]) - xd[i]));
                den = std::max(den, std::abs(xd[i]));
            }
            const double err = num / std::max(den, 1e-300);
            std::printf("       error vs dense = %.2e (tol %.1e)\n", err, tol);
            check(err < tol, s + ": matches dense solve");
        }
    }

    template<class T>
    void suite(const char* tag) {
        std::printf("=== %s ===\n", tag);

        { // tridiagonal SPD
            band A = make_random(97, 1, 1, true, 1);
            run_case<T>("tridiagonal", A, 1, 2, true);
        }
        { // small, several RHS, no refinement: the direct factorization must be enough
            band A = make_random(61, 3, 3, true, 3);
            run_case<T>("dd banded", A, 5, 4, true, 0, -1.0, 0);
        }
        { // non symmetric bandwidths, forced SPIKE partitions
            band A = make_random(213, 2, 6, true, 5);
            run_case<T>("kl<ku", A, 3, 6, true, 4);
            band B = make_random(211, 7, 2, true, 7);
            run_case<T>("kl>ku", B, 2, 8, true, 4);
        }
        { // many partitions on a small matrix (forced)
            band A = make_random(400, 4, 4, true, 9);
            run_case<T>("forced 16 parts", A, 4, 10, true, 16);
            run_case<T>("single partition", A, 4, 10, true, 1);
        }
        { // larger, residual check only
            band A = make_random(3000, 6, 6, true, 11);
            run_case<T>("large", A, 2, 12, false);
        }
        { // non diagonally dominant: needs pivoting / refinement.  In float the
          // attainable accuracy is limited by eps * cond(A), so the tolerance is loose.
            band A = make_random(150, 3, 3, false, 13, 0.9);
            run_case<T>("non dominant", A, 2, 14, true, 0, sizeof(T) == 4 ? 5e-3 : 1e-7);
        }
        { // degenerate band shapes
            band D = make_random(37, 0, 0, true, 19);
            run_case<T>("diagonal", D, 2, 20, true);
            band S = make_random(1, 0, 0, true, 21);
            run_case<T>("1x1", S, 1, 22, true);
        }
        { // reuse: factor once, two different RHS
            band A = make_random(500, 5, 4, true, 15);
            const int n = A.n, lda = A.lda(), nrhs = 3;
            std::vector<T> h(static_cast<size_t>(lda) * n);
            for(size_t i = 0; i < h.size(); ++i) h[i] = static_cast<T>(A.a[i]);
            dev<T> da;
            da.alloc(h.size());
            cudaMemcpy(da.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
            gspike::solver<T> s(n, A.kl, A.ku, da.p, lda);
            std::mt19937 gen(16);
            std::uniform_real_distribution u(-1.0, 1.0);
            std::vector<T> b0(static_cast<size_t>(n) * nrhs);
            for(auto& v : b0) v = static_cast<T>(u(gen));
            dev<T> db;
            db.alloc(b0.size());
            bool ok = true;
            for(int rep = 0; rep < 2; ++rep) {
                cudaMemcpy(db.p, b0.data(), b0.size() * sizeof(T), cudaMemcpyHostToDevice);
                const int info = s.solve(nrhs, db.p, n);
                std::vector<T> x(b0.size());
                cudaMemcpy(x.data(), db.p, x.size() * sizeof(T), cudaMemcpyDeviceToHost);
                const double r = residual<T>(A, x, b0, nrhs);
                ok = ok && info == 0 && r < (sizeof(T) == 4 ? 2e-4 : 2e-10);
            }
            check(ok, "solver reuse (parts=" + std::to_string(s.parts()) + ")");
        }
        { // solve_direct: one direct solve with the factorization, no refinement
            band A = make_random(320, 4, 4, true, 23);
            const int n = A.n, lda = A.lda(), nrhs = 3, ldb = n + 1;
            std::vector<T> h(static_cast<size_t>(lda) * n), b0(static_cast<size_t>(n) * nrhs), b(static_cast<size_t>(ldb) * nrhs);
            for(size_t e = 0; e < h.size(); ++e) h[e] = static_cast<T>(A.a[e]);
            std::mt19937 gen(24);
            std::uniform_real_distribution<double> u(-1.0, 1.0);
            for(auto& v : b0) v = static_cast<T>(u(gen));
            const T sentinel = static_cast<T>(-2.5e-3);
            for(auto& v : b) v = sentinel;
            for(int c = 0; c < nrhs; ++c)
                for(int i = 0; i < n; ++i) b[static_cast<size_t>(c) * ldb + i] = b0[static_cast<size_t>(c) * n + i];
            dev<T> da, db;
            da.alloc(h.size());
            db.alloc(b.size());
            cudaMemcpy(da.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
            cudaMemcpy(db.p, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice);
            gspike::solver<T> s(n, A.kl, A.ku, da.p, lda);
            s.solve_direct(nrhs, db.p, ldb);
            std::vector<T> xraw(b.size()), x(static_cast<size_t>(n) * nrhs), back(h.size());
            cudaMemcpy(xraw.data(), db.p, xraw.size() * sizeof(T), cudaMemcpyDeviceToHost);
            cudaMemcpy(back.data(), da.p, back.size() * sizeof(T), cudaMemcpyDeviceToHost);
            for(int c = 0; c < nrhs; ++c)
                for(int i = 0; i < n; ++i) x[static_cast<size_t>(c) * n + i] = xraw[static_cast<size_t>(c) * ldb + i];
            bool pad_ok = true;
            for(int c = 0; c < nrhs; ++c)
                for(int i = n; i < ldb; ++i) pad_ok = pad_ok && xraw[static_cast<size_t>(c) * ldb + i] == sentinel;
            const double res = residual<T>(A, x, b0, nrhs);
            check(s.info() == 0 && res < (sizeof(T) == 4 ? 2e-6 : 1e-12) && pad_ok && back == h, "solve_direct (no refinement, ldb padding)");
        }
        { // invalid arguments
            band A = make_random(10, 1, 1, true, 17);
            std::vector<T> h(static_cast<size_t>(A.lda()) * A.n), b(10);
            for(size_t i = 0; i < h.size(); ++i) h[i] = static_cast<T>(A.a[i]);
            dev<T> da, db;
            da.alloc(h.size());
            db.alloc(b.size());
            cudaMemcpy(da.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
            check(gspike::spike_gbsv<T>(10, 1, 1, 1, da.p, 1, db.p, 10) < 0, "bad lda rejected");
            check(gspike::spike_gbsv<T>(10, 1, 1, 1, da.p, A.lda(), db.p, 5) < 0, "bad ldb rejected");
            check(gspike::spike_gbsv<T>(10, 1, 1, 1, nullptr, A.lda(), db.p, 10) < 0, "null A rejected");
            check(gspike::spike_gbsv<T>(10, 1, 1, 0, da.p, A.lda(), db.p, 10) == 0, "zero right-hand sides accepted");
            if constexpr(!gspike::index_64bit) // n*nrhs cannot be addressed with 32-bit indices
                check(gspike::spike_gbsv<T>(10, 1, 1, std::numeric_limits<gspike::index_t>::max(), da.p, A.lda(), db.p, 10) == -4, "oversized rhs count rejected");
        }
    }

    // -------------------------------------------------------------------------
    // Mixed precision: a double problem is solved with a single-precision
    // factorization and opts::mixed_iters float solves (the first solves, the
    // rest correct).  The reported and recomputed residuals, an untouched A,
    // untouched ldb padding and the accounting are checked here; accuracy
    // comparisons between iteration counts are done by the caller.
    // -------------------------------------------------------------------------
    struct mixed_result {
        int info = 0;
        int parts = 0;
        int iters = 0;
        double relres = 0.0;
        double res = 0.0;
        double err = 0.0;
        bool padded = true;
    };

    mixed_result mixed_case(const std::string& name, const band& A, const int nrhs, const unsigned seed, const int iters, const double tol, const bool dense = true, const int parts = 0) {
        const int n = A.n, lda = A.lda(), ldb = n + 2;
        std::vector<double> h(static_cast<size_t>(lda) * n), b0(static_cast<size_t>(n) * nrhs), b(static_cast<size_t>(ldb) * nrhs);
        for(size_t e = 0; e < h.size(); ++e) h[e] = A.a[e];
        std::mt19937 bg(seed);
        std::uniform_real_distribution<double> bu(-1.0, 1.0);
        for(auto& v : b0) v = bu(bg);
        constexpr double sentinel = -3.14159e-3; // marks the untouched rows of each column
        for(auto& v : b) v = sentinel;
        for(int c = 0; c < nrhs; ++c)
            for(int i = 0; i < n; ++i) b[static_cast<size_t>(c) * ldb + i] = b0[static_cast<size_t>(c) * n + i];

        dev<double> da, db;
        da.alloc(h.size());
        db.alloc(b.size());
        cudaMemcpy(da.p, h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(db.p, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice);

        gspike::opts o;
        o.parts = parts;
        o.mixed_iters = iters;
        gspike::stats st;
        mixed_result R;
        {
            const gspike::mixed_solver s(n, A.kl, A.ku, da.p, lda, o);
            R.info = s.solve(nrhs, db.p, ldb, &st);
            check(s.n() == n && s.kl() == A.kl && s.ku() == A.ku, name + ": accessors");
            check(s.parts() == st.parts, name + ": parts accessor matches stats");
        }
        R.parts = st.parts;
        R.iters = st.refine;
        R.relres = st.residual;

        std::vector<double> xraw(b.size()), back(h.size()), x(static_cast<size_t>(n) * nrhs);
        cudaMemcpy(xraw.data(), db.p, xraw.size() * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(back.data(), da.p, back.size() * sizeof(double), cudaMemcpyDeviceToHost);
        for(int c = 0; c < nrhs; ++c)
            for(int i = 0; i < n; ++i) x[static_cast<size_t>(c) * n + i] = xraw[static_cast<size_t>(c) * ldb + i];

        for(int c = 0; c < nrhs; ++c)
            for(int i = n; i < ldb; ++i) R.padded = R.padded && xraw[static_cast<size_t>(c) * ldb + i] == sentinel;

        R.res = residual<double>(A, x, b0, nrhs);
        if(dense && n <= 700) {
            const std::vector<double> xd = dense_solve(A, b0, nrhs);
            double num = 0, den = 0;
            for(size_t e = 0; e < xd.size(); ++e) {
                num = std::max(num, std::abs(x[e] - xd[e]));
                den = std::max(den, std::abs(xd[e]));
            }
            R.err = num / std::max(den, 1e-300);
        }

        std::printf("%s  info=%d parts=%d iters=%d relres=%.2e (host %.2e)", name.c_str(), R.info, R.parts, R.iters, R.relres, R.res);
        if(dense && n <= 700) std::printf(" err=%.2e", R.err);
        std::printf("\n");

        check(R.info == 0, name + ": info");
        check(R.res < tol, name + ": residual");
        check(back == h, name + ": A untouched");
        check(R.padded, name + ": ldb padding untouched");
        check(R.parts >= 1, name + ": partitions reported");
        check(R.iters >= 1 && R.iters <= iters, name + ": iterations reported");
        check(std::abs(R.relres - R.res) <= 1e-8 * std::max(1.0, R.res), name + ": stats residual matches");
        if(dense && n <= 700) check(R.err < 10 * tol, name + ": matches dense solve");
        return R;
    }

    void mixed_suite() {
        std::printf("=== mixed precision (double problem, float factorization) ===\n");

        { // two float solves reach far beyond single precision on dominant bands
            band A = make_random(500, 5, 5, true, 31);
            const mixed_result one = mixed_case("mixed 1 iter", A, 3, 32, 1, 1e-4);
            const mixed_result two = mixed_case("mixed 2 iters", A, 3, 32, 2, 1e-9);
            check(two.res < one.res * 1e-2, "second float solve improves the residual by 100x");
            const mixed_result four = mixed_case("mixed 4 iters", A, 3, 32, 4, 1e-9);
            check(four.res <= two.res * 10.0, "extra float solves do not degrade the residual");
        }
        { // non-symmetric bandwidth, forced SPIKE partitions, several RHS
            band A = make_random(300, 3, 7, true, 33);
            mixed_case("mixed kl<ku", A, 4, 34, 2, 1e-9, true, 8);
            band B = make_random(301, 7, 3, true, 35);
            mixed_case("mixed kl>ku", B, 2, 36, 2, 1e-9, true, 8);
        }
        { // stagnation: an unreachable target uses the iteration budget and the
          // last correction, which no longer helps, is dropped.  The budget is
          // generous so that stagnation is reached well before it.
            band A = make_random(400, 5, 5, true, 50);
            const int n = A.n, lda = A.lda(), nrhs = 3, ldb = n + 1;
            std::vector<double> h(static_cast<size_t>(lda) * n), b0(static_cast<size_t>(n) * nrhs), b(static_cast<size_t>(ldb) * nrhs, -2.0);
            for(size_t e = 0; e < h.size(); ++e) h[e] = A.a[e];
            std::mt19937 bg(51);
            std::uniform_real_distribution<double> bu(-1.0, 1.0);
            for(auto& v : b0) v = bu(bg);
            for(int c = 0; c < nrhs; ++c)
                for(int i = 0; i < n; ++i) b[static_cast<size_t>(c) * ldb + i] = b0[static_cast<size_t>(c) * n + i];
            dev<double> da, db;
            da.alloc(h.size());
            db.alloc(b.size());
            cudaMemcpy(da.p, h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(db.p, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice);
            gspike::opts o;
            o.mixed_iters = 12;
            o.rtol = 1e-18; // below double precision: never an early exit
            gspike::mixed_solver s(n, A.kl, A.ku, da.p, lda, o);
            gspike::stats st;
            const int info = s.solve(nrhs, db.p, ldb, &st);
            std::vector<double> xraw(b.size()), x(static_cast<size_t>(n) * nrhs);
            cudaMemcpy(xraw.data(), db.p, xraw.size() * sizeof(double), cudaMemcpyDeviceToHost);
            for(int c = 0; c < nrhs; ++c)
                for(int i = 0; i < n; ++i) x[static_cast<size_t>(c) * n + i] = xraw[static_cast<size_t>(c) * ldb + i];
            const double res = residual<double>(A, x, b0, nrhs);
            std::printf("mixed stagnation  info=%d iters=%d relres=%.2e\n", info, st.refine, res);
            check(info == 0 && res < 1e-9 && st.refine >= 1 && st.refine <= 12, "mixed stagnation keeps a bounded iterate");
            check(st.refine < 12, "mixed stagnation drops the last correction");
        }
        { // degenerate band shapes
            band D = make_random(37, 0, 0, true, 37);
            mixed_case("mixed diagonal", D, 2, 38, 2, 1e-12);
            band S = make_random(1, 0, 0, true, 39);
            mixed_case("mixed 1x1", S, 1, 40, 2, 1e-12);
        }
        { // larger problem, residual check only
            band A = make_random(3000, 6, 6, true, 41);
            mixed_case("mixed large", A, 2, 42, 2, 1e-9, false);
        }
        { // non diagonally dominant: the float factorization needs pivoting
            band A = make_random(150, 3, 3, false, 43, 0.9);
            mixed_case("mixed non dominant", A, 2, 44, 4, 1e-8);
        }
        { // one-shot interface
            band A = make_random(200, 4, 4, true, 45);
            const int n = A.n, lda = A.lda(), nrhs = 3, ldb = n;
            std::vector<double> h(static_cast<size_t>(lda) * n), b0(static_cast<size_t>(n) * nrhs);
            for(size_t e = 0; e < h.size(); ++e) h[e] = A.a[e];
            std::mt19937 bg(46);
            std::uniform_real_distribution<double> bu(-1.0, 1.0);
            for(auto& v : b0) v = bu(bg);
            dev<double> da, db;
            da.alloc(h.size());
            db.alloc(b0.size());
            cudaMemcpy(da.p, h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(db.p, b0.data(), b0.size() * sizeof(double), cudaMemcpyHostToDevice);
            gspike::stats st;
            const int info = gspike::spike_gbsv_mixed(n, A.kl, A.ku, nrhs, da.p, lda, db.p, ldb, gspike::opts(), &st);
            std::vector<double> x(b0.size());
            cudaMemcpy(x.data(), db.p, x.size() * sizeof(double), cudaMemcpyDeviceToHost);
            const double res = residual<double>(A, x, b0, nrhs);
            check(info == 0 && res < 1e-9 && st.refine >= 1 && st.refine <= 2, "spike_gbsv_mixed one-shot");
        }
        { // the factorization is reused, and the scratch space grows with nrhs
            band A = make_random(400, 5, 4, true, 47);
            const int n = A.n, lda = A.lda();
            std::vector<double> h(static_cast<size_t>(lda) * n);
            for(size_t e = 0; e < h.size(); ++e) h[e] = A.a[e];
            dev<double> da;
            da.alloc(h.size());
            cudaMemcpy(da.p, h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice);
            gspike::mixed_solver s(n, A.kl, A.ku, da.p, lda);
            std::mt19937 bg(48);
            std::uniform_real_distribution<double> bu(-1.0, 1.0);
            bool ok = true;
            for(const int nrhs : {3, 5}) {
                const int ldb = n + 1;
                std::vector<double> b0(static_cast<size_t>(n) * nrhs);
                for(auto& v : b0) v = bu(bg);
                std::vector<double> b(static_cast<size_t>(ldb) * nrhs, -1.0);
                for(int c = 0; c < nrhs; ++c)
                    for(int i = 0; i < n; ++i) b[static_cast<size_t>(c) * ldb + i] = b0[static_cast<size_t>(c) * n + i];
                dev<double> db;
                db.alloc(b.size());
                cudaMemcpy(db.p, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice);
                gspike::stats st;
                const int info = s.solve(nrhs, db.p, ldb, &st);
                std::vector<double> xraw(b.size()), x(static_cast<size_t>(n) * nrhs);
                cudaMemcpy(xraw.data(), db.p, xraw.size() * sizeof(double), cudaMemcpyDeviceToHost);
                for(int c = 0; c < nrhs; ++c)
                    for(int i = 0; i < n; ++i) x[static_cast<size_t>(c) * n + i] = xraw[static_cast<size_t>(c) * ldb + i];
                const double r = residual<double>(A, x, b0, nrhs);
                ok = ok && info == 0 && r < 1e-9 && st.refine >= 1 && st.refine <= 2;
                for(int c = 0; c < nrhs; ++c) ok = ok && xraw[static_cast<size_t>(c) * ldb + n] == -1.0;
            }
            check(ok, "mixed_solver reuse (parts=" + std::to_string(s.parts()) + ")");
        }
        { // invalid arguments
            band A = make_random(10, 1, 1, true, 49);
            std::vector<double> h(static_cast<size_t>(A.lda()) * A.n), b(10);
            for(size_t e = 0; e < h.size(); ++e) h[e] = A.a[e];
            dev<double> da, db;
            da.alloc(h.size());
            db.alloc(b.size());
            cudaMemcpy(da.p, h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice);
            check(gspike::spike_gbsv_mixed(10, 1, 1, 1, nullptr, A.lda(), db.p, 10) == -1, "mixed: null A rejected");
            check(gspike::spike_gbsv_mixed(10, 1, 1, 1, da.p, A.lda(), nullptr, 10) == -2, "mixed: null B rejected");
            check(gspike::spike_gbsv_mixed(10, 1, 1, 1, da.p, A.lda(), db.p, 5) == -3, "mixed: bad ldb rejected");
            check(gspike::spike_gbsv_mixed(10, 1, 1, -1, da.p, A.lda(), db.p, 10) == -1, "mixed: negative nrhs rejected");
            check(gspike::spike_gbsv_mixed(10, 1, 1, 0, da.p, A.lda(), db.p, 10) == 0, "mixed: zero rhs accepted");
            check(gspike::spike_gbsv_mixed(10, 1, 1, 1, da.p, 1, db.p, 10) == -1, "mixed: bad lda rejected");
            gspike::opts no_iter;
            no_iter.mixed_iters = 0; // clamped: at least the first float solve runs
            check(gspike::spike_gbsv_mixed(10, 1, 1, 1, da.p, A.lda(), db.p, 10, no_iter) == 0, "mixed: mixed_iters=0 still solves");
        }
    }

    // -------------------------------------------------------------------------
    // Fuzzy testing.
    //
    // Random, well-posed problems: every dimension, bandwidth, partition count,
    // right-hand side count, leading dimension and refinement setting is drawn
    // from a seeded generator.  Matrices are strictly diagonally dominant and
    // optionally column scaled (still nonsingular, but no longer necessarily
    // dominant).  Each solve is checked for info == 0, the relative residual,
    // an untouched A, untouched ldb padding, and -- for small systems -- the
    // dense host reference.  Every so often the same factorization is also
    // checked for batch/single-solve consistency.
    // -------------------------------------------------------------------------
    struct fuzz_rng {
        std::mt19937 g;
        explicit fuzz_rng(const unsigned seed)
            : g(seed) {}
        int i(const int lo, const int hi) { return std::uniform_int_distribution<int>(lo, hi)(g); }
        double u(const double lo, const double hi) { return std::uniform_real_distribution<double>(lo, hi)(g); }
        unsigned next() { return g(); }
    };

    template<class T>
    void fuzzy_suite(const char* tag, const unsigned seed, const int iters) {
        std::printf("=== %s (fuzzy: %d random problems, seed %u) ===\n", tag, iters, seed);
        constexpr bool single = sizeof(T) == 4;
        fuzz_rng r(seed);

        for(int it = 0; it < iters; ++it) {
            const bool big = (it % 17 == 8);
            const int n = big ? r.i(700, 1500) : r.i(1, 300);
            int kl = r.i(0, 8);
            int ku = r.i(0, 8);
            if(it % 11 == 3) { // wide band relative to the matrix size
                const int wide = std::min(n - 1, 40);
                kl = r.i(0, wide);
                ku = r.i(0, wide);
            }
            if(it % 23 == 4) kl = 0; // purely upper banded
            if(it % 29 == 6) ku = 0; // purely lower banded
            if(it % 31 == 7) {       // symmetric bandwidth
                kl = std::min(n - 1, 12);
                ku = kl;
            }

            const double decay = (it % 4 == 0) ? 0.5 : (it % 4 == 1) ? 0.8 :
                                                                       1.0;
            band A = make_random(n, kl, ku, true, r.next(), decay);

            const bool scaled = (it % 3 == 0);
            if(scaled) { // column scaling keeps A nonsingular but may break dominance
                for(int j = 0; j < n; ++j) {
                    const double d = r.u(0.4, 1.2);
                    for(int i = std::max(0, j - ku); i <= std::min(n - 1, j + kl); ++i) A.at(i, j) *= d;
                }
            }

            const int nrhs = r.i(1, big ? 3 : 6);
            const int ldb = n + ((it % 5 == 0) ? r.i(1, 4) : 0); // padded leading dimension
            const int parts = (it % 7 == 0) ? 0 : ((it % 5 == 0) ? 1 : r.i(2, 12));
            const int refine = (it % 9 == 0) ? 0 : r.i(1, 3);
            const int lda = A.lda();

            std::vector<T> h(static_cast<size_t>(lda) * n), b0(static_cast<size_t>(n) * nrhs), b(static_cast<size_t>(ldb) * nrhs);
            for(size_t e = 0; e < h.size(); ++e) h[e] = static_cast<T>(A.a[e]);

            std::mt19937 bg(r.next());
            std::uniform_real_distribution<double> bu(-1.0, 1.0);
            for(auto& v : b0) v = static_cast<T>(bu(bg));
            const T sentinel = static_cast<T>(-3.14159e-3); // marks the untouched part of each column
            for(auto& v : b) v = sentinel;
            for(int c = 0; c < nrhs; ++c)
                for(int i = 0; i < n; ++i) b[static_cast<size_t>(c) * ldb + i] = b0[static_cast<size_t>(c) * n + i];

            dev<T> da, db;
            da.alloc(h.size());
            db.alloc(b.size());
            cudaMemcpy(da.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
            cudaMemcpy(db.p, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice);

            gspike::opts o;
            o.parts = parts;
            o.max_refine = refine;
            gspike::stats st;
            const int info = gspike::spike_gbsv<T>(n, kl, ku, nrhs, da.p, lda, db.p, ldb, o, &st);

            std::vector<T> xraw(b.size()), back(h.size()), x(static_cast<size_t>(n) * nrhs);
            cudaMemcpy(xraw.data(), db.p, xraw.size() * sizeof(T), cudaMemcpyDeviceToHost);
            cudaMemcpy(back.data(), da.p, back.size() * sizeof(T), cudaMemcpyDeviceToHost);
            for(int c = 0; c < nrhs; ++c)
                for(int i = 0; i < n; ++i) x[static_cast<size_t>(c) * n + i] = xraw[static_cast<size_t>(c) * ldb + i];

            bool pad_ok = true;
            for(int c = 0; c < nrhs; ++c)
                for(int i = n; i < ldb; ++i) pad_ok = pad_ok && xraw[static_cast<size_t>(c) * ldb + i] == sentinel;

            const double res = residual<T>(A, x, b0, nrhs);
            const double res_tol = single ? 2e-4 : 2e-10;
            const bool a_ok = (back == h);
            bool ok = info == 0 && res < res_tol && a_ok && pad_ok;
            ok = ok && st.parts >= 1 && st.refine >= 0 && st.refine <= refine;

            double err = 0.0;
            bool dense_checked = false;
            if(refine > 0 && n <= 256) {
                std::vector<double> bd(b0.size());
                for(size_t e = 0; e < b0.size(); ++e) bd[e] = static_cast<double>(b0[e]);
                const std::vector<double> xd = dense_solve(A, bd, nrhs);
                double num = 0, den = 0;
                for(size_t e = 0; e < xd.size(); ++e) {
                    num = std::max(num, std::abs(static_cast<double>(x[e]) - xd[e]));
                    den = std::max(den, std::abs(xd[e]));
                }
                err = num / std::max(den, 1e-300);
                const double dtol = single ? (scaled ? 1e-3 : 2e-4) : (scaled ? 1e-8 : 1e-10);
                ok = ok && err < dtol;
                dense_checked = true;
            }

            std::printf("  [fuzzy %02d] n=%-5d kl=%-2d ku=%-2d nrhs=%d ldb=%-4d parts=%-2d refine=%d%s relres=%.2e", it, n, kl, ku, nrhs, ldb, st.parts, st.refine, scaled ? " scaled" : "", res);
            if(dense_checked) std::printf(" err=%.2e", err);
            std::printf("\n");
            if(!ok) std::printf("       info=%d pad_ok=%d A_untouched=%d\n", info, static_cast<int>(pad_ok), static_cast<int>(a_ok));
            check(ok, "fuzzy case " + std::to_string(it) + " (seed " + std::to_string(seed) + ")");

            // Batch and column-by-column solves of the same factorization must
            // agree, and reusing the solver with a smaller nrhs must reset the
            // arena bookkeeping correctly.
            if(it % 9 == 4 && n <= 200 && nrhs >= 2) {
                gspike::opts o2;
                o2.parts = parts;
                o2.max_refine = 0; // same deterministic direct path for both variants
                gspike::solver<T> s(n, kl, ku, da.p, lda, o2);

                std::vector<T> bp(b.size()), xb(static_cast<size_t>(n) * nrhs), xs(static_cast<size_t>(n) * nrhs), tmp(b.size());
                for(auto& v : bp) v = sentinel;
                for(int c = 0; c < nrhs; ++c)
                    for(int i = 0; i < n; ++i) bp[static_cast<size_t>(c) * ldb + i] = b0[static_cast<size_t>(c) * n + i];

                cudaMemcpy(db.p, bp.data(), bp.size() * sizeof(T), cudaMemcpyHostToDevice);
                const int ib = s.solve(nrhs, db.p, ldb);
                cudaMemcpy(tmp.data(), db.p, tmp.size() * sizeof(T), cudaMemcpyDeviceToHost);
                for(int c = 0; c < nrhs; ++c)
                    for(int i = 0; i < n; ++i) xb[static_cast<size_t>(c) * n + i] = tmp[static_cast<size_t>(c) * ldb + i];

                bool singles_ok = true;
                for(int c = 0; c < nrhs; ++c) {
                    std::vector<T> col(static_cast<size_t>(ldb), sentinel);
                    for(int i = 0; i < n; ++i) col[i] = b0[static_cast<size_t>(c) * n + i];
                    cudaMemcpy(db.p, col.data(), col.size() * sizeof(T), cudaMemcpyHostToDevice);
                    const int ic = s.solve(1, db.p, ldb);
                    singles_ok = singles_ok && ic == 0;
                    cudaMemcpy(col.data(), db.p, col.size() * sizeof(T), cudaMemcpyDeviceToHost);
                    for(int i = 0; i < n; ++i) xs[static_cast<size_t>(c) * n + i] = col[i];
                }

                double num = 0, den = 0;
                for(size_t e = 0; e < xb.size(); ++e) {
                    num = std::max(num, std::abs(static_cast<double>(xb[e]) - xs[e]));
                    den = std::max(den, std::abs(static_cast<double>(xb[e])));
                }
                const double diff = num / std::max(den, 1e-300);
                check(ib == 0 && singles_ok && diff < (single ? 1e-5 : 1e-12), "fuzzy batch/single consistency (case " + std::to_string(it) + ")");
            }
        }
    }

    // Random mixed-precision problems: same generators as the fixed suite but
    // drawn from a seed.  Two float solves are requested; the solver may use
    // fewer if the first correction cannot improve the residual.
    void mixed_fuzzy(const unsigned seed, const int iters) {
        std::printf("=== mixed precision (fuzzy: %d random problems, seed %u) ===\n", iters, seed);
        fuzz_rng r(seed);
        for(int it = 0; it < iters; ++it) {
            const int n = r.i(1, 400);
            int kl = r.i(0, 6), ku = r.i(0, 6);
            if(it % 13 == 5) kl = 0;
            if(it % 17 == 9) ku = 0;
            if(it % 19 == 11) { // symmetric, somewhat wider band
                kl = std::min(n - 1, 10);
                ku = kl;
            }
            const double decay = (it % 3 == 0) ? 0.5 : 1.0;
            band A = make_random(n, kl, ku, true, r.next(), decay);
            const int nrhs = r.i(1, 4);
            const int parts = (it % 6 == 0) ? 0 : r.i(1, 8);
            mixed_case("mixed fuzzy " + std::to_string(it), A, nrhs, r.next(), 2, 1e-9, n <= 256, parts);
        }
    }

} // namespace

int main(const int argc, char** argv) {
    unsigned seed = 20261118u;
    int fuzz_iters = 256;
    for(int i = 1; i < argc; ++i) {
        if(std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            seed = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 0));
        else if(std::strcmp(argv[i], "--fuzzy-iters") == 0 && i + 1 < argc)
            fuzz_iters = std::atoi(argv[++i]);
        else {
            std::printf("usage: %s [--seed N] [--fuzzy-iters N]\n", argv[0]);
            return 2;
        }
    }

    std::printf("=== index type ===\n");
    std::printf("  gspike::index_t = %d-bit (configured %d-bit)\n", static_cast<int>(8 * sizeof(gspike::index_t)), gspike::index_64bit ? 64 : 32);
    check(sizeof(gspike::index_t) == (gspike::index_64bit ? 8u : 4u), "index_t matches the configured width");

    suite<float>("float");
    suite<double>("double");
    mixed_suite();
    if(fuzz_iters > 0) {
        fuzzy_suite<float>("float", seed, fuzz_iters);
        fuzzy_suite<double>("double", seed ^ 0x9e3779b9u, fuzz_iters);
        mixed_fuzzy(seed ^ 0x517cc1b7u, std::max(1, fuzz_iters / 4));
    }
    std::printf("\n%s (%d failure%s)\n", g_fail == 0 ? "ALL TESTS PASSED" : "FAILURES", g_fail, g_fail == 1 ? "" : "s");
    return g_fail == 0 ? 0 : 1;
}
