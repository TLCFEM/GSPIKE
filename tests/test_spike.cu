// Correctness tests for gspike::spike_gbsv.
//
// Every solve is checked against a dense host reference (small problems) and
// against the relative residual ||b - A x|| / ||b||.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <gspike/spike.h>
#include <random>
#include <string>
#include <vector>

namespace {

    int g_fail = 0;

    void check(bool ok, const std::string& what) {
        std::printf("  [%s] %s\n", ok ? " ok " : "FAIL", what.c_str());
        if(!ok) ++g_fail;
    }

    // ---- host band storage: A(i,j) at ab[j*lda + ku + i-j], lda = kl+ku+1 ----
    struct band {
        int n = 0, kl = 0, ku = 0;
        std::vector<double> a;
        int lda() const { return kl + ku + 1; }
        double& at(int i, int j) { return a[(long long)j * lda() + ku + i - j]; }
        double at(int i, int j) const { return a[(long long)j * lda() + ku + i - j]; }
    };

    band make_random(int n, int kl, int ku, bool dominant, unsigned seed, double decay = 1.0) {
        band m;
        m.n = n;
        m.kl = kl;
        m.ku = ku;
        m.a.assign((size_t)m.lda() * n, 0.0);
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
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
    std::vector<double> dense_solve(const band& A, const std::vector<double>& b, int nrhs) {
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

    template<class T>
    struct dev {
        T* p = nullptr;
        ~dev() {
            if(p) cudaFree(p);
        }
        void alloc(size_t n) { cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)); }
    };

    template<class T>
    double residual(const band& A, const std::vector<T>& x, const std::vector<T>& b, int nrhs) {
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
    void run_case(const std::string& name, const band& A, int nrhs, unsigned seed, bool reference, int parts = 0, double err_tol = -1.0) {
        const int n = A.n, lda = A.lda(), ldb = n;
        std::vector<T> h((size_t)lda * n), b((size_t)ldb * nrhs);
        for(size_t i = 0; i < h.size(); ++i) h[i] = (T)A.a[i];
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        for(auto& v : b) v = (T)u(gen);

        dev<T> da, db;
        da.alloc(h.size());
        db.alloc(b.size());
        cudaMemcpy(da.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(db.p, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice);

        gspike::opts o;
        o.parts = parts;
        gspike::stats st;
        const int info = gspike::spike_gbsv<T>(n, A.kl, A.ku, nrhs, da.p, lda, db.p, ldb, o, &st);

        std::vector<T> x(b.size());
        cudaMemcpy(x.data(), db.p, x.size() * sizeof(T), cudaMemcpyDeviceToHost);

        std::vector<T> back((size_t)lda * n);
        cudaMemcpy(back.data(), da.p, back.size() * sizeof(T), cudaMemcpyDeviceToHost);
        const bool a_ok = (back == h);

        const double res = residual<T>(A, x, b, nrhs);
        const double eps = (sizeof(T) == 4) ? 1e-5 : 1e-11;
        const double tol = (err_tol > 0) ? err_tol : eps;

        std::string s = name + " (n=" + std::to_string(n) + " kl=" + std::to_string(A.kl) +
                        " ku=" + std::to_string(A.ku) + " nrhs=" + std::to_string(nrhs) +
                        " parts=" + std::to_string(st.parts) +
                        ")";
        std::printf("%s  info=%d relres=%.2e\n", s.c_str(), info, res);
        check(info == 0, s + ": info");
        check(res < (sizeof(T) == 4 ? 2e-4 : 2e-10), s + ": residual");
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
        { // small, several RHS: the direct factorization must be enough
            band A = make_random(61, 3, 3, true, 3);
            run_case<T>("dd banded", A, 5, 4, true);
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
        { // non diagonally dominant: needs pivoting.  In float the attainable
          // accuracy is limited by eps * cond(A), so the tolerance is loose.
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
            std::vector<T> h((size_t)lda * n);
            for(size_t i = 0; i < h.size(); ++i) h[i] = (T)A.a[i];
            dev<T> da;
            da.alloc(h.size());
            cudaMemcpy(da.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
            gspike::solver<T> s(n, A.kl, A.ku, da.p, lda);
            std::mt19937 gen(16);
            std::uniform_real_distribution<double> u(-1.0, 1.0);
            std::vector<T> b0((size_t)n * nrhs);
            for(auto& v : b0) v = (T)u(gen);
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
        { // randomized fuzz: shapes, bandwidths, right-hand sides and partitions
            std::mt19937 gen(23);
            std::uniform_int_distribution<int> n_dist(2, 260), band_dist(0, 7), rhs_dist(1, 5), part_dist(0, 8);
            for(int t = 0; t < 12; ++t) {
                const int n = n_dist(gen), kl = band_dist(gen), ku = band_dist(gen);
                const bool dominant = (t % 4) != 3;
                const int parts = std::min(part_dist(gen), std::max(1, n / std::max(1, 2 * std::max(kl, ku) + 1)));
                band A = make_random(n, kl, ku, dominant, 100 + t, dominant ? 1.0 : 0.92);
                const double tol = dominant ? -1.0 : (sizeof(T) == 4 ? 5e-3 : 1e-7);
                run_case<T>("fuzz " + std::to_string(t), A, rhs_dist(gen), 200 + t, true, parts, tol);
            }
        }
        { // invalid arguments
            band A = make_random(10, 1, 1, true, 17);
            std::vector<T> h((size_t)A.lda() * A.n), b(10);
            for(size_t i = 0; i < h.size(); ++i) h[i] = (T)A.a[i];
            dev<T> da, db;
            da.alloc(h.size());
            db.alloc(b.size());
            cudaMemcpy(da.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
            check(gspike::spike_gbsv<T>(10, 1, 1, 1, da.p, 1, db.p, 10) < 0, "bad lda rejected");
            check(gspike::spike_gbsv<T>(10, 1, 1, 1, da.p, A.lda(), db.p, 5) < 0, "bad ldb rejected");
            check(gspike::spike_gbsv<T>(10, 1, 1, 1, nullptr, A.lda(), db.p, 10) < 0, "null A rejected");
            check(gspike::spike_gbsv<T>(10, 1, 1, 0, da.p, A.lda(), db.p, 10) == 0, "zero right-hand sides accepted");
        }
    }

} // namespace

int main() {
    suite<float>("float");
    suite<double>("double");
    std::printf("\n%s (%d failure%s)\n", g_fail == 0 ? "ALL TESTS PASSED" : "FAILURES", g_fail, g_fail == 1 ? "" : "s");
    return g_fail == 0 ? 0 : 1;
}
