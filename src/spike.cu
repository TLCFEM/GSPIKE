// SPIKE (DS) banded solver -- CUDA implementation.
//
// Single level SPIKE factorization:
//   1. split the band matrix into p contiguous diagonal blocks, each factored
//      with a band LU with partial pivoting (parallel over partitions),
//   2. form the interface spikes  Vp_i = A_ii^-1 C_i ,  Vm_i = A_ii^-1 B_i,
//   3. eliminate the interior and solve the 2*klu*(p-1) interface system
//      (block tridiagonal, factored once with the same band LU),
//   4. retrieve each block solution with a correction on the coupling rows.
#include "common.cuh"
#include "gspike/spike.h"
#include "kernels.cuh"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace gspike {
    namespace {

        constexpr int kMaxParts = 32;
        constexpr int kInfoNone = 0x7f7f7f7f;
        // A 320 MB cap on the interface band buffer leaves p=4 available for
        // fp64 at n=80000, klu=800 (276 MB) without threatening memory safety.
        constexpr int kReducedMemCapMB = 320;

        // Optional phase timings.  Disabled by default: when enabled it adds
        // one event synchronization per phase, so it must not be used for the
        // headline benchmark numbers.
        class phase_timer {
        public:
            explicit phase_timer(const bool on)
                : on_(on) {
                if(on_) {
                    detail::check(cudaEventCreate(&start_));
                    detail::check(cudaEventCreate(&stop_));
                    detail::check(cudaEventRecord(start_));
                }
            }
            ~phase_timer() {
                if(on_) {
                    cudaEventDestroy(start_);
                    cudaEventDestroy(stop_);
                }
            }
            phase_timer(const phase_timer&) = delete;
            phase_timer& operator=(const phase_timer&) = delete;

            void lap(const char* name) const {
                if(!on_) return;
                detail::check(cudaEventRecord(stop_));
                detail::check(cudaEventSynchronize(stop_));
                float ms = 0;
                detail::check(cudaEventElapsedTime(&ms, start_, stop_));
                std::fprintf(stderr, "[gspike:profile] %-16s %8.3f ms\n", name, ms);
                detail::check(cudaEventRecord(start_));
            }

        private:
            bool on_;
            cudaEvent_t start_{}, stop_{};
        };

    } // namespace

    ////////////////////////////////////////////////////////////////////////////////

    template<class T>
    struct solver<T>::impl {
        int n = 0, kl = 0, ku = 0, klu = 0, p = 1, lda_f = 0, block = detail::kBlock;
        opts o;
        const T* A = nullptr;
        int lda = 0;
        int info0 = 0;

        detail::device_buffer<dk::pdesc<T>> d_pd;
        detail::device_buffer<dk::pdesc<T>> d_rp;
        detail::device_buffer<T*> d_spp;
        detail::device_buffer<T> d_ab;
        detail::device_buffer<int> d_piv;
        detail::device_buffer<T> d_sp;
        detail::device_buffer<T> d_bc;
        detail::device_buffer<T> d_cc;
        detail::device_buffer<T> d_fs;    // full spikes, only when opts::full_spikes
        detail::device_buffer<T> d_diag;  // 1/U(ii), indexed by global row
        detail::device_buffer<T> d_diagR; // 1/U(ii) of the reduced system
        detail::device_buffer<T> d_R;
        detail::device_buffer<int> d_pivR;
        detail::device_buffer<int> d_info;
        detail::device_buffer<int> d_pivoted;
        int m = 0, klr = 0, ldaR = 0;

        detail::device_buffer<T> arena;

        void factor();

        // Small bands are cheaper with one thread per right-hand side.  Wide
        // bands use one warp per right-hand side when there are enough columns
        // to fill the GPU.  With only a few columns (the reduced interface
        // system, or a solve with few right-hand sides) the sequential
        // substitution chain dominates, so each column gets a whole block.
        void launch_band_solve(const dk::pdesc<T>* pd, int npd, int lda_, int kl_, int ku_, const T* diag, T* rhs, int ldr, int ncols) const {
            if(kl_ + ku_ >= 16) {
                const long long chains = (long long)npd * ncols;
                if(chains < 512) { // too few columns for the warp-per-column path
                    const int bs = std::min(1024, ((kl_ + ku_ + 32) / 32) * 32);
                    dk::band_solve_block<T><<<dim3(npd, ncols), bs>>>(pd, lda_, kl_, ku_, diag, rhs, ldr);
                }
                else {
                    // Give wide bands more lanes per substitution chain; the
                    // per-column dependency then hides its L2 latency better.
                    const int W = (kl_ + ku_ >= 750) ? 4 : ((kl_ + ku_ >= 400) ? 2 : 1);
                    const int cols = 32 / W;
                    const dim3 grid(npd, (ncols + cols - 1) / cols);
                    if(W == 1) dk::band_solve_warp<T, 1><<<grid, 1024>>>(pd, lda_, kl_, ku_, diag, rhs, ldr, ncols);
                    if(W == 2) dk::band_solve_warp<T, 2><<<grid, 1024>>>(pd, lda_, kl_, ku_, diag, rhs, ldr, ncols);
                    if(W == 4) dk::band_solve_warp<T, 4><<<grid, 1024>>>(pd, lda_, kl_, ku_, diag, rhs, ldr, ncols);
                }
            }
            else {
                const dim3 grid(npd, (ncols + block - 1) / block);
                dk::band_solve<T><<<grid, block>>>(pd, lda_, kl_, ku_, diag, rhs, ldr, ncols);
            }
            detail::check(cudaGetLastError());
        }

        // Blocked band LU (see kernels.cuh).  Four-column panels are factored
        // by one block per partition; their tiled trailing updates fill the GPU.
        // Used for both the partition factors and the reduced system.
        template<int NB = 4>
        void launch_band_lu(const dk::pdesc<T>* pd, int npd, int size, int lda_, int kl_, int ku_, int* info) const {
            const int tiles_i = std::max(1, (kl_ + 63) / 64);
            const int tiles_k = std::max(1, (kl_ + ku_ + 63) / 64);
            detail::check(cudaMemset(d_pivoted.get(), 0, sizeof(int)));
            for(int j0 = 0; j0 < size; j0 += NB) {
                const int nb = std::min(NB, size - j0);
                dk::band_lu_panel<T, NB><<<npd, block>>>(pd, lda_, kl_, ku_, j0, nb, info, d_pivoted.get());
                if(j0 + NB < size)
                    dk::band_lu_update<T, NB><<<dim3(tiles_k, tiles_i, npd), 256>>>(pd, lda_, kl_, ku_, j0, nb);
            }
            detail::check(cudaGetLastError());
        }

        // Blocked forward/backward triangular solves for spike generation.
        // B rows are solved sequentially by one warp per RHS, then the panel
        // is applied to C columns with a register micro-tile.  Only for the
        // unpivoted, many-RHS case; everything else keeps the scalar path.
        void launch_band_solve_blocked(T* rhs, int ldr, int ncols, int maxsize, int ncC = 0) const {
            // B=32 keeps shared memory at 32 KB in fp64; C/R are the largest
            // tiles that still fit 48 KB and fill the GPU with 256 threads.
            constexpr int B = 32, C = 64, R = 64;
            const dim3 gpanel(p, (ncols + 31) / 32);
            // kl may be zero (pure upper band); still launch one row tile so
            // the kernel can exit cleanly instead of receiving a zero grid.
            const dim3 gfwd((ncols + C - 1) / C, std::max(1, (kl + R - 1) / R), p);
            const dim3 gbwd((ncols + C - 1) / C, (kl + ku + R - 1) / R, p);
            for(int j0 = 0; j0 < maxsize; j0 += B) {
                dk::band_solve_fwd_panel<T, B><<<gpanel, 1024>>>(d_pd.get(), lda_f, kl, ku, j0, rhs, ldr, ncols, ncC);
                dk::band_solve_fwd_update<T, B, C, R><<<gfwd, (R / 4) * (C / 4)>>>(d_pd.get(), lda_f, kl, ku, j0, rhs, ldr, ncols, ncC);
            }
            const int npanels = (maxsize + B - 1) / B;
            for(int pi = npanels - 1; pi >= 0; --pi) {
                const int i0 = pi * B;
                dk::band_solve_bwd_panel<T, B><<<gpanel, 1024>>>(d_pd.get(), lda_f, kl, ku, i0, d_diag.get(), rhs, ldr, ncols);
                dk::band_solve_bwd_update<T, B, C, R><<<gbwd, (R / 4) * (C / 4)>>>(d_pd.get(), lda_f, kl, ku, i0, rhs, ldr, ncols);
            }
            detail::check(cudaGetLastError());
        }

        bool band_lu_pivoted() const {
            int v = 0;
            detail::check(cudaMemcpy(&v, d_pivoted.get(), sizeof(int), cudaMemcpyDeviceToHost));
            return v != 0;
        }

        void alloc_arena(int nrhs) {
            const auto need = (size_t)((long long)n * nrhs + (long long)2 * klu * p * nrhs + (long long)m * nrhs);
            if(need > arena.size()) arena.alloc(need);
        }

        // X (n x nrhs, leading dimension ldx) holds f on entry and the solution on exit.
        void solve_inplace(T* X, int ldx, T* g, T* grj, T* rs, int nrhs, const phase_timer& pt) const {
            const dim3 gpar(p, (nrhs + block - 1) / block);
            const dim3 gred(p > 1 ? p - 1 : 1, (nrhs + block - 1) / block);

            if(p == 1) {
                launch_band_solve(d_pd.get(), p, lda_f, kl, ku, d_diag.get(), X, ldx, nrhs);
                return;
            }
            dk::copy_mat<T><<<nrhs, block>>>(n, nrhs, X, ldx, g, n);
            launch_band_solve(d_pd.get(), p, lda_f, kl, ku, d_diag.get(), g, n, nrhs);
            pt.lap("solve:first_solve");
            dk::extract_corners<T><<<gpar, block>>>(d_pd.get(), g, n, klu, nrhs, grj);
            dk::reduced_rhs<T><<<gred, block>>>(grj, m, klu, rs, nrhs);
            pt.lap("solve:interface_rhs");
            launch_band_solve(d_rp.get(), 1, ldaR, klr, klr, d_diagR.get(), rs, m, nrhs);
            pt.lap("solve:reduced_solve");
            if(d_fs) {
                dk::full_correct<T><<<dim3(p, nrhs), block, 2 * klu * sizeof(T)>>>(d_pd.get(), d_fs.get(), rs, g, klu, m, p, n, X, ldx, nrhs);
                pt.lap("solve:full_correct");
            }
            else {
                dk::correct_rhs<T><<<dim3(p, nrhs), block>>>(d_pd.get(), d_bc.get(), d_cc.get(), rs, klu, m, p, X, ldx, nrhs);
                launch_band_solve(d_pd.get(), p, lda_f, kl, ku, d_diag.get(), X, ldx, nrhs);
                pt.lap("solve:corr_and_solve");
            }
        }
    };

    ////////////////////////////////////////////////////////////////////////////////

    template<class T>
    void solver<T>::impl::factor() {
        impl* q = this;
        const int n = q->n, kl = q->kl, ku = q->ku, klu = q->klu, p = q->p, lda_f = q->lda_f;
        const int block = q->block;
        const phase_timer pt(q->o.profile);

        std::vector<int> row0(p), size(p);
        const int base = n / p, rem = n % p;
        for(int i = 0; i < p; ++i) {
            size[i] = base + (i < rem ? 1 : 0);
            row0[i] = (i == 0) ? 0 : row0[i - 1] + size[i - 1];
        }

        q->d_ab.alloc((size_t)lda_f * n);
        q->d_piv.alloc((size_t)n);
        q->d_pd.alloc((size_t)p);
        q->d_sp.alloc((size_t)p * 4 * klu * klu);
        q->d_bc.alloc((size_t)p * klu * klu);
        q->d_cc.alloc((size_t)p * klu * klu);
        q->d_spp.alloc((size_t)p);
        q->d_info.alloc(1);
        q->d_pivoted.alloc(1);
        detail::check(cudaMemset(q->d_info.get(), 0x7f, sizeof(int)));
        detail::check(cudaMemset(q->d_sp.get(), 0, (size_t)p * 4 * klu * klu * sizeof(T)));

        std::vector<dk::pdesc<T>> hd(p);
        std::vector<T*> hspp(p);
        for(int i = 0; i < p; ++i) {
            hd[i].ab = q->d_ab.get() + (long long)lda_f * row0[i];
            hd[i].piv = q->d_piv.get() + row0[i];
            hd[i].row0 = row0[i];
            hd[i].size = size[i];
            hspp[i] = q->d_sp.get() + (long long)i * 4 * klu * klu;
        }
        detail::check(cudaMemcpy(q->d_pd.get(), hd.data(), p * sizeof(dk::pdesc<T>), cudaMemcpyHostToDevice));
        detail::check(cudaMemcpy(q->d_spp.get(), hspp.data(), p * sizeof(T*), cudaMemcpyHostToDevice));

        const int mb_cols = 8, mb_grid = (*std::max_element(size.begin(), size.end()) + mb_cols - 1) / mb_cols;
        const auto make_blocks = [&] { dk::make_blocks<T><<<dim3(p, mb_grid), mb_cols * 32>>>(q->d_pd.get(), q->A, n, q->lda, lda_f, ku, kl, klu, q->d_bc.get(), q->d_cc.get()); };
        make_blocks();
        detail::check(cudaGetLastError());
        pt.lap("factor:pack");
        q->launch_band_lu(q->d_pd.get(), p, *std::max_element(size.begin(), size.end()), lda_f, kl, ku, q->d_info.get());
        pt.lap("factor:band_lu");
        if(q->band_lu_pivoted()) { // blocked updates assume an unpivoted band
            make_blocks();
            detail::check(cudaMemset(q->d_info.get(), 0x7f, sizeof(int)));
            dk::band_lu_scalar<T><<<p, block>>>(q->d_pd.get(), lda_f, kl, ku, q->d_info.get());
            detail::check(cudaGetLastError());
        }

        q->d_diag.alloc((size_t)n);
        dk::invert_diag<T><<<p, block>>>(q->d_pd.get(), lda_f, kl + ku, q->d_diag.get());
        pt.lap("factor:invert_diag");

        if(p == 1) return;

        // spikes: A_ii [C_i | B_i], solved in chunks of columns (2*klu in total).
        // All spike columns are solved at once (one thread each) as long as the
        // temporary buffer stays within budget.
        const int ncol = 2 * klu;
        int chunk = ncol;
        while(chunk > 1 && (long long)n * chunk * (long long)sizeof(T) > (512LL << 20)) chunk /= 2;
        detail::device_buffer<T> rhs;
        rhs.alloc((size_t)n * chunk);
        const int nb = (2 * klu * klu + block - 1) / block;
        if(q->o.full_spikes) q->d_fs.alloc((size_t)n * ncol);
        // The blocked solve needs an unpivoted LU and enough RHS to amortize
        // the extra launches; narrow bands stay on the scalar path.
        const bool blocked = ncol >= 128 && kl + ku >= 64 && !q->band_lu_pivoted();
        for(int c0 = 0; c0 < ncol; c0 += chunk) {
            const int nc = std::min(chunk, ncol - c0);
            dk::spike_rhs<T><<<dim3(p, nc), block>>>(q->d_pd.get(), q->d_bc.get(), q->d_cc.get(), klu, rhs.get(), n, c0, ncol);
            if(blocked) q->launch_band_solve_blocked(rhs.get(), n, nc, *std::max_element(size.begin(), size.end()), std::min(nc, std::max(0, klu - c0)));
            else q->launch_band_solve(q->d_pd.get(), p, lda_f, kl, ku, q->d_diag.get(), rhs.get(), n, nc);
            dk::spike_extract<T><<<dim3(p, nb), block>>>(q->d_pd.get(), rhs.get(), n, klu, c0, nc, q->d_sp.get());
            if(q->d_fs) dk::copy_mat<T><<<nc, block>>>(n, nc, rhs.get(), n, q->d_fs.get() + (long long)c0 * n, n);
            detail::check(cudaGetLastError());
        }
        pt.lap("factor:spikes");
    }

    ////////////////////////////////////////////////////////////////////////////////

    template<class T>
    solver<T>::solver(int n, int kl, int ku, const T* A, int lda, const opts& o)
        : p_(std::make_unique<impl>()) {
        impl* q = p_.get();
        q->o = o;
        if(o.block >= 32 && o.block <= 1024) q->block = o.block;
        if(n < 1 || kl < 0 || ku < 0 || !A) throw std::invalid_argument("gspike: bad dimensions");
        if(lda < kl + ku + 1) throw std::invalid_argument("gspike: lda < kl+ku+1");

        q->n = n;
        q->kl = kl;
        q->ku = ku;
        q->klu = std::max(kl, ku);
        q->lda_f = 2 * kl + ku + 1;
        q->A = A;
        q->lda = lda;

        const int pmax = (q->klu == 0) ? 1 : std::max(1, n / (2 * q->klu + 1));
        int p = o.parts;
        if(p <= 0) {
            // Two bounds on the partition count.  The first is the older
            // size-based rule, which works well for narrow bands.  The second
            // balances the per-partition cost, falling as 1/p, against the
            // reduced interface LU, whose work grows with p*klu^3: the fixed
            // per-column cost of the band LU is worth ~2000 of the klu^2
            // update.  The minimum keeps narrow bands fully partitioned while
            // stopping wide bands (~1% of n) from building a huge interface.
            const double k = q->klu;
            const double size_rule = (double)n / (16.0 * k + 32.0);
            // sqrt(2) accounts for the tiled update: it now uses the band's
            // full kl+ku fill bandwidth rather than just ku per column.
            const double bal_rule = 1.3 * std::sqrt((double)n * (2000.0 + k * k) / (10.0 * k * k * k));
            p = std::min({(int)std::max(1.0, std::round(size_rule)), (int)std::lround(bal_rule), kMaxParts});
        }
        p = std::min(std::max(p, 1), pmax);
        while(p > 1) { // keep the interface system within a memory budget
            const long long m = 2LL * q->klu * (p - 1);
            const long long ldaR = 9LL * q->klu + 1;
            if(m * ldaR * (long long)sizeof(T) < ((long long)kReducedMemCapMB << 20)) break;
            --p;
        }
        q->p = p;
        q->m = 2 * q->klu * (p - 1);
        q->klr = 3 * q->klu;
        q->ldaR = 3 * q->klr + 1;

        // With few right-hand sides the full-spike correction removes a second
        // band solve and is almost free; enable it when the extra vectors fit
        // comfortably in device memory.  Explicit opts::full_spikes always wins.
        if(p > 1 && !o.full_spikes && o.auto_full_spikes) {
            const size_t fs_bytes = (size_t)2 * q->klu * n * sizeof(T);
            size_t free_mem = 0, total_mem = 0;
            if(cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess &&
               fs_bytes <= (512ULL << 20) && free_mem >= 3 * fs_bytes)
                q->o.full_spikes = true;
        }

        q->factor();
        detail::check(cudaDeviceSynchronize());

        int info = 0;
        detail::check(cudaMemcpy(&info, q->d_info.get(), sizeof(int), cudaMemcpyDeviceToHost));
        q->info0 = (info == kInfoNone) ? 0 : info;

        if(p > 1) {
            const phase_timer rpt(q->o.profile);
            q->d_R.alloc((size_t)q->ldaR * q->m);
            q->d_pivR.alloc((size_t)q->m);
            q->d_rp.alloc(1);
            detail::check(cudaMemset(q->d_R.get(), 0, (size_t)q->ldaR * q->m * sizeof(T)));

            dk::reduced_diag<T><<<p - 1, q->block>>>(q->d_spp.get(), p, q->klu, q->klr, q->ldaR, q->d_R.get());
            detail::check(cudaGetLastError());

            dk::pdesc<T> rp{};
            rp.ab = q->d_R.get();
            rp.piv = q->d_pivR.get();
            rp.row0 = 0;
            rp.size = q->m;
            detail::check(cudaMemcpy(q->d_rp.get(), &rp, sizeof(rp), cudaMemcpyHostToDevice));
            detail::check(cudaMemset(q->d_info.get(), 0x7f, sizeof(int)));
            q->launch_band_lu(q->d_rp.get(), 1, q->m, q->ldaR, q->klr, q->klr, q->d_info.get());
            if(q->band_lu_pivoted()) {
                detail::check(cudaMemset(q->d_R.get(), 0, (size_t)q->ldaR * q->m * sizeof(T)));
                dk::reduced_diag<T><<<p - 1, q->block>>>(q->d_spp.get(), p, q->klu, q->klr, q->ldaR, q->d_R.get());
                detail::check(cudaMemset(q->d_info.get(), 0x7f, sizeof(int)));
                dk::band_lu_scalar<T><<<1, q->block>>>(q->d_rp.get(), q->ldaR, q->klr, q->klr, q->d_info.get());
            }
            q->d_diagR.alloc((size_t)q->m);
            dk::invert_diag<T><<<1, q->block>>>(q->d_rp.get(), q->ldaR, 2 * q->klr, q->d_diagR.get());
            detail::check(cudaGetLastError());
            rpt.lap("factor:reduced_lu");
            detail::check(cudaDeviceSynchronize());
            detail::check(cudaMemcpy(&info, q->d_info.get(), sizeof(int), cudaMemcpyDeviceToHost));
            if(info != kInfoNone) {
                // row index in the reduced system; report it offset past the matrix
                if(q->info0 == 0) q->info0 = n + info;
            }
        }

        if(o.verbose) {
            std::printf("[gspike] n=%d kl=%d ku=%d parts=%d lda_f=%d reduced_dim=%d\n", n, kl, ku, p, q->lda_f, q->m);
            if(q->info0) std::printf("[gspike] zero pivot at %d (perturbed)\n", q->info0);
        }
    }

    template<class T>
    solver<T>::~solver() = default;

    template<class T>
    int solver<T>::solve(int nrhs, T* B, int ldb, stats* st) const {
        impl* q = p_.get();
        if(nrhs < 0) return -1;
        if(!B) return -2;
        if(ldb < q->n) return -3;
        if(st) st->parts = q->p;
        if(nrhs == 0) return 0;
        if((long long)q->n * nrhs > INT_MAX) return -4; // buffers are counted in int

        q->alloc_arena(nrhs);
        const long long tot = (long long)q->n * nrhs;
        T* g = q->arena.get();
        T* grj = g + tot;
        T* rs = grj + (long long)2 * q->klu * q->p * nrhs;

        const phase_timer pt(q->o.profile);
        q->solve_inplace(B, ldb, g, grj, rs, nrhs, pt);
        detail::check(cudaGetLastError());

        if(q->o.verbose) std::printf("[gspike] solve nrhs=%d parts=%d\n", nrhs, q->p);
        return q->info0;
    }

    template<class T>
    int solver<T>::n() const { return p_->n; }
    template<class T>
    int solver<T>::kl() const { return p_->kl; }
    template<class T>
    int solver<T>::ku() const { return p_->ku; }
    template<class T>
    int solver<T>::parts() const { return p_->p; }

    template<class T>
    int spike_gbsv(int n, int kl, int ku, int nrhs, const T* A, int lda, T* B, int ldb, const opts& o, stats* st) {
        // Reject bad solve arguments before paying for a factorization, with
        // the same negative codes as solver<T>::solve().
        if(nrhs < 0) return -1;
        if(!B) return -2;
        if(ldb < n) return -3;
        if((long long)n * nrhs > INT_MAX) return -4;
        try {
            solver<T> s(n, kl, ku, A, lda, o);
            return s.solve(nrhs, B, ldb, st);
        }
        catch(const std::invalid_argument&) {
            return -1;
        }
    }

    template class solver<float>;
    template class solver<double>;
    template int spike_gbsv<float>(int, int, int, int, const float*, int, float*, int, const opts&, stats*);
    template int spike_gbsv<double>(int, int, int, int, const double*, int, double*, int, const opts&, stats*);

} // namespace gspike
