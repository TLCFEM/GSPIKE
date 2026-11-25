<div align="center">

# ⚡ GSPIKE

**A mixed-precision GPU direct solver for banded linear systems.**
**Factor once. Solve fast. Keep the residual.**

[![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-76B900?logo=nvidia&logoColor=white&style=flat-square)](https://developer.nvidia.com/cuda-zone)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white&style=flat-square)](https://isocpp.org/)
[![CI](https://img.shields.io/badge/CI-simulated%20T4-blue?style=flat-square)](.github/workflows/ci.yml)
[![Validated on](https://img.shields.io/badge/validated%20on-NVIDIA%20L40-green?style=flat-square)](https://www.nvidia.com/en-us/data-center/l40/)
![Dependencies](https://img.shields.io/badge/link%20deps-CUDA%20runtime%20only-success?style=flat-square)

</div>

---

GSPIKE is a small C++20 CUDA library for solving `A · X = B` when `A` is banded.
It is a GPU port of the Fortran [`Xspike_smp`](ref/) solver: the matrix is split
into independent diagonal blocks, factored in parallel, and the coupling between
blocks is solved through a tiny interface system. Support for `float` and `double`
is built in, including a mixed-precision path that factorizes in `float` and
recovers near-`double` accuracy with a `double` residual.

## ✨ Why GSPIKE?

| | What you get |
|---|---|
| 🚀 **Throughput** | Blocked band LU and a blocked triangular solve that reuses band factors across 64 right-hand sides at once. |
| 🎯 **Mixed precision that works** | Float factorization + double residual + one correction step: ~`1e-13` residual at a fraction of the fp64 cost. |
| 🧠 **Memory-aware** | Full spike vectors are kept automatically when they fit (≤ 512 MB and 3× free memory), removing a second band solve from every solve. |
| 🛡️ **Robust** | Partial pivoting with a scalar fallback; non-dominant and asymmetric bands are covered by the test suite. |
| 🧩 **Small surface** | One header, two scalar types, no dependency beyond the CUDA runtime. `A` is never modified. |
| 🔬 **Observable** | `--profile` reports factorization and solve phase timings; the benchmark reports GFLOP/s and retained device memory. |

## 🚀 30-second start

```cpp
#include <gspike/spike.h>

// dA, dB live in device memory; A is read-only, B is overwritten with X.
gspike::opts o;             // defaults are fine
gspike::stats st;
int info = gspike::spike_gbsv<double>(n, kl, ku, nrhs, dA, lda, dB, ldb, o, &st);
// info == 0 means success
```

Solve many right-hand sides from one factorization:

```cpp
gspike::solver<double> s(n, kl, ku, dA, lda);
s.solve(nrhs, dB, ldb, &st);          // call repeatedly
```

Or use the mixed-precision path for a double matrix:

```cpp
gspike::mixed_solver s(n, kl, ku, dA, lda);
s.solve(nrhs, dB, ldb, &st);          // two float solves + one double residual
```

**Storage.** `A(i,j)` lives at `A[lda * j + ku + i - j]` with `lda >= kl + ku + 1`,
for `i - j` in `[-ku, kl]` — exactly LAPACK `dgbmv` / `dgbtrf` band storage.
Column-major, device pointers.

### Pick your solver

| API | Factorization | Correction | Best for |
|---|---|---|---|
| `spike_gbsv<T>` | once, internally | — | one-off solves |
| `solver<T>` | once, explicitly | optional full-spike matvec | many RHS against the same matrix |
| `mixed_solver` | `float` | `double` residual + float solve | double problems at near-float cost |

## 🧭 How it works

```mermaid
flowchart LR
    A["Banded A<br/>+ few RHS"] --> B["Split into p partitions"]
    B --> C["Blocked band LU<br/>4-column panels"]
    C --> D["Blocked triangular solve<br/>2·klu spike columns"]
    D --> E["Reduced interface system<br/>2·klu·(p−1) unknowns"]
    E --> F["Interface solve<br/>+ full-spike correction"]
    F --> G["Solution X"]
```

Single-level SPIKE-DS with `klu = max(kl, ku)` interface unknowns per side:

1. Split the matrix into `p` contiguous diagonal blocks and factor each block in
   parallel with a blocked band LU with partial pivoting (one CUDA block per
   panel, tiled rank-4 trailing updates). `A` is untouched; factors live in a
   private band buffer in LAPACK's `2*kl + ku + 1` layout.
2. Solve each block against its coupling columns to obtain the spikes
   `Vp_i = A_ii⁻¹ C_i` and `Vm_i = A_ii⁻¹ B_i` (`2*klu` columns in total).
3. Eliminate the interiors. What remains is a `2*klu*(p-1)` block-tridiagonal
   interface system, assembled in band storage with half-bandwidth `3*klu` and
   factored once with the same band LU kernel.
4. Back-substitute the interface unknowns, correct the coupling rows of each
   block, then solve each block.

The partition count is chosen from a size-based rule (`n / (16*klu + 32)`, good
for narrow bands) balanced against the growing interface LU (`~p*klu³`). Narrow
bands get up to 32 partitions; bands around 1% of `n` get about 4–6. Every
partition keeps at least `2*klu + 1` rows and the interface band stays under
320 MB. `p == 1` degenerates to a plain parallel band LU.

## 📊 Performance at a glance

Factorization + solve for one right-hand side on an **NVIDIA L40**, CUDA 12.6,
diagonally dominant random bands. MKL uses 8 threads. Speedups are CPU/GPU time.

| n | klu | GSPIKE mixed | GSPIKE double | MKL double band | double speedup | mixed speedup |
|---:|----:|-------------:|--------------:|----------------:|---------------:|--------------:|
| 10000 | 100 | 17.5 ms | 25.5 ms | 34.6 ms | **1.36×** | **1.98×** |
| 25000 | 250 | 65.5 ms | 107.4 ms | 127.3 ms | **1.19×** | **1.94×** |
| 50000 | 500 | 206.9 ms | 426.5 ms | 529.0 ms | **1.24×** | **2.56×** |
| 80000 | 800 | 570.2 ms | 1436.1 ms | 1532.8 ms | **1.07×** | **2.69×** |

> **Well-conditioned + very few RHS?** A band-aware BiCGStab converges in 5–6
> iterations and is far faster than any factorization. GSPIKE is the tool when
> you need robust pivoting, many RHS from one factorization, or a tight residual.

<details>
<summary><b>Full sweep: all precisions, timings, throughput, memory, residuals</b></summary>

Example on an NVIDIA L40 (CUDA 12.6), diagonally dominant random bands:

| type   |     n | klu | nrhs | parts | factor ms | solve ms | total ms | factor GFLOP/s | solve GFLOP/s | mem MB | rel. residual |
|--------|------:|----:|-----:|------:|----------:|---------:|---------:|---------------:|--------------:|-------:|--------------:|
| double | 10000 | 100 |    1 |     5 |     24.0  |     1.8  |     25.8  |          70.9  |          7.2  |     48  |      7.8e-16 |
| double | 25000 | 250 |    1 |     4 |     99.4  |     9.3  |    108.6  |         258.4  |          8.4  |    278  |      1.2e-15 |
| double | 50000 | 500 |    2 |     4 |    406.7  |    23.7  |    430.4  |         504.7  |         26.1  |   1110  |      1.7e-15 |
| double | 80000 | 800 |    1 |     4 |   1360.5  |    78.2  |   1438.7  |         617.9  |         10.1  |   1852  |      2.1e-15 |
| double | 25000 | 250 |    1 |     1 |    149.6  |    25.6  |    175.2  |          41.4  |          1.5  |    146  |      1.2e-15 |
| float  | 10000 | 100 |    1 |     5 |     11.3  |     1.4  |     12.7  |         150.0  |          9.3  |     24  |      3.5e-07 |
| float  | 25000 | 250 |    1 |     4 |     48.5  |     7.3  |     55.8  |         528.9  |         10.6  |    140  |      5.4e-07 |
| float  | 50000 | 500 |    2 |     4 |    170.3  |    17.8  |    188.1  |        1205.5  |         34.6  |    556  |      7.6e-07 |
| float  | 80000 | 800 |    1 |     4 |    485.1  |    38.5  |    523.6  |        1733.1  |         20.5  |   1416  |      9.6e-07 |
| float  | 25000 | 250 |    1 |     1 |     93.9  |    12.5  |    106.4  |          65.9  |          3.0  |     72  |      5.5e-07 |
| mixed  | 10000 | 100 |    1 |     5 |     11.4  |     2.8  |     14.2  |         149.7  |         10.5  |     32  |      1.4e-13 |
| mixed  | 25000 | 250 |    1 |     4 |     48.1  |    15.0  |     63.1  |         533.6  |         12.0  |    188  |      3.3e-13 |
| mixed  | 50000 | 500 |    2 |     4 |    170.1  |    37.6  |    207.7  |        1207.1  |         38.2  |    750  |      6.4e-13 |
| mixed  | 80000 | 800 |    1 |     4 |    491.3  |    81.9  |    573.1  |        1711.3  |         22.4  |   1908  |      1.0e-12 |

`parts = 1` is the plain parallel band LU path (SPIKE disabled). `mem MB` is
the device memory retained by the solver plus solve scratch, excluding the
input matrix. Full spikes are enabled automatically for the float/mixed rows
and for double up to 50k/500; at 80k/800 in double the 512 MB auto cap excludes
it.

</details>

<details>
<summary><b>What the automatic full-spike correction buys</b></summary>

With `--no-full-spikes`, the second band solve is used instead of the saved
spike matvec (same factorizations):

| n | klu | nrhs | parts | mixed solve, auto | mixed solve, no full | mem MB, auto / no |
|--:|----:|-----:|------:|------------------:|---------------------:|------------------:|
| 10000 | 100 | 1 | 5 | 2.8 ms | 4.6 ms | 32 / 24 |
| 25000 | 250 | 1 | 4 | 15.0 ms | 18.4 ms | 188 / 140 |
| 50000 | 500 | 2 | 4 | 37.6 ms | 61.3 ms | 750 / 558 |
| 80000 | 800 | 1 | 4 | 81.9 ms | 132.7 ms | 1908 / 1418 |

</details>

<details>
<summary><b>Original Fortran SPIKE-SMP reference (CPU double vs GPU mixed)</b></summary>

The reference runs with 8 OpenMP threads on a 24-core AMD EPYC 9334. `factor x`,
`solve x`, `total x` are CPU/GPU time; values above 1 favour the GPU.

| n | klu | nrhs | CPU parts | GPU parts | CPU factor ms | GPU factor ms | factor x | CPU solve ms | GPU solve ms | solve x | CPU total ms | GPU total ms | total x |
|--:|----:|-----:|----------:|----------:|--------------:|--------------:|---------:|-------------:|-------------:|--------:|-------------:|-------------:|--------:|
| 10000 | 100 | 1 | 8 | 5 | 10.1 | 11.3 | 0.89 | 0.59 | 2.80 | 0.21 | 10.6 | 14.1 | 0.75 |
| 25000 | 250 | 1 | 8 | 4 | 57.3 | 48.5 | 1.18 | 4.34 | 15.04 | 0.29 | 61.6 | 63.5 | 0.97 |
| 50000 | 500 | 2 | 8 | 4 | 338.0 | 170.2 | 1.99 | 21.71 | 37.61 | 0.58 | 359.7 | 207.8 | 1.73 |
| 80000 | 800 | 1 | 8 | 4 | 1219.9 | 489.7 | 2.49 | 44.75 | 81.87 | 0.55 | 1264.7 | 571.6 | 2.21 |

Bands are around 1% of `n`, so factorization dominates. The GPU triangular
solve is slower for a single RHS because substitution is serial, but it is a
small part of the total: GPU mixed wins overall from `n = 50000`.

The original Fortran solver takes its panel width from `ILAENV`, factorizes a
panel with BLAS2, then applies it with `DTRSM`/`DGEMM`; the partition loop and
recursive reduced system use OpenMP. CMake links the best LAPACK/BLAS it finds —
**with Intel MKL the same factorization is ~0.34 s instead of ~3.9 s with netlib
BLAS** at `n = 50000, klu = 500`. Eight OpenMP threads beat sixteen: each worker
gets two balanced partitions and the last reduced-system levels suffer less
from idle workers.

Against MKL's own general band LU (`dgbtrf`/`dgbtrs`, 8 threads):

| n | klu | Fortran SPIKE factor | MKL `dgbtrf` | speedup |
|--:|----:|---------------------:|-------------:|--------:|
| 10000 | 100 | 10.2 ms | 34.0 ms | 3.3× |
| 25000 | 250 | 57.7 ms | 117.2 ms | 2.0× |
| 50000 | 500 | 336.1 ms | 495.1 ms | 1.5× |
| 80000 | 800 | 1218.8 ms | 1456.5 ms | 1.2× |

Both use MKL BLAS underneath; the gain comes from SPIKE's partitioning.

</details>

<details>
<summary><b>MKL and sparse GPU head-to-head (all six methods)</b></summary>

The optional `spike_duel` benchmark (built when Intel MKL, cuSPARSE and cuBLAS
are available) solves the same diagonally dominant band matrix with six
methods. Times are setup/factorization plus solve for one RHS; MKL uses 8 threads.

| n | klu | gspike mixed | gspike double | MKL double band | MKL float band | cuSPARSE CSR BiCGStab | band BiCGStab |
|--:|----:|-------------:|--------------:|----------------:|---------------:|----------------------:|--------------:|
| 10000 | 100 | 17.5 ms | 25.5 ms | 34.6 ms | 18.9 ms | 7.3 ms | 1.1 ms |
| 25000 | 250 | 65.5 ms | 107.4 ms | 127.3 ms | 80.8 ms | 8.9 ms | 2.7 ms |
| 50000 | 500 | 206.9 ms | 426.5 ms | 529.0 ms | 266.5 ms | 19.8 ms | 8.9 ms |
| 80000 | 800 | 570.2 ms | 1436.1 ms | 1532.8 ms | 731.1 ms | 45.6 ms | 22.7 ms |

`band BiCGStab` uses the same iteration as the cuSPARSE variant but reads
contiguous band segments directly — one warp per column, no column indices and
no CSR build. It converges in 5–6 iterations on these well-conditioned bands,
beating generic CSR by 2–6× and every direct solver by 10–60×. On non-dominant
bands (`n = 25000, klu = 250`) both BiCGStab variants stagnate while the direct
solvers finish in 0.1–0.4 s.

</details>

## 🔬 Inside the fast path

- **Blocked band LU.** A 4-column panel is factored by one block, then a tiled
  rank-4 update applies it to the trailing band with 4×4 register micro-tiles.
  A scalar kernel remains for matrices that actually pivot.
- **Blocked spike solve.** Spike generation was the dominant factorization cost.
  It now solves 32-row panels sequentially (one warp per RHS), then applies the
  panel to 64 columns with 64×64 tiles built from 4×4 register micro-tiles.
  Band factors are loaded
  once per 64 columns instead of once per column: the spike phase at
  `n = 50000, klu = 500` fell from **187 ms to 42.6 ms** in fp32, taking
  factorization from ~312 ms to ~165 ms. C-spike columns have zero RHS above
  the last `klu` rows, so their forward sweep starts there.
- **Memory-aware solves.** `opts::full_spikes` stores the complete spike
  solutions and replaces the second band solve with a matvec (1.2–1.8× faster
  solves here). It is enabled automatically when it costs ≤ 512 MB and ≥ 3×
  that much device memory is free.
- **Phase visibility.** `--profile` at `n = 50000, klu = 500, nrhs = 2, parts = 4`
  (mixed) reports, in ms: pack 9.7, partition LU 80.4, reciprocal diagonal 0.1,
  spikes 42.6, reduced LU 32.3; a solve takes 12.5 (first band solve), 0.04
  (interface RHS), 4.5 (reduced solve), 0.8 (full-spike correction). The reduced
  system is still a single CUDA block — the next scaling target.

## 🛠 Build

Requires CUDA (tested with 12.6) and CMake ≥ 3.24 (for
`CMAKE_CUDA_ARCHITECTURES=native`); C++20 is used. The benchmark builds by
default. The optional CPU reference (Fortran + LAPACK + OpenMP) and the
MKL/cuSPARSE head-to-head are enabled automatically when their dependencies are
found, and skipped otherwise. Neither is ever linked into the library.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

```sh
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug -j
ctest --test-dir build-debug --output-on-failure
```

Debug builds have debug symbols but **no AddressSanitizer** (its flags were
removed because CUDA's UVA needs the shadow gap ASan protects). If you enable
ASan externally, use `ASAN_OPTIONS=protect_shadow_gap=0`; device errors need
`compute-sanitizer`.

The top-level `CMakeLists.txt` only builds the library; tests live in
`tests/CMakeLists.txt` and benchmarks in `bench/CMakeLists.txt`, and neither is
added when gspike is a subdirectory of another project. The library target is
`gspike` (alias `gspike::gspike`); the single public header is
`include/gspike/spike.h` and the only link dependency is the CUDA runtime. Only
`float` and `double` are instantiated — other scalar types fail at link time
rather than silently doing something unexpected.

### Use as a dependency

```cmake
add_subdirectory(path/to/GSPIKE)
target_link_libraries(my_app PRIVATE gspike::gspike)
```

There is no installed package config: consumers use the source tree. As a
subproject, GSPIKE does not change your C++ standard, build type or CUDA
architecture, and builds no tests or benchmarks. Pass `-DGSPIKE_BENCH=OFF` to a
standalone configure if you only want the library.

## 🧪 Tests & benchmark

`spike_tests` checks both precisions against a dense host LU reference and the
relative residual: tridiagonal, diagonally dominant, asymmetric and one-sided
bands, wide bands, forced/automatic partitions, multiple RHS, non-dominant
matrices with pivoting (including full spikes), warp solves with partially
filled terminal blocks, solver reuse, `opts::profile`, argument validation and a
randomized fuzz pass. The mixed solver is checked against the dense reference
and a shape/partition sweep.

`spike_bench` reports factorization and solve time separately, their sum,
retained device memory and effective GFLOP/s from analytic operation counts.
`factor` covers construction; `solve` is averaged over five solves. `--profile`
adds opt-in phase timings to stderr (it adds synchronizations, so it is not used
for the headline numbers).

```sh
./build/spike_bench                        # default sweep, double
./build/spike_bench --type float           # default sweep, float
./build/spike_bench --type mixed           # default sweep, mixed
./build/spike_bench --full-spikes --type mixed    # force full-spike solves
./build/spike_bench --no-full-spikes --type mixed # force two-solve correction
./build/spike_bench --profile --type mixed        # per-phase GPU timings
./build/spike_bench --type cpu             # original Fortran solver
./build/spike_bench --compare              # CPU double vs GPU mixed sweep
./build/spike_bench 200000 50 4 32         # n klu(=kl=ku) nrhs parts(0=auto)
./build/spike_bench --compare 50000 500 2  # one CPU/GPU comparison
./build/spike_bench --cpu-threads 8 ...    # reference threads (rounded to 2^k)
```

Default sweeps use bands around 1% of `n` (`klu = n/100`) and few RHS; the
largest case is `n = 80000, klu = 800`. `--type cpu` and `--compare` require the
CPU reference (built when Fortran, LAPACK and OpenMP are available). The
reference uses OpenMP with power-of-two partitions, so
thread counts are rounded down (`--cpu-threads 24` uses 16) and pinned to the
allowed CPUs.

## 📌 Good to know

- `opts::full_spikes` costs `2*klu*n` scalars (~1 GB in double at `n = 80000,
  klu = 800`). It is auto-enabled only within the 512 MB / 3×-free-memory
  budget; `--no-full-spikes` disables the automatic choice. Accuracy is unchanged.
- `opts::profile` prints phase timings to stderr and adds event synchronizations.
- `info > 0` is the first zero (perturbed) pivot: values ≤ `n` are rows of `A`,
  larger ones belong to the reduced interface system. Check the residual.
- `solver` and `mixed_solver` keep the pointer to `A`, so `A` must outlive them;
  `mixed_solver` reads the original double matrix on every solve for the
  residual. `solve()` uses private scratch and is not thread-safe on one object;
  separate objects are independent.
- CUDA launch failures throw `std::runtime_error`; asynchronous kernel errors
  surface at the next synchronization. Bad arguments return negative `info`
  (or throw from the `solver` constructor).

## 🗂 Repository layout

```
include/gspike/spike.h   public API (no CUDA types in the header)
src/common.cuh           shared host helpers (error checks, device buffers)
src/kernels.cuh          device kernels
src/spike.cu             host side: partitioning, factorization, solve
src/mixed.cu             double/float mixed-precision solver
tests/test_spike.cu      correctness tests
tests/CMakeLists.txt     test target registration
bench/bench.cu           benchmark tool
bench/duel.cu            optional MKL/cuSPARSE head-to-head
bench/CMakeLists.txt     benchmark build (auto-detected comparison targets)
ref/                     original Fortran SPIKE-SMP solver
ref/spike_ref.f90        C-callable wrapper used by the benchmark
```

---

<div align="center">

Benchmarks on **NVIDIA L40 · CUDA 12.6**, with CI on a simulated **T4**.
Built for FEM-style banded systems where factorization dominates and right-hand
sides are few.

</div>
