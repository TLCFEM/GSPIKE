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

Factorization + solve for one right-hand side (two at `n = 50000`) on an
**NVIDIA L40**, CUDA 12.6, diagonally dominant random bands. Both CPU solvers use
8 threads. Times are total ms (factorization + solve); speedups are MKL time /
GSPIKE time.

| n | klu | nrhs | GSPIKE mixed | GSPIKE double | MKL double band | Fortran SPIKE | mixed vs MKL | double vs MKL |
|---:|----:|-----:|-------------:|--------------:|----------------:|--------------:|-------------:|--------------:|
| 10000 | 100 | 1 | 14.2 ms | 25.4 ms | 26.4 ms | 8.5 ms | **1.86×** | **1.04×** |
| 25000 | 250 | 1 | 63.3 ms | 107.8 ms | 116.9 ms | 70.4 ms | **1.85×** | **1.08×** |
| 50000 | 500 | 2 | 209.0 ms | 432.5 ms | 562.1 ms | 391.6 ms | **2.69×** | **1.30×** |
| 80000 | 800 | 1 | 571.6 ms | 1439.2 ms | 1551.1 ms | 1376.6 ms | **2.71×** | **1.08×** |

All four contenders are direct solvers: a SPIKE factorization (GPU mixed, GPU
double, original Fortran CPU) or LAPACK-style general band LU (MKL). The
Fortran SPIKE solver is the fastest CPU option here, while the GPU mixed path
trades a single-precision factorization for a double residual correction and
stays ahead of MKL's double band LU in every case. No iterative methods are
part of the comparison.

<details>
<summary><b>Full sweep: all precisions, timings, throughput, memory, residuals</b></summary>

Example on an NVIDIA L40 (CUDA 12.6), diagonally dominant random bands:

| type   |     n | klu | nrhs | parts | factor ms | solve ms | total ms | factor GFLOP/s | solve GFLOP/s | mem MB | rel. residual |
|--------|------:|----:|-----:|------:|----------:|---------:|---------:|---------------:|--------------:|-------:|--------------:|
| double | 10000 | 100 |    1 |     5 |     24.1  |     1.8  |     25.8  |          70.7  |          7.2  |     48  |      7.8e-16 |
| double | 25000 | 250 |    1 |     4 |     99.4  |     9.3  |    108.7  |         258.2  |          8.4  |    278  |      1.2e-15 |
| double | 50000 | 500 |    2 |     4 |    407.5  |    23.7  |    431.2  |         503.7  |         26.1  |   1110  |      1.7e-15 |
| double | 80000 | 800 |    1 |     4 |   1358.2  |    78.2  |   1436.4  |         619.0  |         10.1  |   1852  |      2.1e-15 |
| double | 25000 | 250 |    1 |     1 |    150.2  |    25.6  |    175.7  |          41.2  |          1.5  |    146  |      1.2e-15 |
| float  | 10000 | 100 |    1 |     5 |     11.4  |     1.4  |     12.8  |         149.1  |          9.3  |     24  |      3.5e-07 |
| float  | 25000 | 250 |    1 |     4 |     48.7  |     7.3  |     56.0  |         527.1  |         10.6  |    140  |      5.4e-07 |
| float  | 50000 | 500 |    2 |     4 |    168.6  |    17.8  |    186.5  |        1217.2  |         34.6  |    556  |      7.6e-07 |
| float  | 80000 | 800 |    1 |     4 |    486.5  |    38.5  |    525.0  |        1728.0  |         20.5  |   1416  |      9.6e-07 |
| float  | 25000 | 250 |    1 |     1 |     94.6  |    12.5  |    107.2  |          65.4  |          3.0  |     72  |      5.5e-07 |
| mixed  | 10000 | 100 |    1 |     5 |     11.3  |     2.8  |     14.1  |         150.2  |         10.5  |     32  |      1.4e-13 |
| mixed  | 25000 | 250 |    1 |     4 |     48.4  |    15.0  |     63.4  |         530.6  |         11.9  |    188  |      3.3e-13 |
| mixed  | 50000 | 500 |    2 |     4 |    170.1  |    37.6  |    207.7  |        1206.9  |         38.2  |    750  |      6.4e-13 |
| mixed  | 80000 | 800 |    1 |     4 |    488.7  |    81.9  |    570.6  |        1720.3  |         22.4  |   1908  |      1.0e-12 |

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
| 80000 | 800 | 1 | 4 | 81.9 ms | 132.6 ms | 1908 / 1418 |

</details>

<details>
<summary><b>Original Fortran SPIKE-SMP reference (CPU double vs GPU mixed)</b></summary>

The reference runs with 8 OpenMP threads on a 24-core AMD EPYC 9334. `factor x`,
`solve x`, `total x` are CPU/GPU time; values above 1 favour the GPU.

| n | klu | nrhs | CPU parts | GPU parts | CPU factor ms | GPU factor ms | factor x | CPU solve ms | GPU solve ms | solve x | CPU total ms | GPU total ms | total x |
|--:|----:|-----:|----------:|----------:|--------------:|--------------:|---------:|-------------:|-------------:|--------:|-------------:|-------------:|--------:|
| 10000 | 100 | 1 | 8 | 5 | 9.9 | 11.4 | 0.87 | 0.44 | 2.80 | 0.16 | 10.4 | 14.2 | 0.73 |
| 25000 | 250 | 1 | 8 | 4 | 64.7 | 48.1 | 1.35 | 4.53 | 14.97 | 0.30 | 69.2 | 63.0 | 1.10 |
| 50000 | 500 | 2 | 8 | 4 | 366.2 | 170.2 | 2.15 | 24.22 | 37.60 | 0.64 | 390.4 | 207.8 | 1.88 |
| 80000 | 800 | 1 | 8 | 4 | 1319.0 | 491.5 | 2.68 | 52.47 | 81.86 | 0.64 | 1371.4 | 573.4 | 2.39 |

Bands are around 1% of `n`, so factorization dominates. The GPU triangular
solve is slower for a single RHS because substitution is serial, but it is a
small part of the total: GPU mixed wins overall from `n = 25000`.

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
| 10000 | 100 | 7.9 ms | 25.4 ms | 3.2× |
| 25000 | 250 | 65.6 ms | 102.5 ms | 1.6× |
| 50000 | 500 | 368.4 ms | 488.4 ms | 1.3× |
| 80000 | 800 | 1323.2 ms | 1443.7 ms | 1.1× |

Both use MKL BLAS underneath; the gain comes from SPIKE's partitioning.

</details>

<details>
<summary><b>Direct-solver head-to-head (all four methods)</b></summary>

The optional `spike_duel` benchmark (built when Intel MKL, Fortran, LAPACK and
OpenMP are available) solves the same diagonally dominant band matrix with four
direct solvers only: GPU mixed, GPU double, MKL general band LU (double), and
the original Fortran SPIKE-SMP solver. Factorization is timed once and solve is
averaged over five solves; MKL and the Fortran solver use 8 threads. The
`factor / solve` breakdown is in ms.

| n | klu | nrhs | gspike mixed | gspike double | MKL double band | Fortran SPIKE |
|--:|----:|-----:|-------------:|--------------:|----------------:|--------------:|
| 10000 | 100 | 1 | 14.2 ms | 25.4 ms | 26.4 ms | 8.5 ms |
| 25000 | 250 | 1 | 63.3 ms | 107.8 ms | 116.9 ms | 70.4 ms |
| 50000 | 500 | 2 | 209.0 ms | 432.5 ms | 562.1 ms | 391.6 ms |
| 80000 | 800 | 1 | 571.6 ms | 1439.2 ms | 1551.1 ms | 1376.6 ms |

| n | klu | mixed fac / sol | double fac / sol | MKL fac / sol | Fortran fac / sol |
|--:|----:|----------------:|-----------------:|--------------:|------------------:|
| 10000 | 100 | 11.4 / 2.8 | 23.7 / 1.8 | 25.4 / 1.1 | 7.9 / 0.5 |
| 25000 | 250 | 48.3 / 15.0 | 98.6 / 9.3 | 102.5 / 14.4 | 65.6 / 4.8 |
| 50000 | 500 | 171.4 / 37.6 | 408.8 / 23.7 | 488.4 / 73.8 | 368.4 / 23.2 |
| 80000 | 800 | 489.7 / 81.9 | 1361.0 / 78.2 | 1443.7 / 107.4 | 1323.2 / 53.3 |

MKL's band solve is memory-bound and gets much slower as the band widens, while
the GPU and Fortran SPIKE solves stay cheap. Final relative residuals are
`~1e-13 … 1e-12` for the mixed path and `~1e-15` for the three double paths.
There are no iterative solvers in this comparison.

</details>

## 📊 Large-scale benchmark (~6 GB matrix)

An out-of-sweep case with `n = 200000, klu = 2000` and one RHS. The input matrix
alone occupies `lda × n × 8 = 4001 × 200000 × 8` bytes = **6.4 GB (5.96 GiB)**
in double-precision LAPACK band storage. Figures are from an NVIDIA L40, CUDA
12.6, diagonally dominant random band, 8 CPU threads.

```sh
./build/spike_duel 200000 2000 1
./build/spike_bench --type mixed 200000 2000 1
./build/spike_bench --type double 200000 2000 1
./build/spike_bench --compare 200000 2000 1
```

Four direct solvers (factor / solve / total in ms):

| method | parts | factor ms | solve ms | total ms | relres |
|---|---:|---:|---:|---:|---:|
| gspike mixed (GPU) | 2 | 7334 | 983 | 8317 | 2.5e-12 |
| gspike double (GPU) | 1 | 10873 | 546 | 11419 | 3.5e-15 |
| MKL double band (CPU) | — | 10575 | 1002 | 11577 | 4.1e-15 |
| Fortran SPIKE (CPU) | 8 | 19240 | 500 | 19739 | 3.6e-15 |

Against MKL's double band LU, GPU mixed is **1.39×** faster overall and GPU
double is on par (**1.01×**); Fortran SPIKE is the slowest at this size (1.71×
MKL's time). The mixed factorization sustains 1721 GFLOP/s and the solver
retains 8106 MB of device memory (double: 9352 MB).

`spike_bench --compare` measures the same Fortran SPIKE solver directly:
factor 23960 ms, solve 504 ms, total 24464 ms — **2.94×** slower than GPU
mixed. The interface system is too large for the 320 MB cap at this bandwidth,
so the automatic partitioner selects `parts = 2` for the mixed solver and falls
back to plain band LU (`parts = 1`) for double.

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
default. The optional CPU reference (Fortran + LAPACK + OpenMP) and the MKL
direct-solver head-to-head are enabled automatically when their dependencies
are found, and skipped otherwise. Neither is ever linked into the library; no
cuBLAS or cuSPARSE dependency remains.

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

`spike_duel` is the four-way direct-solver head-to-head: GPU mixed, GPU double,
MKL double band LU, and the original Fortran SPIKE-SMP solver on the same
matrix. It is built only when MKL and the CPU reference dependencies are found.
Factorization is timed once and solve is averaged over five solves; every method
reports a double-precision relative residual.

```sh
./build/spike_duel                       # default sweep (four direct solvers)
./build/spike_duel 50000 500 2           # n klu nrhs
DUEL_CPU_THREADS=16 ./build/spike_duel   # MKL and OpenMP thread count (default 8)
```

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
bench/duel.cu            optional four-way direct-solver head-to-head (MKL + Fortran)
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
