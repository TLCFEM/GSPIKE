# gspike

A small CUDA library for solving banded linear systems with a SPIKE
(factorization + interface reduction) solver.  It is a C++20 port of the
Fortran `Xspike_smp` banded solver in `ref/`, templated on the scalar type
and limited to `float` and `double`.

Given a band matrix `A` (lower bandwidth `kl`, upper bandwidth `ku`) in LAPACK
band storage and a right-hand side `B`, it computes `X` with `A * X = B`:

```cpp
#include <gspike/spike.h>

gspike::opts o;             // defaults are fine
gspike::stats st;
int info = gspike::spike_gbsv<double>(n, kl, ku, nrhs, dA, lda, dB, ldb, o, &st);
```

`A`, `B` live in device memory in column-major order.  `A` is only read; `B`
is overwritten with the solution.  `info == 0` means success.

When the same matrix is solved for many right-hand sides, factor once:

```cpp
gspike::solver<double> s(n, kl, ku, dA, lda);
s.solve(nrhs, dB, ldb, &st);      // can be called repeatedly
```

A double matrix can also be factorized in float and corrected against the
original double matrix:

```cpp
gspike::mixed_solver s(n, kl, ku, dA, lda);
s.solve(nrhs, dB, ldb, &st);      // two float solves, double residual
```

The mixed solver copies `A` to device float storage, factorizes it with
`solver<float>`, solves for the right-hand side, forms the residual with the
double matrix and applies one more float solve as a correction.  For well
conditioned systems this recovers about double precision while paying for a
float factorization.

## Storage

`A(i,j)` is stored at `A[lda * j + ku + i - j]` with `lda >= kl + ku + 1`, for
`i - j` in `[-ku, kl]`.  This is exactly the layout used by LAPACK `dgbmv` /
`dgbtrf` before factorization.

## Build

Requires CUDA (tested with 12.6) and CMake >= 3.20; C++20 is used.  The
benchmark builds by default; pass `-DGSPIKE_BENCH=OFF` to skip it.  Its CPU
comparison uses the original Fortran solver and additionally needs a Fortran
compiler, LAPACK and OpenMP; pass `-DGSPIKE_BUILD_REF=OFF` to disable it (it
is skipped automatically when no Fortran compiler is found).  The reference
is linked into `spike_bench` only and is never part of the `gspike` library.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Debug builds enable host-side AddressSanitizer (`-fsanitize=address`),
including the host code of the `.cu` files.  ASan does not instrument device
memory; use `compute-sanitizer` for device-side errors.  CUDA's UVA allocator
needs the shadow gap that ASan protects by default, so direct runs need
`ASAN_OPTIONS=protect_shadow_gap=0`; CTest sets this for `spike_tests`.

```sh
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug -j
ctest --test-dir build-debug --output-on-failure
```

The library target is `gspike` (alias `gspike::gspike`); the header is
`include/gspike/spike.h` and depends only on the CUDA runtime at link time.  Only `float` and `double` are instantiated, so a translation unit
that uses another type fails to link rather than silently doing something
unexpected.

## Tests and benchmark

`spike_tests` checks both precisions against a dense host LU reference and
against the relative residual, for tridiagonal, diagonally dominant,
non-symmetric bandwidth (`kl < ku` and `kl > ku`), forced and automatic
partition counts, multiple right-hand sides, non diagonally dominant matrices
(with pivoting), solver reuse, argument validation and a randomized fuzz pass.
The mixed solver is checked against the dense reference as well.

`spike_bench` measures factorization and solve time separately and derives
effective GFLOP/s from analytic operation counts (band LU, triangular solves,
spike work, and the mixed-precision residual); the solver is not instrumented.
`factor` covers construction of the factorized object and `solve` is averaged
over five solves.

```sh
./build/spike_bench                       # default sweep, double
./build/spike_bench --type float          # default sweep, float
./build/spike_bench --type mixed          # default sweep, mixed
./build/spike_bench --full-spikes --type mixed  # same, with full-spike solves
./build/spike_bench --type cpu            # original Fortran solver
./build/spike_bench --compare             # CPU double vs GPU mixed sweep
./build/spike_bench 200000 50 4 32        # n klu(=kl=ku) nrhs parts(0=auto)
./build/spike_bench --compare 50000 500 2 # one CPU/GPU comparison
./build/spike_bench --cpu-threads 8 ...   # reference threads (rounded to 2^k)
```

The default sweeps use bands around 1% of `n` (`klu = n/100`) and at most a
few right-hand sides.  The largest case is `n = 80000, klu = 800`, which peaks
near 2.5 GB of device memory; `n = 100000` comes in around 3.7 GB and still
fits in 4 GB.

`--compare` runs the double-precision CPU reference against the GPU mixed
solver, which factorizes in float and corrects with a double residual.

The reference uses OpenMP; its partition count must be a power of two, so the
requested thread count is rounded down (`--cpu-threads 24` uses 16) and the
pool is pinned to the allowed CPUs.  `--type cpu` and `--compare` require
`GSPIKE_BUILD_REF`.

Example on an NVIDIA L40 (CUDA 12.6), diagonally dominant random bands:

| type   |     n | klu | nrhs | parts | factor ms | solve ms | factor GFLOP/s | solve GFLOP/s | rel. residual |
|--------|------:|----:|-----:|------:|----------:|---------:|---------------:|--------------:|--------------:|
| double | 10000 | 100 |    1 |     5 |     28.9  |     5.6  |          58.9  |          2.3  |      7.8e-16 |
| double | 25000 | 250 |    1 |     4 |    130.9  |    30.1  |         196.1  |          2.6  |      1.2e-15 |
| double | 50000 | 500 |    2 |     4 |    684.4  |    90.5  |         299.9  |          6.8  |      1.7e-15 |
| double | 80000 | 800 |    1 |     3 |   2557.2  |   204.3  |         319.6  |          3.8  |      2.1e-15 |
| double | 25000 | 250 |    1 |     1 |    182.7  |    50.2  |          33.9  |          0.7  |      1.2e-15 |
| float  | 10000 | 100 |    1 |     5 |     13.2  |     2.9  |         128.8  |          4.3  |      3.4e-07 |
| float  | 25000 | 250 |    1 |     4 |     75.4  |     9.7  |         340.5  |          8.0  |      5.2e-07 |
| float  | 50000 | 500 |    2 |     4 |    397.3  |    37.2  |         516.6  |         16.6  |      7.4e-07 |
| float  | 80000 | 800 |    1 |     4 |   1406.7  |    76.7  |         597.6  |         10.3  |      9.3e-07 |
| mixed  | 25000 | 250 |    1 |     4 |     75.0  |    22.9  |         342.1  |          7.8  |      3.0e-13 |
| mixed  | 50000 | 500 |    2 |     4 |    402.9  |    76.4  |         509.6  |         18.8  |      6.0e-13 |
| mixed  | 80000 | 800 |    1 |     4 |   1408.1  |   158.2  |         597.1  |         11.6  |      9.5e-13 |

(`parts = 1` is the plain parallel band LU path, i.e. SPIKE disabled.)

The same benchmark compares against the original Fortran solver
(`--compare`), using the same analytic operation counts:

|      n | klu | nrhs | CPU parts | GPU parts | CPU factor ms | GPU factor ms | factor x | CPU solve ms | GPU solve ms | solve x | CPU factor GFLOP/s | GPU factor GFLOP/s |
|-------:|----:|-----:|----------:|----------:|--------------:|--------------:|---------:|-------------:|-------------:|--------:|-------------------:|-------------------:|
|  10000 | 100 |    1 |        16 |         5 |          33.5 |          13.2 |     2.54 |         1.38 |         5.94 |    0.23 |               69.8 |              129.0 |
|  25000 | 250 |    1 |        16 |         4 |         492.0 |          75.0 |     6.56 |        18.13 |        22.73 |    0.80 |               74.2 |              342.2 |
|  50000 | 500 |    2 |        16 |         4 |        3917.5 |         401.4 |     9.76 |       125.03 |        76.39 |    1.64 |               74.5 |              511.4 |
|  80000 | 800 |    1 |        16 |         4 |       16300.1 |        1412.9 |    11.54 |       416.06 |       158.20 |    2.63 |               73.3 |              595.0 |

(`factor x` and `solve x` are CPU time / GPU time, so > 1 means the GPU is
faster.)

The reference runs with 16 OpenMP threads on a 24-core AMD EPYC 9334.  With
bands around 1% of `n`, factorization dominates.  The mixed GPU factorization
is 6.6-11.5x faster than the CPU double factorization over the sweep, and the
GPU double factorization is also 3-4x faster.  The GPU triangular solve is
still slower with a single right-hand side, because the substitution chain is
serial, but it is a small part of the total at these bandwidths: mixed GPU is
faster than the CPU double solver overall at every size (about 4x at
`n = 50000`).

The partition band LU is blocked: a 16-column panel is factored by one block,
then a tiled rank-16 update applies it to the trailing band.  Each thread owns
a 4x4 micro-tile, so the factors staged in shared memory are reused sixteen
times, and the update fills all SMs.  A scalar kernel is retained as a
fallback for the rare matrix that actually pivots.  With few right-hand sides
the triangular solve gives each column a whole block (or four warps in the
spike solves), and the band is packed with one warp per column.

## Against MKL and sparse GPU solvers

The optional `spike_duel` benchmark (`-DGSPIKE_DUEL=ON`, requires Intel MKL,
cuSPARSE and cuBLAS) solves the same diagonally dominant band matrix with four
methods.  Times are factor/setup plus solve for one right-hand side; MKL uses
8 threads.

|     n | klu | gspike mixed | MKL double band | MKL float band | cuSPARSE CSR BiCGStab | band BiCGStab |
|------:|----:|-------------:|----------------:|---------------:|----------------------:|--------------:|
| 10000 | 100 |        20.6  |           35.8  |          18.6  |                  7.6  |         1.1  |
| 25000 | 250 |       101.0  |          128.5  |          74.6  |                  8.8  |         2.6  |
| 50000 | 500 |       464.0  |          529.2  |         267.0  |                 19.9  |         8.9  |
| 80000 | 800 |      1513.0  |         1530.4  |         743.3  |                 45.8  |        22.8  |

(`band BiCGStab` uses the same iteration as the cuSPARSE variant, but replaces
the CSR matrix-vector product with a kernel that reads the contiguous band
segments directly: one warp per column, no column indices and no CSR build.)

On these well-conditioned bands the iterative methods converge in 5-6
iterations, so a direct factorization cannot win: the band-aware BiCGStab is
2-6x faster than generic CSR on the GPU, and 10-50x faster than any direct
solver.  gspike's mixed factorization is still 1.1-1.7x faster than MKL's
double band LU and 1.2-1.8x faster than MKL's float band LU, with a much
better residual than MKL float (1e-13 vs 1e-6).

The picture reverses on non-dominant bands: with `n = 25000, klu = 250` both
BiCGStab variants stagnate (2000 iterations, NaN residual), while the direct
solvers finish in about 0.1-0.4 s.  A direct factorization remains the robust
choice when pivoting is needed or many right-hand sides are solved from one
factorization; for a well-conditioned band and few right-hand sides, an
iterative method is the right tool.

### The CPU reference

The original Fortran SPIKE-SMP solver (`GSPIKE_BUILD_REF`) is a blocked band
LU: `DGBALU` takes its panel width from `ILAENV` (capped by `min(kl, ku)`),
factorizes a panel with BLAS2, then applies it with `DTRSM` and `DGEMM`.  The
partition loop and the recursive reduced system use OpenMP.  CMake links the
best LAPACK/BLAS it finds; **when Intel MKL is installed the reference uses
MKL's BLAS kernels** — with the netlib BLAS the same factorization takes about
3.9 s instead of 0.34 s at `n = 50000, klu = 500`.

Profiling the 16-thread run at that size gives ~185 ms for the partition
factorizations and ~215 ms for the recursive reduced system, whose last level
has only one or two active blocks and therefore little OpenMP parallelism.
Eight threads are faster than sixteen: each worker then gets two balanced
partitions and the reduced system suffers less from idle workers.  The
benchmark defaults to eight threads; equal-sized partitions and SPIKE's own
auto-tuning were both measured slower, so the original partition ratios are
kept.

Against MKL's own general band LU (`dgbtrf`/`dgbtrs`, 8 threads) on the same
matrix:

|      n | klu | SPIKE factor ms | MKL `dgbtrf` ms | speedup |
|-------:|----:|----------------:|----------------:|--------:|
|  10000 | 100 |            10.2 |            34.0 |    3.3x |
|  25000 | 250 |            57.7 |           117.2 |    2.0x |
|  50000 | 500 |           336.1 |           495.1 |    1.5x |
|  80000 | 800 |          1218.8 |          1456.5 |    1.2x |

Both use MKL's BLAS underneath; the win comes from SPIKE's divide-and-conquer
partitioning, not from a different kernel library.  The single-thread
reference remains competitive with MKL as well (674 ms at `n = 50000`).

## Algorithm

Single-level SPIKE-DS, with `klu = max(kl, ku)` interface unknowns per side:

1. Split the matrix into `p` contiguous diagonal blocks and factor each block
   in parallel with a blocked band LU with partial pivoting (one CUDA block
   per panel, with tiled rank-16 trailing updates).  `A` itself is untouched; the factors live in a private band
   buffer in LAPACK's `2*kl + ku + 1` layout.
2. Solve each block against its coupling columns to obtain the spikes
   `Vp_i = A_ii^-1 C_i` and `Vm_i = A_ii^-1 B_i` (`2*klu` columns in total).
3. Eliminate the interiors, which leaves a `2*klu*(p-1)` interface system that
   is block tridiagonal with `2*klu` blocks.  It is assembled in band storage
   with half-bandwidth `3*klu` and factored once with the same band LU kernel.
4. Back-substitute the interface unknowns and correct the coupling rows of each
   block, then solve each block.

The number of partitions is chosen automatically from two bounds: the
size-based `n / (16*klu + 32)` rule (good for narrow bands) and a balance
between the per-partition cost, which falls as `1/p`, and the reduced
interface LU, whose work grows with `p*klu^3`.  The smaller of the two leaves
narrow bands at up to 32 partitions while picking 4-12 for bands around 1% of
`n`.  The count is further limited so that every partition has at least
`2*klu + 1` rows and so that the interface system stays below 256 MB (384 MB
in double precision).  `p == 1` degenerates to a plain band LU and solve.

## Behavior notes

* `opts::full_spikes` retains the complete spike solutions computed during
  factorization, so each `solve()` applies the interface correction with a
  matvec instead of a second band solve (about 1.4-3x faster solves).  It costs
  `2*klu*n` scalars of device memory -- about 1 GB in double at
  `n = 80000, klu = 800` -- so it is off by default.  Accuracy is unchanged.
* `info > 0` reports a zero (perturbed) pivot: values up to `n` are rows of
  `A`, larger values belong to the reduced interface system.  Factorization
  still completes, so verify the residual in the application.
* `solver` and `mixed_solver` keep the pointers passed to their
  constructors, so `A` must outlive them.  `solve()` uses private scratch
  space and is not safe to call concurrently on the same object; separate
  objects are independent.
* CUDA failures throw `std::runtime_error`; bad arguments are
  reported through negative `info` values (or an exception from the `solver`
  constructor).

## Layout

```
include/gspike/spike.h   public API (no CUDA types in the header)
src/common.cuh           shared host helpers (error checks, device buffers)
src/kernels.cuh          device kernels
src/spike.cu             host side: partitioning, factorization, solve
src/mixed.cu             double/float mixed-precision solver
tests/test_spike.cu      correctness tests
bench/bench.cu           benchmark tool
bench/CMakeLists.txt     benchmark build (optional CPU reference)
ref/                     original Fortran SPIKE-SMP solver
ref/spike_ref.f90        C-callable wrapper used by the benchmark
```
