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

`spike_bench` measures factorization and solve time separately, reports their
sum, the device memory retained by the solver, and derives effective GFLOP/s
from analytic operation counts (band LU, triangular solves, spike work, and the
mixed-precision residual); the solver kernels are not instrumented.  `factor`
covers construction of the factorized object and `solve` is averaged over five
solves.  `--profile` adds opt-in event timings for the factorization and solve
phases (pack, band LU, spikes, reduced system, first solve, interface solve,
correction) to stderr; it adds synchronizations, so it is not used for the
headline numbers.

```sh
./build/spike_bench                       # default sweep, double
./build/spike_bench --type float          # default sweep, float
./build/spike_bench --type mixed          # default sweep, mixed
./build/spike_bench --no-full-spikes --type mixed  # force two-solve correction
./build/spike_bench --profile --type mixed      # per-phase GPU timings
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

(`parts = 1` is the plain parallel band LU path, i.e. SPIKE disabled.  `mem MB`
is the device memory retained by the solver plus its solve scratch space, not
including the input matrix.  Full spikes are enabled automatically at these
sizes; see the next table.)

With `--no-full-spikes` the second band solve is used instead of the saved
spike matvec.  The solve cost changes as follows (same factorizations):

|     n | klu | nrhs | parts | mixed solve ms (auto) | mixed solve ms (no full) | mem MB (auto / no) |
|------:|----:|-----:|------:|----------------------:|-------------------------:|-------------------:|
| 10000 | 100 |    1 |     5 |                   2.8 |                     4.6 |              32/24 |
| 25000 | 250 |    1 |     4 |                  15.0 |                    18.4 |             188/140 |
| 50000 | 500 |    2 |     4 |                  37.6 |                    61.3 |             750/558 |
| 80000 | 800 |    1 |     4 |                  81.9 |                   132.7 |            1908/1418 |

The same benchmark compares against the original Fortran solver
(`--compare`), using the same analytic operation counts:

|      n | klu | nrhs | CPU parts | GPU parts | CPU factor ms | GPU factor ms | factor x | CPU solve ms | GPU solve ms | solve x | CPU total ms | GPU total ms | total x |
|-------:|----:|-----:|----------:|----------:|--------------:|--------------:|---------:|-------------:|-------------:|--------:|-------------:|-------------:|--------:|
|  10000 | 100 |    1 |         8 |         5 |          10.1 |          11.3 |     0.89 |         0.59 |         2.80 |    0.21 |         10.6 |         14.1 |    0.75 |
|  25000 | 250 |    1 |         8 |         4 |          57.3 |          48.5 |     1.18 |         4.34 |        15.04 |    0.29 |         61.6 |         63.5 |    0.97 |
|  50000 | 500 |    2 |         8 |         4 |         338.0 |         170.2 |     1.99 |        21.71 |        37.61 |    0.58 |        359.7 |        207.8 |    1.73 |
|  80000 | 800 |    1 |         8 |         4 |        1219.9 |         489.7 |     2.49 |        44.75 |        81.87 |    0.55 |       1264.7 |        571.6 |    2.21 |

(`factor x`, `solve x` and `total x` are CPU time / GPU time, so > 1 means the
GPU is faster.)

The reference runs with 8 OpenMP threads on a 24-core AMD EPYC 9334.  With
bands around 1% of `n`, factorization dominates.  The mixed GPU factorization
is 1.2-2.5x faster than the CPU double factorization over the sweep; at
`n = 10000` the CPU reference (which has very little work per thread) is
slightly faster.  The GPU triangular solve is slower with a single right-hand
side because the substitution chain is serial, but it is a small part of the
total at these bandwidths: mixed GPU is faster than the CPU double solver
overall from `n = 50000` (1.7x) and `n = 80000` (2.2x).

Phase timings from `--profile` at `n = 50000, klu = 500, nrhs = 2, parts = 4`
(mixed, in ms, after the optimizations below) are: pack 9.7, partition band LU
80.4, reciprocal diagonal 0.1, spikes 42.6, reduced LU 32.3; a solve costs
12.5 for the first band solve, 0.04 for the interface RHS, 4.5 for the reduced
solve and 0.8 for the full-spike correction.  Before the blocked spike solve
the spike phase alone took 187 ms, so factorization dropped from about 312 ms
to about 165 ms at this size.  The reduced system is a single CUDA block and
therefore does not scale with the GPU; it is the next target.

The partition band LU is blocked: a 4-column panel is factored by one block,
then a tiled rank-4 update applies it to the trailing band.  Each thread owns
a 4x4 micro-tile and the factors staged in shared memory; a scalar kernel is
retained for the rare matrix that actually pivots.

Spike generation (the dominant factorization phase for wide bands) uses a
blocked triangular solve instead of one warp per right-hand side: 32 rows are
solved sequentially by one warp per column, then a 64x64 register micro-tile
applies the panel to 64 columns at once.  Each band factor is therefore loaded
once per 64 columns rather than once per column, which cut the spike phase
from 187 ms to 44 ms at `n = 50000, klu = 500` in fp32.  The C-spike columns
have a zero right-hand side above the last `klu` rows, so their forward sweep
starts there.  Pivoting falls back to the scalar solve, and narrow bands keep
the original warp-per-column path.  With few right-hand sides the solve still
gives each column a whole 1024-thread block, since the substitution chain is
serial.

## Against MKL and sparse GPU solvers

The optional `spike_duel` benchmark (`-DGSPIKE_DUEL=ON`, requires Intel MKL,
cuSPARSE and cuBLAS) solves the same diagonally dominant band matrix with six
methods.  Times are setup/factorization plus solve for one right-hand side;
MKL uses 8 threads.

|     n | klu | gspike mixed | gspike double | MKL double band | MKL float band | cuSPARSE CSR BiCGStab | band BiCGStab |
|------:|----:|-------------:|--------------:|----------------:|---------------:|----------------------:|--------------:|
| 10000 | 100 |        17.5  |         25.5  |           34.6  |          18.9  |                  7.3  |         1.1  |
| 25000 | 250 |        65.5  |        107.4  |          127.3  |          80.8  |                  8.9  |         2.7  |
| 50000 | 500 |       206.9  |        426.5  |          529.0  |         266.5  |                 19.8  |         8.9  |
| 80000 | 800 |       570.2  |       1436.1  |         1532.8  |         731.1  |                 45.6  |        22.7  |

(`band BiCGStab` uses the same iteration as the cuSPARSE variant, but replaces
the CSR matrix-vector product with a kernel that reads the contiguous band
segments directly: one warp per column, no column indices and no CSR build.)

On these well-conditioned bands the iterative methods converge in 5-6
iterations, so a direct factorization cannot win: the band-aware BiCGStab is
2-6x faster than generic CSR on the GPU, and 10-60x faster than any direct
solver.  gspike's **double-precision** factorization is 1.1-1.4x faster than
MKL's double band LU (and its full-spike solve 1.1-1.5x faster), so gspike
double beats MKL double end to end at every size.  The mixed solver is 2.0-2.7x
faster end to end than MKL double, with a much better residual than MKL float
(1e-13 vs 1e-6).

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
`2*klu + 1` rows and so that the interface system stays below 320 MB.
`p == 1` degenerates to a plain band LU and solve.

## Behavior notes

* `opts::full_spikes` retains the complete spike solutions computed during
  factorization, so each `solve()` applies the interface correction with a
  matvec instead of a second band solve (1.6-1.8x faster solves here).  It
  costs `2*klu*n` scalars of device memory -- about 1 GB in double at
  `n = 80000, klu = 800` -- so when `opts::full_spikes` is false the solver
  enables it automatically only if the cost is at most 512 MB and at least
  three times that much device memory is free (`--no-full-spikes` in the
  benchmark disables the automatic choice).  Accuracy is unchanged.
* `opts::profile` prints factorization and solve phase timings to stderr.  It
  is intended for benchmarking and adds event synchronizations.
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
