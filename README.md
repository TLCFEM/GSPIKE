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
./build/spike_bench --type cpu            # original Fortran solver
./build/spike_bench --compare             # CPU reference vs GPU, double
./build/spike_bench 200000 50 4 32        # n klu(=kl=ku) nrhs parts(0=auto)
./build/spike_bench --compare 200000 50 4 # one CPU/GPU comparison
./build/spike_bench --cpu-threads 8 ...   # reference threads (rounded to 2^k)
```

The reference uses OpenMP; its partition count must be a power of two, so the
requested thread count is rounded down (`--cpu-threads 24` uses 16) and the
pool is pinned to the allowed CPUs.  `--type cpu` and `--compare` require
`GSPIKE_BUILD_REF`.

Example on an NVIDIA L40 (CUDA 12.6), diagonally dominant random bands:

| type   |      n | klu | nrhs | parts | factor ms | solve ms | factor GFLOP/s | solve GFLOP/s | rel. residual |
|--------|-------:|----:|-----:|------:|----------:|---------:|---------------:|--------------:|--------------:|
| double |  20000 |  10 |    1 |    32 |      4.2  |     1.4  |           8.2  |          1.8  |      2.9e-16 |
| double | 100000 |  20 |    1 |    32 |     20.4  |     6.1  |          32.4  |          4.1  |      3.9e-16 |
| double | 100000 |  20 |    8 |    32 |     19.5  |     6.4  |          33.8  |         30.9  |      3.9e-16 |
| double | 200000 |  50 |    4 |    32 |    135.6  |    21.4  |          60.7  |         23.0  |      6.6e-16 |
| double | 200000 |  50 |    4 |     1 |   1073.0  |   254.3  |           1.9  |          1.0  |      6.6e-16 |
| float  | 200000 |  50 |    4 |    32 |    124.0  |    14.3  |          66.4  |         34.4  |      3.2e-07 |
| float  | 200000 |  50 |    4 |     1 |    875.7  |   172.1  |           2.3  |          1.4  |      3.2e-07 |
| mixed  | 200000 |  50 |    4 |    32 |    124.5  |    30.4  |          66.1  |         37.6  |      1.3e-13 |

(`parts = 1` is the plain parallel band LU path, i.e. SPIKE disabled.)

The same benchmark can compare against the original Fortran solver
(`--compare`), using the same analytic operation counts:

|      n | klu | nrhs | CPU parts | GPU parts | CPU factor ms | GPU factor ms | CPU solve ms | GPU solve ms | CPU factor GFLOP/s | GPU factor GFLOP/s |
|-------:|----:|-----:|----------:|----------:|--------------:|--------------:|-------------:|-------------:|-------------------:|-------------------:|
|  20000 |  10 |    1 |        16 |        32 |          1.71 |          4.19 |         0.15 |         1.43 |               19.5 |                8.2 |
|  50000 |  20 |    1 |        16 |        32 |          4.74 |         12.81 |         0.37 |         3.56 |               69.3 |               26.3 |
| 100000 |  20 |    1 |        16 |        32 |          7.09 |         20.35 |         0.62 |         6.05 |               92.0 |               32.4 |
| 100000 |  20 |    8 |        16 |        32 |          6.99 |         19.53 |         4.09 |         6.35 |               93.3 |               33.8 |
| 200000 |  50 |    4 |        16 |        32 |         49.65 |        135.88 |        13.95 |        21.36 |              163.6 |               60.6 |

The reference runs with 16 OpenMP threads on a 24-core AMD EPYC 9334.  The
CPU is still faster overall for a single banded matrix, especially for
factorization: the band LU and triangular solves have a long dependency along
columns, and the CPU threads pay less per-column overhead.  The GPU path is
retained for float/mixed precision and for workloads with more independent
parallelism (larger `nrhs` or many matrices).

With few right-hand sides the triangular solve is latency-bound, so a single
warp per column leaves most of the GPU idle.  Each column is now solved by a
whole block whose lanes sweep the band together, which roughly halves the
double solve time at `nrhs = 4` and narrows the gap to the CPU solve.  The
factorized matrix is unchanged, so factorization timing is not affected.


## Algorithm

Single-level SPIKE-DS, with `klu = max(kl, ku)` interface unknowns per side:

1. Split the matrix into `p` contiguous diagonal blocks and factor each block
   in parallel with a band LU with partial pivoting (one CUDA block per
   partition).  `A` itself is untouched; the factors live in a private band
   buffer in LAPACK's `2*kl + ku + 1` layout.
2. Solve each block against its coupling columns to obtain the spikes
   `Vp_i = A_ii^-1 C_i` and `Vm_i = A_ii^-1 B_i` (`2*klu` columns in total).
3. Eliminate the interiors, which leaves a `2*klu*(p-1)` interface system that
   is block tridiagonal with `2*klu` blocks.  It is assembled in band storage
   with half-bandwidth `3*klu` and factored once with the same band LU kernel.
4. Back-substitute the interface unknowns and correct the coupling rows of each
   block, then solve each block.

The number of partitions is chosen automatically:
`p = clamp(n / (16*klu + 32), 1, 32)`, further limited so that every partition
has at least `2*klu + 1` rows and so that the interface system stays below
256 MB.  `p == 1` degenerates to a plain band LU and solve.

## Behavior notes

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
