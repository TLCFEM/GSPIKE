# gspike

A small CUDA library for solving banded linear systems with a SPIKE
(factorization + interface reduction) solver.  It is a C++20 port of the
Fortran `Xspike_smp` banded solver in `SPIKE/`, templated on the scalar type
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

## Storage

`A(i,j)` is stored at `A[lda * j + ku + i - j]` with `lda >= kl + ku + 1`, for
`i - j` in `[-ku, kl]`.  This is exactly the layout used by LAPACK `dgbmv` /
`dgbtrf` before factorization.

## Build

Requires CUDA (tested with 12.6) and CMake >= 3.20; C++20 is used.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The library target is `gspike` (alias `gspike::gspike`); the header is
`include/gspike/spike.h` and depends only on the CUDA runtime and cuBLAS at
link time.  Only `float` and `double` are instantiated, so a translation unit
that uses another type fails to link rather than silently doing something
unexpected.

## Tests and benchmark

`spike_tests` checks both precisions against a dense host LU reference and
against the relative residual, for tridiagonal, diagonally dominant,
non-symmetric bandwidth (`kl < ku` and `kl > ku`), forced and automatic
partition counts, multiple right-hand sides, non diagonally dominant matrices
(with pivoting / iterative refinement), solver reuse and argument validation.

`spike_bench` measures factorization and solve time separately:

```sh
./build/spike_bench                       # default sweep, double
./build/spike_bench --type float          # default sweep, float
./build/spike_bench 200000 50 4 32        # n klu(=kl=ku) nrhs parts(0=auto)
```

Example on an NVIDIA L40 (CUDA 12.6), diagonally dominant random bands:

| type   |      n | klu | nrhs | parts | factor ms | solve ms | rel. residual |
|--------|-------:|----:|-----:|------:|----------:|---------:|--------------:|
| double |  20000 |  10 |    1 |    32 |     18.8  |     5.2  |      2.9e-16 |
| double | 100000 |  20 |    1 |    32 |     38.8  |    26.8  |      3.9e-16 |
| double | 100000 |  20 |    8 |    32 |     38.8  |    35.6  |      3.9e-16 |
| double | 200000 |  50 |    4 |    32 |    380.0  |   243.9  |      6.7e-16 |
| double | 200000 |  50 |    4 |     1 |   2171.3  |  2243.7  |      6.7e-16 |
| float  | 200000 |  50 |    4 |    32 |    254.4  |   113.1  |      3.6e-07 |

(`parts = 1` is the plain parallel band LU path, i.e. SPIKE disabled; the
last row shows the speedup from the partitioning.)

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
5. Optionally refine against the original matrix (`opts::max_refine`,
   `opts::rtol`) and report the achieved relative residual in `stats`.

The number of partitions is chosen automatically:
`p = clamp(n / (16*klu + 32), 1, 32)`, further limited so that every partition
has at least `2*klu + 1` rows and so that the interface system stays below
256 MB.  `p == 1` degenerates to a plain band LU and solve.

## Behavior notes

* `opts::max_refine = 3` by default.  Refinement stops early when the residual
  target is met or cannot be improved; `stats::refine` reports the number of
  steps actually taken.  Set it to `0` for a pure direct solve.
* `info > 0` reports a zero (perturbed) pivot: values up to `n` are rows of
  `A`, larger values belong to the reduced interface system.  Factorization
  still completes, so inspect `stats::residual`.
* `solver` keeps the pointers passed to its constructor, so `A` must outlive
  it.  `solve()` uses private scratch space and is not safe to call
  concurrently on the same object; separate objects are independent.
* CUDA or cuBLAS failures throw `std::runtime_error`; bad arguments are
  reported through negative `info` values (or an exception from the `solver`
  constructor).

## Layout

```
include/gspike/spike.h   public API (no CUDA types in the header)
src/kernels.cuh          device kernels
src/spike.cu             host side: partitioning, factorization, solve
tests/test_spike.cu      correctness tests
bench/bench.cu           benchmark tool
SPIKE/                   original Fortran reference implementation
```
