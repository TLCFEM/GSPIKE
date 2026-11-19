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

## Mixed precision

A double problem can be solved with a single-precision factorization.  The
matrix is copied to float once and factorized; the solution is then built in
double precision by iterative refinement, where the residual `B - A X` and the
accumulated `X` are double and every correction is a solve with the same float
factors:

```cpp
gspike::mixed_solver s(n, kl, ku, dA, lda);  // float factorization of dA
s.solve(nrhs, dB, ldb, &st);                 // double in, double out
```

`opts::mixed_iters` (default 2) caps the number of float solves: the first
produces `A_f^-1 B` and each further one corrects it.  The factorization is
reused, so extra right-hand sides cost about one float solve each.
`stats::refine` reports the float solves actually kept, including the first (a
late correction that does not reduce the residual is undone), and
`stats::residual` is the double residual of the returned solution.
`spike_gbsv_mixed()` is the one-shot form.

With two float solves the residual of the random banded problems used by the
benchmark is typically `1e-14` to `1e-13`, versus about `1e-7` for one float
solve and `1e-16` for a pure double solve.  A third solve reaches the last
digits when the target is tightened too (`opts::mixed_iters = 3`,
`opts::rtol = 1e-15`).  The one-shot `solver<T>::solve_direct(nrhs,
RHS, ldr)` exposes the underlying direct solve with the stored factors (no
refinement), and `solver<T>::info()` reports the factorization status.

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
`include/gspike/spike.h` and depends only on the CUDA runtime, cuBLAS and
cuSOLVER at link time.  Only `float` and `double` are instantiated, so a translation unit
that uses another type fails to link rather than silently doing something
unexpected.

## Index type

All sizes, strides and offsets are expressed with `gspike::index_t`
(`include/gspike/index.h`).  It is 32-bit by default; configure with
`-DGSPIKE_INDEX_TYPE=int64` to build the 64-bit variant:

```sh
cmake -S . -B build-int64 -DGSPIKE_INDEX_TYPE=int64
```

The choice is part of the ABI: the library and its kernels are compiled for
the selected width, and CMake propagates the matching definition to targets
that link `gspike`.  `gspike::index_64bit` reports the active setting.  A
64-bit build removes the `2^31` limit on the number of matrix and right-hand
side elements; the grid dimensions of individual kernels still have their
usual CUDA limits.

## Tests

`spike_tests` checks both precisions against a dense host LU reference and
against the relative residual, for tridiagonal, diagonally dominant,
non-symmetric bandwidth (`kl < ku` and `kl > ku`), forced and automatic
partition counts, multiple right-hand sides, non diagonally dominant matrices
(with pivoting / iterative refinement), solver reuse and argument validation.
It also runs the benchmark's tuning matrix (one, four, twenty and 1024
right-hand sides at `klu = 20` and `klu = 65`, and explicit partition counts of
32, 64 and 128), solves the same wide-band problem once with the dense cuSOLVER
interface and once with the banded one to check that both paths agree, and
covers a zero pivot in the reduced interface system (`info > n`).

It also runs a seeded randomized ("fuzzy") suite that varies `n`, `kl`, `ku`,
`nrhs`, the leading dimension (including padding), the partition count and the
refinement setting; matrices are randomly banded, dominant, decayed and/or
column scaled.  Every case checks `info`, the relative residual, that `A` and
the `ldb` padding are untouched and, for small systems, the dense reference
solution; batch and column-by-column solves are also cross-checked.  The
generator is reproducible and can be widened:

```sh
./build/spike_tests                              # fixed cases + 256 fuzzy cases
./build/spike_tests --seed 123456 --fuzzy-iters 2000
```

The mixed-precision interface has its own fixed and seeded-random cases: one,
two and four float solves on dominant and non diagonally dominant matrices,
the one-shot entry point, factorization reuse across right-hand-side sets (and
scratch growth), and argument validation.  The tests also check that
`solve_direct` leaves the `ldb` padding untouched.

## Benchmark and performance

`spike_bench` times factorization and each solve with CUDA events and reports
algorithmic GFLOPS next to the times.  The operation count is computed on the
host from the public API (`n`, `kl`, `ku`, `solver::parts()`,
`solver::dense_interface()` and `stats::refine`) using the same loop structure
as the kernels, so the library is not instrumented and the timings are
unaffected.  A right-hand side reset between repetitions is outside the timed
region, and every timed construction is preceded by an untimed one of the same
solver so that `fact_ms` excludes the process-wide first touches (cuBLAS and
cuSOLVER handle creation, kernel module loading, cuSOLVER's own per-size
workspaces).

```sh
./build/spike_bench                          # default sweep, double (ends with nrhs 1..20)
./build/spike_bench --type float             # same sweep, float
./build/spike_bench --mixed                  # same sweep, mixed precision
./build/spike_bench --matrix                 # the partition-count tuning matrix
./build/spike_bench --nrhs-sweep             # only the nrhs = 1..20 solve-scaling table
./build/spike_bench --reps 10 200000 50 4    # n klu(=kl=ku) nrhs [parts]
./build/spike_bench 200000 50 4 32           # explicit partition count
```

`--matrix` runs the problems the automatic partition count was tuned on
(`20000/10`, `100000/20`, `100000/65`, `200000/50`, `500000/20`, `500000/65`),
each with one and four right-hand sides, and respects `--type`/`--mixed`.

Columns: `ref` is the number of refinement steps actually performed;
`fact_ms` / `solve_ms` are CUDA-event times, with the solve averaged over
`--reps` runs; `fact_G` / `solve_G` are GFLOP per factorization / per solve;
`fact_GF/s` / `solve_GF/s` are the corresponding throughputs.  In the
solve-scaling table `us/rhs` is the average cost of one right-hand side.

Flop conventions: a division counts as one flop and a fused multiply-add as
two; pivot search/comparisons and data movement are not counted.  The counts
include the `2*klu` spike solves, the reduced interface factorization and
solve, the corner corrections and the iterative-refinement work (matvec,
update, norms).  The interface factorization is counted as the band LU
(`O(m * klr^2)`) or, when cuSOLVER is selected, as a dense LU (`2/3 m^3`).
The numbers therefore describe the algorithm rather than a hardware counter;
when the dense interface dominates, the reported factor GF/s is much higher
than what the band kernels alone sustain and is not comparable with releases
that always counted a banded interface.

Example on an NVIDIA L40 (CUDA 12.6, 32-bit index build), diagonally dominant
random bands, one default sweep:

| type   |      n | klu | nrhs | parts | fact ms | fact GF/s | solve ms | solve GF/s | rel. residual |
|--------|-------:|----:|-----:|------:|--------:|----------:|---------:|-----------:|--------------:|
| double |  20000 |  10 |    1 |    23 |    3.8  |      8.8  |     2.1  |       1.7  |      2.7e-16 |
| double | 100000 |  20 |    8 |    32 |   14.9  |     44.4  |    11.3  |      23.6  |      3.5e-16 |
| double | 200000 |  50 |    4 |    27 |   85.7  |    230.0  |    28.3  |      24.8  |      6.0e-16 |
| double | 200000 |  50 |    4 |     1 |  636.0  |     12.6  |   337.4  |       1.2  |      6.0e-16 |
| double | 500000 |  65 |    4 |    24 |  248.5  |    207.8  |    91.0  |      23.8  |      6.7e-16 |
| double |  80000 |  15 | 1024 |    34 |   11.3  |     26.4  |   146.5  |     140.7  |      3.1e-16 |
| float  |  20000 |  10 |    1 |    33 |    2.5  |     82.4  |     0.8  |       5.2  |      1.5e-07 |
| float  | 100000 |  20 |    8 |    46 |    7.8  |    581.3  |     3.4  |      92.3  |      1.9e-07 |
| float  | 200000 |  50 |    4 |    36 |   28.6  |   1278.7  |    15.7  |      47.5  |      3.2e-07 |
| float  | 200000 |  50 |    4 |     1 |  404.0  |     19.9  |   245.7  |       1.7  |      3.2e-07 |
| float  | 500000 |  65 |    4 |    28 |   95.9  |    653.0  |    58.5  |      37.5  |      3.6e-07 |
| float  |  80000 |  15 | 1024 |    49 |    6.2  |    369.0  |    35.4  |     694.2  |      1.7e-07 |

The `parts` column is now chosen by the policy described under *Algorithm*, so
it differs from row to row; the `parts = 1` rows replace SPIKE with a plain
band LU and remain by far the slowest configurations (a single 200000x50
factorization takes 0.40 s in float and 0.64 s in double, against 29 ms and
86 ms with 27-36 partitions).

The factor throughput is highest where the reduced interface is dense: at
`n = 200000`, `klu = 50` the interface is `m = 3500`, and its `2/3 m^3` LU
dominates the operation count even though it is only a small part of the
runtime (hence 1279 GF/s in float and 230 GF/s in double -- algorithmic, not a
hardware rate).  Excluding that term, the band kernels sustain roughly
300-400 GF/s in float and 90 GF/s in double on these sizes.  A batched solve
reaches 694 GF/s (float) and 141 GF/s (double) with 1024 right-hand sides at
`n = 80000`, `kl = ku = 15`; the figure plateaus there, since 60000-100000 rows
and 15-20 half-bandwidth all land within a few percent.  Solve throughput
grows with `nrhs` because the band solves parallelize over columns; with few
right-hand sides the serial row recurrences leave most of the GPU idle (the
`nrhs = 1..20` sweep below quantifies this).  Factorization is limited by the
same recurrences and by the small number of spike columns (`2*klu`).  Timings
and GFLOPS vary with clocks and run order; the 1024-RHS cases also allocate
several GB of scratch space.

The 64-bit index build (`-DGSPIKE_INDEX_TYPE=int64`) is a few percent slower in
factorization and up to ~1.5x slower on solve-heavy rows, because `index_t`
arithmetic sits in the serial row recurrences.

### Mixed precision

`--mixed` runs the same sweep with the mixed solver (float factorization, two
float solves per solve, double in and out):

```sh
./build/spike_bench --mixed
```

| type  |      n | klu | nrhs | parts | fact ms | solve ms | rel. residual |
|-------|-------:|----:|-----:|------:|--------:|---------:|--------------:|
| mixed |  20000 |  10 |    1 |    33 |    2.6  |     1.7  |      1.9e-14 |
| mixed | 100000 |  20 |    8 |    46 |    7.9  |     7.4  |      3.2e-14 |
| mixed | 200000 |  50 |    4 |    36 |   28.8  |    32.1  |      1.1e-13 |
| mixed | 200000 |  50 |    4 |     1 |  404.0  |   492.6  |      1.1e-13 |
| mixed | 500000 |  65 |    4 |    28 |   98.7  |   119.9  |      1.3e-13 |
| mixed |  80000 |  15 | 1024 |    49 |    6.2  |    89.5  |      2.3e-14 |

Compared with the double rows above, the factorization is about twice as fast
and a solve with a few right-hand sides is ~1.5x faster, while the residual
grows from ~1e-16 to ~1e-13.  Batched solves with many right-hand sides are
dominated by the float solves, so 1024 right-hand sides take 89 ms (two float
solves) instead of 35 ms; requesting one float solve
(`opts::mixed_iters = 1`) is as fast as the float solver but only reaches
float accuracy.  The GFLOPS columns use the same flop model, with the float
solves counted like refinement steps.

### Solve scaling for 1-20 right-hand sides

The default sweeps end with a table focused on few right-hand sides: one
factorization of a fixed problem (`n = 100000`, `kl = ku = 20`) is made once
and reused for every row, so only the solve changes with `nrhs`.
`--nrhs-sweep` runs just this table for the selected solver:

```sh
./build/spike_bench --nrhs-sweep            # double
./build/spike_bench --type float --nrhs-sweep
./build/spike_bench --mixed --nrhs-sweep
```

| nrhs | double ms | double GF/s | float ms | float GF/s | mixed ms | mixed GF/s | mixed rel. residual |
|-----:|----------:|------------:|---------:|-----------:|---------:|-----------:|--------------------:|
|    1 |      7.3  |        4.5  |     2.9  |      13.4  |     6.0  |       13.2 |            3.2e-14 |
|    2 |      7.3  |        9.1  |     3.1  |      25.5  |     6.3  |       24.9 |            3.2e-14 |
|    4 |      8.4  |       15.8  |     3.1  |      50.5  |     6.5  |       48.4 |            3.2e-14 |
|    8 |     11.3  |       23.6  |     3.4  |      91.8  |     7.3  |       86.1 |            3.2e-14 |
|   12 |     11.1  |       36.1  |     3.6  |     130.4  |     8.1  |      116.8 |            3.2e-14 |
|   16 |     11.7  |       45.6  |     3.6  |     172.9  |     8.9  |      141.2 |            3.2e-14 |
|   20 |     11.4  |       58.4  |     3.7  |     210.1  |     8.9  |      176.5 |            3.2e-14 |

Twenty times the work costs only ~1.3-1.6x the time: with one right-hand side
the solve is dominated by the serial row recurrences and kernel latency, so
the per-RHS cost falls from 7.3 ms to 0.57 ms in double (2.9 ms to 0.19 ms in
float) and the throughput rises from 4.5 to 58 GF/s (13 to 210 GF/s) -- still
below the batched plateau above.  A mixed solve performs two float solves and
is now slightly faster than a single double solve here (6.0 vs 7.3 ms at
`nrhs = 1`, 8.9 vs 11.4 ms at 20) while keeping a ~3e-14 residual.  Its main
advantage is still the factorization: the float copy of this problem factors in
7.9 ms against 14.9 ms in double, which matters when a matrix is factorized
and then solved for only a few right-hand sides.

## Algorithm

Single-level SPIKE-DS, with `klu = max(kl, ku)` interface unknowns per side:

1. Split the matrix into `p` contiguous diagonal blocks and factor each block
   in parallel with a band LU with partial pivoting (one CUDA block per
   partition).  `A` itself is untouched; the factors live in a private band
   buffer in LAPACK's `2*kl + ku + 1` layout.
2. Solve each block against its coupling columns to obtain the spikes
   `Vp_i = A_ii^-1 C_i` and `Vm_i = A_ii^-1 B_i` (`2*klu` columns in total).
3. Eliminate the interiors, which leaves a `2*klu*(p-1)` interface system that
   is block tridiagonal with `2*klu` blocks.  It is factored once, either as a
   dense matrix with cuSOLVER's LU or in band storage (half-bandwidth `3*klu`)
   with the band LU kernel, whichever is predicted to be faster.
4. Back-substitute the interface unknowns and correct the coupling rows of each
   block, then solve each block.
5. Optionally refine against the original matrix (`opts::max_refine`,
   `opts::rtol`) and report the achieved relative residual in `stats`.

The band solves in steps 2 and 4 run one warp per right-hand side (a thread per
right-hand side above 32 columns), so a block stages its right-hand sides in
shared memory only when every warp fits.  Forward substitution is cooperative
within a warp; the backward pass walks the `U` rows with the lanes covering
one band row at a time.  That pass is latency-bound and accounts for roughly
half of a one-right-hand-side solve, but two alternatives measured slower on
the tuning matrix: a right-looking update loop (+5-15% solve time on every
tested shape) and a compact row-major copy of `U` (+15% at `kl = ku = 65` but
6-10% at `kl = ku = 20`, plus `n * (kl + ku)` extra storage), so the band
layout is kept.

The number of partitions is chosen automatically:
`p = clamp(1.65 * n^0.42 / klu^0.5, 1, 128)` (`1.15` replaces `1.65` for
`double`), raised to at least `min(18, n / (16*klu + 32))` so that a partition
never grows so large that the band solve re-reads it once per RHS tile, and
further limited by the device's SM count, by a minimum of `2*klu + 1` rows per
partition, and by the interface system: it must fit the storage of its
factorization path and, for the automatic count, the banded path must not be
predicted to take longer than 25 ms.  The dense path is preferred only while
`m^2` stays within its cap and its LU is cheaper than the banded one, so narrow
bands keep many partitions while wide bands stop at a small interface.
`p == 1` degenerates to a plain band LU and solve.

## Behavior notes

* `opts::max_refine = 3` by default.  Refinement stops early when the residual
  target is met or cannot be improved; `stats::refine` reports the number of
  steps actually taken.  Set it to `0` for a pure direct solve.
* `info > 0` reports a zero (perturbed) pivot: values up to `n` are rows of
  `A`, larger values belong to the reduced interface system.  Factorization
  still completes, so inspect `stats::residual`.
* `solver::dense_interface()` reports whether the reduced interface was
  factorized densely with cuSOLVER (`true`) or in band storage (`false`).  The
  choice is made in the constructor from the interface size and the cost
  model; a zero pivot on the dense path falls back to the banded one.
  The first dense factorization in a process also pays a one-time cuSOLVER
  setup (measured ~30 ms at `m ~= 1800`, ~4 ms at `m ~= 300`), which a
  one-shot solve includes but a reused `solver` object pays only once.
* `solver` keeps the pointers passed to its constructor, so `A` must outlive
  it.  `solve()` uses private scratch space and is not safe to call
  concurrently on the same object; separate objects are independent.
* CUDA or cuBLAS failures throw `std::runtime_error`; bad arguments are
  reported through negative `info` values (or an exception from the `solver`
  constructor).

## Layout

```
include/gspike/spike.h   public API (no CUDA types in the header)
include/gspike/index.h   gspike::index_t (32- or 64-bit, configured by CMake)
src/kernels.cuh          device kernels
src/spike.cu             host side: partitioning, factorization, solve
tests/test_spike.cu      correctness tests
bench/bench.cu           benchmark tool
SPIKE/                   original Fortran reference implementation
```
