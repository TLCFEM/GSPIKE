# GSPIKE benchmarks

`spike_bench` measures factorization and solve time, retained device memory and relative residuals for each precision.
`spike_duel` compares four direct solvers on the same matrix: GPU mixed, GPU double, MKL general band LU and the original Fortran SPIKE-SMP solver.
Both build by default in a standalone build; the CPU comparison targets need Fortran, LAPACK and OpenMP, and `spike_duel` also needs MKL.

The figures below are a **snapshot**, not a guarantee.
They were measured from a Release build with `CMAKE_CUDA_ARCHITECTURES=native` on an **NVIDIA L40** (cc 8.9, 142 SMs, 96 MB L2), CUDA 12.6.85, driver 560.35.03, with gfortran 13.3, LAPACK and OpenMP on 8 threads; MKL uses `mkl_rt` with the GNU threading layer.
Matrices are diagonally dominant random bands with `klu ≈ n/100`.
Factorization is timed once, solve is averaged over five solves, and times are in milliseconds.

## How to run

```sh
./build/spike_bench                        # default sweep, double
./build/spike_bench --type float           # default sweep, float
./build/spike_bench --type float16         # default sweep, CUDA __half factorization
./build/spike_bench --type mixed16         # default sweep, __half + 4 double refinements
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

`--type float16` uses CUDA's native `__half` (implemented in the parallel `src/spike_half.cu` / `src/kernels_half.cuh` files, sharing the common kernels in `src/kernels.cuh` (the float/double adapter is `src/kernels_float_double.cuh`)), so it runs on any CUDA build. Its half factorization alone gives ~1e-3 residuals. `--type mixed16` factorizes in `__half` and performs four double-residual refinement iterations; residuals are comparable to the float mixed path (about 1e-13 … 1e-12 in the sweep, 1e-15 on the small well-conditioned test case). The residual is rescaled into the normal range of half before each correction solve, since half subnormals would otherwise stall refinement near 1e-7.
`--profile` adds opt-in phase timings to stderr (it adds synchronizations, so it is not used for the headline numbers).
Default sweeps use bands around 1% of `n` and few RHS; the largest case is `n = 80000, klu = 800`.
`--type cpu` and `--compare` require the CPU reference.
The reference uses OpenMP with power-of-two partitions, so thread counts are rounded down (`--cpu-threads 24` uses 16) and pinned to the allowed CPUs.

```sh
./build/spike_duel                       # default sweep (four direct solvers)
./build/spike_duel 50000 500 2           # n klu nrhs
DUEL_CPU_THREADS=16 ./build/spike_duel   # MKL and OpenMP thread count (default 8)
```

## Direct-solver head-to-head

The optional `spike_duel` benchmark solves the same band matrix with four direct solvers only — no iterative methods.
Total time is factorization + solve; speedups are MKL time / GSPIKE time.

| n | klu | nrhs | GSPIKE mixed | GSPIKE double | MKL double band | Fortran SPIKE | mixed vs MKL | double vs MKL |
|--:|----:|-----:|-------------:|--------------:|----------------:|--------------:|-------------:|--------------:|
| 10000 | 100 | 1 | 14.22 ms | 25.43 ms | 27.84 ms | 8.47 ms | **1.96×** | **1.09×** |
| 25000 | 250 | 1 | 63.21 ms | 107.73 ms | 118.70 ms | 69.88 ms | **1.88×** | **1.10×** |
| 50000 | 500 | 2 | 209.11 ms | 432.93 ms | 551.20 ms | 399.15 ms | **2.64×** | **1.27×** |
| 80000 | 800 | 1 | 570.32 ms | 1438.16 ms | 1520.06 ms | 1353.80 ms | **2.67×** | **1.06×** |

Factor / solve breakdown, in ms:

| n | klu | mixed fac / sol | double fac / sol | MKL fac / sol | Fortran fac / sol |
|--:|----:|----------------:|-----------------:|--------------:|------------------:|
| 10000 | 100 | 11.41 / 2.80 | 23.68 / 1.75 | 26.51 / 1.33 | 8.02 / 0.45 |
| 25000 | 250 | 48.22 / 14.99 | 98.48 / 9.25 | 105.06 / 13.64 | 65.32 / 4.56 |
| 50000 | 500 | 171.51 / 37.61 | 409.22 / 23.70 | 477.74 / 73.46 | 375.84 / 23.31 |
| 80000 | 800 | 488.45 / 81.87 | 1359.99 / 78.16 | 1414.41 / 105.65 | 1302.93 / 50.87 |

The Fortran SPIKE solver is the fastest CPU option at the two smallest sizes, while the GPU mixed path trades a single-precision factorization for a double residual correction and stays ahead of MKL's double band LU in every case.
MKL's band solve is memory-bound and gets much slower as the band widens, while the GPU and Fortran SPIKE solves stay cheap.
Final relative residuals are `~1e-13 … 1e-12` for the mixed path and `~1e-15` for the three double paths.

## Full GPU sweep

| type   |     n | klu | nrhs | parts | factor ms | solve ms | total ms | factor GFLOP/s | solve GFLOP/s | mem MB | rel. residual |
|--------|------:|----:|-----:|------:|----------:|---------:|---------:|---------------:|--------------:|-------:|--------------:|
| double | 10000 | 100 |    1 |     5 |     24.01  |     1.75  |     25.76  |          70.87  |          7.25  |     48  |      7.84e-16 |
| double | 25000 | 250 |    1 |     4 |     99.34  |     9.25  |    108.59  |         258.39  |          8.35  |    278  |      1.22e-15 |
| double | 50000 | 500 |    2 |     4 |    408.76  |    23.70  |    432.47  |         502.18  |         26.06  |   1110  |      1.70e-15 |
| double | 80000 | 800 |    1 |     4 |   1361.11  |    78.17  |   1439.28  |         617.66  |         10.11  |   1852  |      2.14e-15 |
| double | 25000 | 250 |    1 |     1 |    149.90  |    25.59  |    175.49  |          41.28  |          1.45  |    146  |      1.23e-15 |
| float  | 10000 | 100 |    1 |     5 |     11.41  |     1.36  |     12.77  |         149.19  |          9.31  |     24  |      3.53e-07 |
| float  | 25000 | 250 |    1 |     4 |     48.50  |     7.27  |     55.78  |         529.18  |         10.62  |    140  |      5.42e-07 |
| float  | 50000 | 500 |    2 |     4 |    169.34  |    17.84  |    187.18  |        1212.19  |         34.62  |    556  |      7.62e-07 |
| float  | 80000 | 800 |    1 |     4 |    486.14  |    38.48  |    524.62  |        1729.33  |         20.54  |   1416  |      9.56e-07 |
| float  | 25000 | 250 |    1 |     1 |     93.77  |    12.52  |    106.28  |          66.00  |          2.97  |     72  |      5.49e-07 |
| mixed  | 10000 | 100 |    1 |     5 |     11.32  |     2.80  |     14.12  |         150.35  |         10.48  |     32  |      1.37e-13 |
| mixed  | 25000 | 250 |    1 |     4 |     48.38  |    15.04  |     63.41  |         530.56  |         11.94  |    188  |      3.25e-13 |
| mixed  | 50000 | 500 |    2 |     4 |    170.11  |    37.62  |    207.72  |        1206.73  |         38.15  |    750  |      6.41e-13 |
| mixed  | 80000 | 800 |    1 |     4 |    487.32  |    81.86  |    569.17  |        1725.16  |         22.43  |   1908  |      1.00e-12 |

`parts = 1` is the plain parallel band LU path (SPIKE disabled).
`mem MB` is the device memory retained by the solver plus solve scratch, excluding the input matrix.
Full spikes are enabled automatically for the float/mixed rows and for double up to 50k/500; at 80k/800 in double the 512 MB auto cap excludes it.

### What the automatic full-spike correction buys

With `--no-full-spikes`, the second band solve is used instead of the saved spike matvec (same factorizations).
The saved matvec makes solves 1.2–1.7× faster here at roughly 1.3× the retained memory:

| n | klu | nrhs | parts | mixed solve, auto | mixed solve, no full | mem MB, auto / no |
|--:|----:|-----:|------:|------------------:|---------------------:|------------------:|
| 10000 | 100 | 1 | 5 | 2.80 ms | 4.63 ms | 32 / 24 |
| 25000 | 250 | 1 | 4 | 15.04 ms | 18.39 ms | 188 / 140 |
| 50000 | 500 | 2 | 4 | 37.62 ms | 61.31 ms | 750 / 558 |
| 80000 | 800 | 1 | 4 | 81.86 ms | 132.64 ms | 1908 / 1418 |

## Original Fortran SPIKE-SMP reference

`spike_bench --compare` runs the original Fortran solver with 8 OpenMP threads on a 24-core AMD EPYC 9334.
`factor x`, `solve x`, `total x` are CPU/GPU time; values above 1 favour the GPU.

| n | klu | nrhs | CPU parts | GPU parts | CPU factor ms | GPU factor ms | factor x | CPU solve ms | GPU solve ms | solve x | CPU total ms | GPU total ms | total x |
|--:|----:|-----:|----------:|----------:|--------------:|--------------:|---------:|-------------:|-------------:|--------:|-------------:|-------------:|--------:|
| 10000 | 100 | 1 | 8 | 5 | 13.18 | 11.32 | 1.16 | 0.52 | 2.80 | 0.19 | 13.71 | 14.13 | 0.97 |
| 25000 | 250 | 1 | 8 | 4 | 65.68 | 48.37 | 1.36 | 5.11 | 15.04 | 0.34 | 70.79 | 63.41 | 1.12 |
| 50000 | 500 | 2 | 8 | 4 | 373.77 | 171.06 | 2.19 | 23.81 | 37.61 | 0.63 | 397.58 | 208.67 | 1.91 |
| 80000 | 800 | 1 | 8 | 4 | 1323.29 | 489.93 | 2.70 | 51.67 | 81.86 | 0.63 | 1374.96 | 571.80 | 2.40 |

Bands are around 1% of `n`, so factorization dominates.
The GPU triangular solve is slower for a single RHS because substitution is serial, but it is a small part of the total: GPU mixed wins overall from `n = 25000`.

The original Fortran solver takes its panel width from `ILAENV`, factorizes a panel with BLAS2, then applies it with `DTRSM`/`DGEMM`; the partition loop and recursive reduced system use OpenMP.
CMake links the best LAPACK/BLAS it finds — **with Intel MKL the same factorization is ~0.34 s instead of ~3.9 s with netlib BLAS** at `n = 50000, klu = 500`.
Eight OpenMP threads beat sixteen: each worker gets two balanced partitions and the last reduced-system levels suffer less from idle workers.

Against MKL's own general band LU (`dgbtrf`/`dgbtrs`, 8 threads):

| n | klu | Fortran SPIKE factor | MKL `dgbtrf` | speedup |
|--:|----:|---------------------:|-------------:|--------:|
| 10000 | 100 | 8.0 ms | 26.5 ms | 3.3× |
| 25000 | 250 | 65.3 ms | 105.1 ms | 1.6× |
| 50000 | 500 | 375.8 ms | 477.7 ms | 1.3× |
| 80000 | 800 | 1302.9 ms | 1414.4 ms | 1.1× |

Both use MKL BLAS underneath; the gain comes from SPIKE's partitioning.

## Large-scale benchmark (~6 GB matrix)

An out-of-sweep case with `n = 200000, klu = 2000` and one RHS.
The input matrix alone occupies `lda × n × 8 = 4001 × 200000 × 8` bytes = **6.4 GB (5.96 GiB)** in double-precision LAPACK band storage.

```sh
./build/spike_duel 200000 2000 1
./build/spike_bench --type mixed 200000 2000 1
./build/spike_bench --type double 200000 2000 1
./build/spike_bench --compare 200000 2000 1
```

Four direct solvers (factor / solve / total in ms):

| method | parts | factor ms | solve ms | total ms | relres |
|---|---:|---:|---:|---:|---:|
| gspike mixed (GPU) | 2 | 7329 | 983 | 8313 | 2.49e-12 |
| gspike double (GPU) | 1 | 10874 | 546 | 11420 | 3.51e-15 |
| MKL double band (CPU) | — | 10761 | 502 | 11263 | 4.09e-15 |
| Fortran SPIKE (CPU) | 8 | 19274 | 500 | 19774 | 3.61e-15 |

Against MKL's double band LU, GPU mixed is **1.35×** faster overall and GPU double is on par (1.01× MKL's time); Fortran SPIKE is the slowest at this size (1.76× MKL's time).
The mixed factorization sustains 1720 GFLOP/s and the solver retains 8106 MB of device memory (double: 9352 MB).

`spike_bench --compare` measures the same Fortran SPIKE solver directly: factor 21529 ms, solve 939 ms, total 22468 ms — **2.70×** slower than GPU mixed.
The interface system is too large for the 320 MB cap at this bandwidth, so the automatic partitioner selects `parts = 2` for the mixed solver and falls back to plain band LU (`parts = 1`) for double.

## Matrices with more elements than `INT_MAX` (~2.2 G elements)

Measured on the L40 in `float`: `n = 1,100,000`, `kl = ku = 1000`, `nrhs = 1`.
`lda = 2001` → `A` has **2,201,100,000 elements** (> 2,147,483,647), 8.8 GB.
The private band-factor buffer has `lda_f = 3001` → 3,301,100,000 elements (13.2 GB).

```sh
./build/spike_bench --type float 1100000 1000 1          # auto: 5 partitions
./build/spike_bench --type float 1100000 1000 1 1        # plain band LU
./build/spike_bench --type float 1100000 1000 1 2        # two SPIKE partitions
./build/spike_bench --type float --full-spikes 1100000 1000 1
```

| mode | parts | factor ms | solve ms | retained MB | relres |
|---|---:|---:|---:|---:|---:|
| auto | 5 | 17211 | 639 | 13008 | 1.07e-06 |
| forced plain LU | 1 | 11860 | 1527 | 12638 | 1.07e-06 |
| forced SPIKE | 2 | 35674 | 1549 | 12732 | 1.07e-06 |
| auto + full spikes | 5 | 17619 | 354 | 21402 | 1.07e-06 |

## Profiling and optimization notes

- **Blocked spike solve.** Band factors are loaded once per 64 columns instead of once per column: the spike phase at `n = 50000, klu = 500` fell from **187 ms to ~43 ms** in fp32, taking factorization from ~312 ms to ~170 ms.
- **Phase visibility.** `--profile` at `n = 50000, klu = 500, nrhs = 2, parts = 4` (mixed) reports, in ms: pack 7.0, band LU 81.3, reciprocal diagonal 0.05, spikes 43.1, reduced LU 32.4; a solve takes 12.5 (first band solve), 0.04 (interface RHS), 4.5 (reduced solve), 0.8 (full-spike correction).
  The reduced system is still a single CUDA block — the next scaling target.
