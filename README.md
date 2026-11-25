<div align="center">

# ⚡ `GSPIKE`

**A mixed-precision GPU direct solver for banded linear systems.**
**Factor once. Solve fast. Keep the residual.**

[![CUDA](https://img.shields.io/badge/CUDA-GPU-76B900?logo=nvidia&logoColor=white&style=flat-square)](https://developer.nvidia.com/cuda-zone)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white&style=flat-square)](https://isocpp.org/)
![Dependencies](https://img.shields.io/badge/link%20deps-CUDA%20runtime%20only-success?style=flat-square)

</div>

---

`GSPIKE` is a small C++20 CUDA library for solving $AX=B$ when $A$ is banded.
It is a GPU port of the Fortran SPIKE solver: the matrix is split into independent diagonal blocks, factored in parallel, and the coupling between blocks is solved through a tiny interface system.
Support for `float` and `double` is built in, including a mixed-precision path that factorizes in `float` and recovers near-`double` accuracy with a `double` residual.

## ✨ Why `GSPIKE`?

| | What you get |
|---|---|
| 🚀 **Throughput** | Blocked band LU and a blocked triangular solve that reuses band factors across many right-hand sides at once. |
| 🎯 **Mixed Precision** | Float factorization + double residual + one correction step: near-double accuracy at a fraction of the fp64 cost. |
| 🧠 **Memory-aware** | Full spike vectors are kept automatically when they fit, removing a second band solve from every solve. |
| 🛡️ **Robust** | Partial pivoting with a scalar fallback; non-dominant and asymmetric bands are covered by the test suite. |
| 🧩 **Small** | One header, two scalar types, no dependency beyond the CUDA runtime. `A` is never modified. |
| 🔬 **Observable** | `--profile` reports factorization and solve phase timings; the benchmark reports GFLOP/s and retained device memory. |

## 🚀 Quick Start

```cpp
#include <gspike/spike.h>

// dA, dB live in device memory; A is read-only, B is overwritten with X.
gspike::opts o; // defaults are fine
gspike::stats st;
int info = gspike::spike_gbsv<double>(n, kl, ku, nrhs, dA, lda, dB, ldb, o, &st);
// info == 0 means success
```

Solve many right-hand sides from one factorization:

```cpp
gspike::solver<double> s(n, kl, ku, dA, lda);
s.solve(nrhs, dB, ldb, &st); // call repeatedly
```

Or use the mixed-precision path for a double matrix:

```cpp
gspike::mixed_solver s(n, kl, ku, dA, lda);
s.solve(nrhs, dB, ldb, &st); // two float solves + one double residual
```

### Pick Your Solver

| API | Factorization | Correction | Best for |
|---|---|---|---|
| `spike_gbsv<T>` | once, internally | — | one-off solves |
| `solver<T>` | once, explicitly | optional full-spike matvec | many RHS against the same matrix |
| `mixed_solver` | `float` | `double` residual + float solve | double problems at near-float cost |

## 🧭 How It Works

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

1. Split the matrix into `p` contiguous diagonal blocks and factor each block in parallel with a blocked band LU with partial pivoting (one CUDA block per panel, tiled rank-4 trailing updates).
   `A` is untouched; factors live in a private band buffer in LAPACK's `2*kl + ku + 1` layout.
2. Solve each block against its coupling columns to obtain the spikes `Vp_i = A_ii⁻¹ C_i` and `Vm_i = A_ii⁻¹ B_i` (`2*klu` columns in total).
3. Eliminate the interiors.
   What remains is a `2*klu*(p-1)` block-tridiagonal interface system, assembled in band storage with half-bandwidth `3*klu` and factored once with the same band LU kernel.
4. Back-substitute the interface unknowns, correct the coupling rows of each block, then solve each block.

The partition count is chosen automatically by balancing block parallelism against the cost of the reduced interface system, subject to a memory cap on its factors.
Narrow bands get more partitions; wider bands get fewer.
Every partition keeps enough rows to factor efficiently, and `p == 1` degenerates to a plain parallel band LU.

Measured performance and how to reproduce it live in [bench/README.md](bench/README.md).

## 🛠 Build

Requires CUDA and CMake ≥ 3.24 (for `CMAKE_CUDA_ARCHITECTURES=native`); C++20 is used.
The benchmark builds by default.
The optional CPU reference (Fortran + LAPACK + OpenMP) and the MKL direct-solver head-to-head are enabled automatically when their dependencies are found, and skipped otherwise.
Neither is ever linked into the library; no cuBLAS or cuSPARSE dependency remains.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The top-level `CMakeLists.txt` only builds the library; tests live in `tests/CMakeLists.txt` and benchmarks in `bench/CMakeLists.txt`, and neither is added when `gspike` is a subdirectory of another project.
The library target is `gspike` (alias `gspike::gspike`); the single public header is `include/gspike/spike.h` and the only link dependency is the CUDA runtime.
Only `float` and `double` are instantiated — other scalar types fail at link time rather than silently doing something unexpected.

## 🧪 Tests

`spike_tests` checks both precisions against a dense host LU reference and the relative residual: tridiagonal, diagonally dominant, asymmetric and one-sided bands, wide bands, forced/automatic partitions, multiple RHS, non-dominant matrices with pivoting (including full spikes), warp solves with partially filled terminal blocks, solver reuse, `opts::profile`, argument validation and a randomized fuzz pass.
The mixed solver is checked against the dense reference and a shape/partition sweep.

Benchmark tools, results and how to run them are documented in [bench/README.md](bench/README.md).

## 📌 Good to know

- `opts::full_spikes` costs `2*klu*n` scalars.
  It is auto-enabled only within a device-memory budget; `--no-full-spikes` disables the automatic choice.
  Accuracy is unchanged.
- `opts::profile` prints phase timings to stderr and adds event synchronizations.
- `info > 0` is the first zero (perturbed) pivot: values ≤ `n` are rows of `A`, larger ones belong to the reduced interface system.
  Check the residual.
- `solver` and `mixed_solver` keep the pointer to `A`, so `A` must outlive them; `mixed_solver` reads the original double matrix on every solve for the residual.
  `solve()` uses private scratch and is not thread-safe on one object; separate objects are independent.
- CUDA launch failures throw `std::runtime_error`; asynchronous kernel errors surface at the next synchronization.
  Bad arguments return negative `info` (or throw from the `solver` constructor).
