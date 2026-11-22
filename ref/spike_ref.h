// C interface to the original Fortran SPIKE-SMP solver (ref/spike_ref.f90).
// This is used only by the benchmark; the gspike library does not depend on it.
#pragma once

extern "C" {

// Factor A (host band storage, lda x n, LAPACK layout).  A is copied and not
// modified.  Returns a context id (>= 1), or -1 on failure.  The measured
// factorization time is stored in factor_ms; parts is the number of level-0
// SPIKE partitions and info is the Fortran info code (0 = success).
int spike_ref_create(int n, int kl, int ku, const double* A, int lda, int nthreads, int* parts, double* factor_ms, int* info);

// Solve with a factorized context.  B (ldb x nrhs) is overwritten by the
// solution and solve_ms receives the measured solve time.
int spike_ref_solve(int id, int nrhs, double* B, int ldb, double* solve_ms);

void spike_ref_destroy(int id);

} // extern "C"
