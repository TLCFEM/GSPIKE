! C-callable wrapper around the original double-precision SPIKE-SMP solver.
! Only the benchmark uses this; the CUDA library never links it.
!
! The original code factorizes A in place, so create() keeps a private band
! copy.  The context array avoids exposing Fortran derived types through the
! C ABI.
module spike_ref_mod
    use iso_c_binding
    use omp_lib
    implicit none
    private

    integer, parameter :: max_ctx = 16

    type :: ref_ctx
        logical :: used = .false.
        integer :: n = 0, kl = 0, ku = 0, lda = 0, parts = 0
        integer :: spm(64) = 0
        real(c_double), allocatable :: C(:, :)
        real(c_double), allocatable :: work(:)
    end type

    type(ref_ctx), save :: ctxs(max_ctx)

    interface
        subroutine spike_pin_thread(i) bind(C, name="spike_pin_thread")
            import :: c_int
            integer(c_int), value :: i
        end subroutine spike_pin_thread
    end interface

contains

    integer(c_int) function find_free() result(id)
        integer :: i
        id = -1
        do i = 1, max_ctx
            if(.not. ctxs(i)%used) then
                id = i
                return
            end if
        end do
    end function find_free

    ! id = spike_ref_create(n, kl, ku, A, lda, nthreads, parts, factor_ms, info)
    ! A is host band storage (lda x n, LAPACK layout) and is not modified.
    integer(c_int) function spike_ref_create(n, kl, ku, A, lda, nthreads, parts, factor_ms, info) &
            bind(C, name="spike_ref_create") result(id)
        implicit none
        integer(c_int), value :: n, kl, ku, lda, nthreads
        real(c_double), intent(in) :: A(*)
        integer(c_int), intent(out) :: parts, info
        real(c_double), intent(out) :: factor_ms

        integer :: i, j, klu, lda_i, nth, slot, infoloc, p2
        real(c_double) :: t0, t1

        id = -1
        parts = 0
        info = 0
        factor_ms = 0.0_c_double
        if(n < 1 .or. kl < 0 .or. ku < 0 .or. lda < kl + ku + 1) then
            info = -1
            return
        end if

        slot = find_free()
        if(slot < 0) then
            info = -2
            return
        end if

        klu = max(kl, ku)
        nth = nthreads
        if(nth <= 0) nth = omp_get_max_threads()
        ! SPIKE partitions are a power of two.  Asking for more threads than
        ! partitions leaves extra workers without a partition, so round down.
        p2 = 1
        do while(p2*2 <= nth)
            p2 = p2*2
        end do
        nth = p2
        call omp_set_num_threads(nth)
        ! Pin the pool once; libgomp reuses these threads for later regions.
        !$omp parallel num_threads(nth)
        call spike_pin_thread(omp_get_thread_num())
        !$omp end parallel

        lda_i = kl + ku + 1
        ctxs(slot)%used = .true.
        ctxs(slot)%n = n
        ctxs(slot)%kl = kl
        ctxs(slot)%ku = ku
        ctxs(slot)%lda = lda_i

        call spikeinit_nthread(ctxs(slot)%spm, n, klu, nth)
        ctxs(slot)%spm(1) = 0 ! no printing
        ctxs(slot)%spm(2) = 0 ! no auto tuning
        ctxs(slot)%spm(3) = 0 ! no pivoting
        ctxs(slot)%parts = ctxs(slot)%spm(20)

        allocate(ctxs(slot)%C(lda_i, n))
        do j = 1, n
            do i = 1, lda_i
                ctxs(slot)%C(i, j) = A((j - 1)*lda + i)
            end do
        end do
        allocate(ctxs(slot)%work(klu*klu*ctxs(slot)%spm(10)))

        t0 = omp_get_wtime()
        call dspike_gbtrf(ctxs(slot)%spm, n, kl, ku, ctxs(slot)%C, lda_i, ctxs(slot)%work, infoloc)
        t1 = omp_get_wtime()

        factor_ms = (t1 - t0)*1000.0_c_double
        parts = ctxs(slot)%parts
        info = infoloc
        id = slot
    end function spike_ref_create

    integer(c_int) function spike_ref_solve(id, nrhs, B, ldb, solve_ms) &
            bind(C, name="spike_ref_solve") result(info)
        implicit none
        integer(c_int), value :: id, nrhs, ldb
        real(c_double), intent(inout) :: B(ldb, *)
        real(c_double), intent(out) :: solve_ms

        real(c_double) :: t0, t1

        info = -3
        solve_ms = 0.0_c_double
        if(id < 1 .or. id > max_ctx) return
        if(.not. ctxs(id)%used) return
        if(nrhs < 0) return
        if(ldb < ctxs(id)%n) return
        if(nrhs == 0) then
            info = 0
            return
        end if

        t0 = omp_get_wtime()
        call dspike_gbtrs(ctxs(id)%spm, 'n', ctxs(id)%n, ctxs(id)%kl, ctxs(id)%ku, &
                          nrhs, ctxs(id)%C, ctxs(id)%lda, ctxs(id)%work, B, ldb)
        t1 = omp_get_wtime()

        solve_ms = (t1 - t0)*1000.0_c_double
        info = 0
    end function spike_ref_solve

    subroutine spike_ref_destroy(id) bind(C, name="spike_ref_destroy")
        implicit none
        integer(c_int), value :: id

        if(id < 1 .or. id > max_ctx) return
        if(.not. ctxs(id)%used) return
        if(allocated(ctxs(id)%C)) deallocate(ctxs(id)%C)
        if(allocated(ctxs(id)%work)) deallocate(ctxs(id)%work)
        ctxs(id)%used = .false.
    end subroutine spike_ref_destroy

end module spike_ref_mod
