!
! This subroutine computes the eigenvalues of the matrix used in 
! Pullians work on artificial dissipation. First stpes towards 
! Stability analysis.
! 
!
subroutine eig(matrix,size_matrix,eigs)
!
    implicit none
!
!   inputs
!
    integer(kind=4), intent(in) :: size_matrix
    complex(kind=8), dimension(size_matrix) :: eigs
    complex(kind=8), dimension(size_matrix,size_matrix) :: matrix
!
!   parameters
!
    integer(kind=4) :: n
    integer(kind=4) :: lda
    integer(kind=4) :: ldvl
    integer(kind=4) :: ldvr
    integer(kind=4) :: lwmax
!
    integer(kind=4) :: i
!
!   local scalars
!
    integer(kind=4) :: info, lwork
!
!   local arrays
!
    real(kind=8), allocatable, dimension(:) :: rwork
!
!   Declare arrays.
!
    complex(kind=8), allocatable, dimension(:,:) :: vl
    complex(kind=8), allocatable, dimension(:,:) :: vr
    complex(kind=8), allocatable, dimension(:)   :: w
    complex(kind=8), allocatable, dimension(:)   :: work
!
!   Define the needed parameters.
!
    n     = size_matrix
    lda   = n
    ldvl  = n
    ldvr  = n
    lwmax = 1000
!
!   Allocate the working matrices.
!
    allocate(vl(ldvl, n))
    allocate(vr(ldvr, n))
    allocate(w(n))
    allocate(work(lwmax))
    allocate(rwork(2*N))
!
!   query the optimal workspace.
!
    lwork = -1
!
    call zgeev( 'vectors', 'vectors', n, matrix, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info )
!
!   Compute the proper size for lwork.
!
    lwork = min( lwmax, int( work( 1 ) ) )
!
!   solve eigenproblem.
!
    call zgeev( 'vectors', 'vectors', n, matrix, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info )
!
!   check for convergence.
!
    if( info.gt.0 ) then
        write(*,*)'the algorithm failed to compute eigenvalues.'
        stop
    end if
!
!   Fill up the answer.
!
    do i = 1, size_matrix
        eigs(i) = w(i)
    end do
!
!   No memory leaks.
!
    deallocate(vl)
    deallocate(vr)
    deallocate(w)
    deallocate(work)
    deallocate(rwork)
!
end subroutine eig

program eigenvalue
!
    implicit none
!
    integer(4) :: i,j
    complex(kind=8) :: eigs(8)
    complex(kind=8) :: a(8,8)
!
!  The matrix is well behaved, so lets start all with zero.
!
    a(:,:) = (0.0d0,0.0d0)
    eigs(:) = (0.0d0,0.0d0)
!
!   First line of the matrix.
!
    a(1,1) = -3.0d0
    a(1,2) =  3.0d0
    a(1,3) = -1.0d0
!
!   Second line of the matrix.
!
    a(2,1) =  4.0d0 
    a(2,2) = -6.0d0
    a(2,3) =  4.0d0 
    a(2,4) = -1.0d0
!
!   Fill three other lines
!
    j = 0

    do i = 3,6


        a(i,j+1) = -1.0d0
        a(i,j+2) =  4.0d0 
        a(i,j+3) = -6.0d0
        a(i,j+4) =  4.0d0 
        a(i,j+5) = -1.0d0

        j = j + 1

    end do

    a(7,5) = -1.0d0
    a(7,6) =  4.0d0
    a(7,7) = -6.0d0
    a(7,8) =  4.0d0

    a(8,6) = -1.0d0
    a(8,7) =  3.0d0
    a(8,8) = -3.0d0
!
!   Solve the eigenvalue problem.
!
    call eig(a,8,eigs)
!
!   Dump the solution.
!
    do i = 1,8
        write(*,'(8(F0.0,SP,F0.0,"i | "))') a(i,1),a(i,2),a(i,3),a(i,4),a(i,5),a(i,6),a(i,7),a(i,8)
    end do 
!
    do i = 1,8
        write(*,*) eigs(i)
    end do
!
end program eigenvalue

