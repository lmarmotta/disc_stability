module shared_vars

    implicit none

    ! Main property vectors.

    real(8), allocatable, dimension(:,:) :: Q
    real(8), allocatable, dimension(:,:) :: F
    real(8), allocatable, dimension(:,:) :: RHS

end module shared_vars
