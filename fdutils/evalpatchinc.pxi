# copy from file firedrake/cython/petschdr.pxi  
cimport petsc4py.PETSc as PETSc
cimport mpi4py.MPI as MPI
cimport numpy as np

cdef extern from "mpi-compat.h":
    pass

IF COMPLEX:
   ctypedef np.complex128_t PetscScalar
ELSE:
   ctypedef double PetscScalar

cdef extern from "petsc.h":
   ctypedef long PetscInt
   ctypedef double PetscReal
   ctypedef double PetscLogDouble
   ctypedef enum PetscBool:
       PETSC_TRUE, PETSC_FALSE
   ctypedef enum PetscCopyMode:
       PETSC_COPY_VALUES,
       PETSC_OWN_POINTER,
       PETSC_USE_POINTER

cdef extern from "petscsys.h" nogil:
   int PetscMalloc1(PetscInt,void*)
   int PetscFree(void*)
   int PetscSortIntWithArray(PetscInt,PetscInt[],PetscInt[])
   int PetscCommBuildTwoSided(MPI.MPI_Comm,int,MPI.MPI_Datatype,int,const int *, const void *,
                              int *, int **, void *)

   int PetscMallocGetCurrentUsage(PetscLogDouble*)
   int PetscMemoryGetCurrentUsage(PetscLogDouble*)

    
# --- Error handling taken from petsc4py (src/PETSc.pyx) -------------

cdef extern from *:
    void PyErr_SetObject(object, object)
    void *PyExc_RuntimeError

cdef object PetscError = <object>PyExc_RuntimeError

cdef inline int SETERR(int ierr) with gil:
    if (<void*>PetscError) != NULL:
        PyErr_SetObject(PetscError, <long>ierr)
    else:
        PyErr_SetObject(<object>PyExc_RuntimeError, <long>ierr)
    return ierr

cdef inline int CHKERR(int ierr) nogil except -1:
    if ierr == 0:
        return 0 # no error
    else:
        SETERR(ierr)
        return -1
