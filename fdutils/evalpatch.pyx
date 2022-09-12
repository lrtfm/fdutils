# cython: language_level=3

# copy from file firedrake/cython/dmcommon.pyx
import cython
import numpy as np
from firedrake.petsc import PETSc
from mpi4py import MPI

# from firedrake.utils import IntType, ScalarType
# from libc.string cimport memset
# from libc.stdlib cimport qsort

cimport numpy as np
cimport mpi4py.MPI as MPI
cimport petsc4py.PETSc as PETSc


np.import_array()


include "evalpatchinc.pxi"


def build_two_sided(MPI.Comm comm, count, MPI.Datatype dtype,
                    np.ndarray[int, ndim=1, mode="c"] toranks,
                    np.ndarray[int, ndim=1, mode="c"] todata):
    cdef:
        int *fromranks
        int nto, nfrom
        int ccount = count
        int i
        int *fromdata
        np.ndarray[int, ndim=1, mode="c"] pyfromdata
        np.ndarray[int, ndim=1, mode="c"] pyfromranks

    nto = len(toranks)
    CHKERR(PetscCommBuildTwoSided(comm.ob_mpi, ccount,
                                  dtype.ob_mpi, nto,
                                  <const int *>toranks.data,
                                  <const void *>todata.data,
                                  &nfrom, &fromranks,
                                  <void *>&fromdata))
    pyfromdata = np.empty(nfrom, dtype=todata.dtype)
    pyfromranks = np.empty(nfrom, dtype=toranks.dtype)
    for i in range(nfrom):
        pyfromdata[i] = fromdata[i]
        pyfromranks[i] = fromranks[i]
    CHKERR(PetscFree(fromdata))
    CHKERR(PetscFree(fromranks))
    return pyfromranks, pyfromdata

def get_memory_usage():
    cdef:
        double mal
        double mem

    CHKERR(PetscMallocGetCurrentUsage(&mal))
    CHKERR(PetscMemoryGetCurrentUsage(&mem))

    return mal, mem
