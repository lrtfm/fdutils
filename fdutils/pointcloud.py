from collections import OrderedDict

import os
import sys
import numpy as np

from firedrake.mesh import spatialindex
# from firedrake.dmplex import build_two_sided
from firedrake.utils import cached_property
from firedrake.petsc import PETSc
from pyop2 import op2
from pyop2.mpi import MPI
from pyop2.datatypes import IntType, RealType, ScalarType
from pyop2.profiling import timed_region
import logging

from ctypes import POINTER, c_int, c_double, c_void_p
c_petsc_int = np.ctypeslib.as_ctypes_type(IntType)

from firedrake.function import _CFunction, PointNotInDomainError
import firedrake.utils as utils

from mpi4py.util.dtlib import to_numpy_dtype as mpi_to_numpy_dtype
# this should same with PetscMPIInt: i.e. int32
# Ref: https://petsc.org/release/docs/manualpages/Sys/PetscMPIInt/
MPIIntType = mpi_to_numpy_dtype(MPI.INT)

try:
    from fdutils.evalpatch import build_two_sided
except ModuleNotFoundError:
     raise
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise

__all__ = ["PointCloud"]


# TODO: move to a file?
logger = logging.getLogger("fdutils")
for handler in logger.handlers:
    logger.removeHandler(handler)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(fmt="%(name)s:%(levelname)s %(message)s"))
logger.addHandler(handler)

@PETSc.Log.EventDecorator()
def syncPrint(*args, **kwargs):
    """Perform a PETSc syncPrint operation with given arguments if the logging level is
    set to at least debug.
    """
    if logger.isEnabledFor(logging.DEBUG):
        PETSc.Sys.syncPrint(*args, **kwargs)

@PETSc.Log.EventDecorator()
def Print(*args, **kwargs):
    if logger.isEnabledFor(logging.DEBUG):
        PETSc.Sys.Print(*args, **kwargs)

@PETSc.Log.EventDecorator()
def syncFlush(*args, **kwargs):
    """Perform a PETSc syncFlush operation with given arguments if the logging level is
    set to at least debug.
    """
    if logger.isEnabledFor(logging.DEBUG):
        PETSc.Sys.syncFlush(*args, **kwargs)


class PointCloud(object):
    """Store points for repeated location in a mesh. Facilitates lookup and evaluation
    at these point locations.

    Most code of this class are copied form firedrake repo.

    This class is used to evaluate functions for many times on
    a group of points. It works in case that you give different points
    on different mpi rank. This is an alternative solution before
    `VertexOnlyMesh` in Firedrake supporting this function.

    Exmaple code:
        ```
        from firedrake import *
        from fdutils import PointCloud

        mesh = RectangleMesh(10, 10, 1, 1)
        x, y = SpatialCoordinate(mesh)
        V = FunctionSpace(mesh, 'CG', 1)
        f1 = Function(V).interpolate(x**2 + y**2)
        f2 = Function(V).interpolate(sin(x) + y**2)

        mesh2 = RectangleMesh(20, 20, 1, 1)
        points = mesh2.coordinates.dat.data_ro

        pc = PointCloud(mesh, points, tolerance=1e-12)
        v1 = pc.evaluate(f1)
        v2 = pc.evaluate(f2)
        ```
    """

    def __init__(self, mesh, points, tolerance=None, *args, **kwargs):
        """Initialise the PointCloud.

        :arg mesh: A mesh object.
        :arg points: An N x mesh.geometric_dimension() array of point locations.
        """
        self.mesh = mesh
        points = np.asarray(points, dtype=ScalarType)
        if utils.complex_mode:
            if not np.allclose(points.imag, 0):
                raise ValueError("Provided points have non-zero imaginary part")
            self.points = points.real.copy()
        else:
            self.points = points
        syncPrint('[%d]'%mesh.comm.rank, points)
        self.tolerance = tolerance if tolerance is not None else 1e-12
        _, dim = points.shape
        if dim != mesh.geometric_dimension():
            raise ValueError("Points must be %d-dimensional, (got %d)" %
                             (mesh.geometric_dimension(), dim))

        # Initialise dictionary to store location statistics for evaluation.
        self.statistics = OrderedDict()
        # Initialise counters.
        self.statistics["num_messages_sent"] = 0
        self.statistics["num_points_evaluated"] = 0
        self.statistics["num_points_found"] = 0

        self.points_not_found_indices = None

    @cached_property
    def locations(self):
        """Determine the process rank and element that holds each input point.

        The location algorithm works as follows:
        1. Query local spatial index for list of input points (`self.points`) to determine
           if each is held locally.
        2. For each point not found locally, query spatial index of processes to
           identify those that may contain it. Location requests will be sent to each
           process for the points they may contain.
        3. Perform communication round so that each process knows how many points it will
           receive from the other processes.
        4. Perform sparse communication round to receive points.
        5. Lookup these points in local spatial index and obtain result.
        6. Perform sparse communication round to return results.
        7. Process responses to obtain final results array. If multiple processes report
           containing the same point, choose the process with the lower rank.

        :returns: An array of (rank, cell number) pairs; (-1, -1) if not found.
        """
        rank_cell_pairs = self._locate_mesh_elements(tolerance=self.tolerance)
        return np.array(rank_cell_pairs, dtype=IntType)

    def clear_locations(self):
        """Reset the :attr:`locations` on this mesh geometry.
        Use this if you move the mesh (for example by reassigning to
        the coordinate field)."""
        # TODO: should call it here?
        self.mesh.clear_spatial_index()
        try:
            del self.locations
        except AttributeError:
            pass

    @PETSc.Log.EventDecorator()
    def _get_candidate_processes(self, point):
        """Determine candidate processes for a given point.

        :arg point: A point on the mesh.

        :returns: A numpy array of candidate processes.
        """
        candidates = spatialindex.bounding_boxes(self.mesh.processes_spatial_index, point)
        return candidates[candidates != self.mesh.comm.rank]

    @PETSc.Log.EventDecorator()
    def _perform_sparse_communication_round(self, recv_buffers, send_data):
        """Perform a sparse communication round in the point location process.

        :arg recv_buffers: A dictionary where the keys are process ranks and the
             corresponding items are numpy arrays of buffers in which to receive data
             from that rank.
        :arg send_data: A dictionary where the keys are process ranks and the
             corresponding items are lists of buffers from which to send data to that
             rank.
        """
        # Create lists to hold send and receive communication request objects.
        recv_reqs = []
        send_reqs = []

        for rank, buffers in recv_buffers.items():
            req = self.mesh.comm.Irecv(buffers, source=rank)
            recv_reqs.append(req)

        for rank, points in send_data.items():
            req = self.mesh.comm.Isend(points, dest=rank)
            send_reqs.append(req)
            self.statistics["num_messages_sent"] += 1

        MPI.Request.Waitall(recv_reqs + send_reqs)

    @PETSc.Log.EventDecorator()
    def _locate_mesh_elements(self, tolerance=None):
        """Determine the location of each input point using the algorithm described in
        `self.locations`.

        :returns: A numpy array of (rank, cell number) pairs; (-1, -1) if not found.
        """

        # Create an array of (rank, element) tuples for storing located
        # elements.
        located_elements = np.full((len(self.points), 2), -1, dtype=IntType)

        # Check if points are located locally.
        with timed_region("LocalLocation"):
            with timed_region("LocalQuerying"):
                # Evaluate all points locally.
                local_results = self.mesh.locate_cells(self.points, tolerance=tolerance)
                self.statistics["num_points_evaluated"] += len(self.points)

                # Update points that have been found locally.
                located_elements[:, 1] = local_results
                located_elements[local_results != -1, 0] = self.mesh.comm.rank

                # Store the points and that have not been found locally, and the indices
                # of those points in `self.points`.
                not_found_indices, = np.where(local_results == -1)
                points_not_found = self.points[not_found_indices]

            with timed_region("CandidateIdentification"):
                # Create dictionaries for storing processes that may contain these points.
                local_candidates = {}
                local_candidate_indices = {}

                for point, idx in zip(points_not_found, not_found_indices):
                    # Point not found locally -- get candidates from processes spatial
                    # index.
                    point_candidates = self._get_candidate_processes(point)
                    for candidate in point_candidates:
                        local_candidates.setdefault(candidate, []).append(point)
                        local_candidate_indices.setdefault(candidate, []).append(idx)

                    # syncPrint("[%d] Cell not found locally for point %s. Candidates: %s"
                    #           % (self.mesh.comm.rank, point, point_candidates),
                    #           comm=self.mesh.comm)

            syncFlush(comm=self.mesh.comm)

        with timed_region("StatisticsInfoA"):
            self.statistics["num_points_found"] += \
                np.count_nonzero(located_elements[:, 0] != -1)
            self.statistics["local_found_frac"] = \
                self.statistics["num_points_found"] / (len(self.points) or 1)  # fix ZeroDivisionError: division by zero
            self.statistics["num_candidates"] = len(local_candidates)

        # with timed_region("DebugPrintLocateInfoA"):
        #     syncPrint("[%d] Located elements: %s" % (self.mesh.comm.rank,
        #                                              located_elements.tolist()),
        #               comm=self.mesh.comm)
        #     syncFlush(comm=self.mesh.comm)
        #     syncPrint("[%d] Local candidates: %s" % (self.mesh.comm.rank, local_candidates),
        #               comm=self.mesh.comm)
        #     syncFlush(comm=self.mesh.comm)

        # Exchange data for point requests.
        with timed_region("PointExchange"):
            # First get number of points to receive from each rank through sparse
            # communication round.

            # Create input arrays from candidates dictionary.
            to_ranks = np.zeros(len(local_candidates), dtype=MPIIntType)
            to_data = np.zeros(len(local_candidates), dtype=MPIIntType)
            for i, (rank, points) in enumerate(local_candidates.items()):
                to_ranks[i] = rank
                to_data[i] = len(points)

            # `build_two_sided` provides an interface for PetscCommBuildTwoSided, which
            # facilitates a sparse communication round between the processes to identify
            # which processes will be sending points and how many points they wish to
            # send.
            # The output array `from_ranks` holds the ranks of the processes that will be
            # sending points, and the corresponding element in the `from_data` array
            # specifies the number of points that will be sent.
            from_ranks, from_data = build_two_sided(self.mesh.comm, 1, MPI.INT, to_ranks,
                                                    to_data)

            # Create dictionary to hold all receive buffers for point requests from
            # each process.
            recv_points_buffers = {}
            for i in range(0, len(from_ranks)):
                recv_points_buffers[from_ranks[i]] = np.empty(
                    (from_data[i], self.mesh.geometric_dimension()), dtype=RealType)

            # Receive all point requests

            local_candidates = {r: np.asarray(p) for r, p in local_candidates.items()}
            self._perform_sparse_communication_round(recv_points_buffers,
                                                     local_candidates)

        with timed_region("DebugPrintLocateInfoB"):
            syncPrint("[%d] Point queries requested: %s" % (self.mesh.comm.rank,
                                                            str(recv_points_buffers)),
                      comm=self.mesh.comm)
            syncFlush(comm=self.mesh.comm)

        # Evaluate all point requests and prepare responses
        with timed_region("RemoteLocation"):
            # Create dictionary to store results.
            point_responses = {}

            # Evaluate results.
            for rank, points_buffers in recv_points_buffers.items():
                point_responses[rank] = np.array(
                    self.mesh.locate_cells(points_buffers, tolerance=tolerance), dtype=IntType)
                self.statistics["num_points_evaluated"] += len(points_buffers)
                self.statistics["num_points_found"] += \
                    np.count_nonzero(point_responses[rank] != -1)

        with timed_region("DebugPrintLocateInfoC"):
            syncPrint("[%d] Point responses: %s" % (self.mesh.comm.rank,
                                                    str(point_responses)),
                      comm=self.mesh.comm)
            syncFlush(comm=self.mesh.comm)

        # Receive all responses
        with timed_region("ResultExchange"):
            # Create dictionary to hold all output buffers indexed by rank.
            recv_results = {}
            # Initialise these.
            for rank, points in local_candidates.items():
                # Create receive buffer(s).
                recv_results[rank] = np.empty((len(points), 1), dtype=IntType)

            self._perform_sparse_communication_round(recv_results, point_responses)

        with timed_region("DebugPrintLocateInfoD"):
            syncPrint("[%d] Point location request results: %s" % (self.mesh.comm.rank,
                                                                   str(recv_results)),
                      comm=self.mesh.comm)
            syncFlush(comm=self.mesh.comm)

        # Process and return results.
        with timed_region("ResultProcessing"):
            # Iterate through all points. If they have not been located locally,
            # iterate through each point request reponse to find the element.
            # Sometimes an element can be reported as found by more than one
            # process -- in this case, choose the process with the lower rank.
            for rank, result in recv_results.items():
                indices = local_candidate_indices[rank]
                found, _ = np.where(result != -1)
                for idx in found:
                    i = indices[idx]
                    loc_rank = located_elements[i, 0]
                    if loc_rank == -1 or rank < loc_rank:
                        located_elements[i, :] = (rank, result[idx][0])

        with timed_region("DebugPrintLocateInfoE"):
            syncPrint("[%d] Located elements: %s" % (self.mesh.comm.rank,
                                                     located_elements.tolist()),
                      comm=self.mesh.comm)
            syncFlush(comm=self.mesh.comm)

        with timed_region("GetNotFoundInfo"):
            points_not_found_indices = np.where(located_elements[:, 1] == -1)[0]
            self.points_not_found_indices = points_not_found_indices
            if len(points_not_found_indices) > 0:
                logger.warning('[%2d/%2d] PointCloud._locate_mesh_elements: %d points not located!'%(
                    self.mesh.comm.rank, self.mesh.comm.size, len(points_not_found_indices)))
        # PETSc.Sys.syncFlush()

        return located_elements

    @cached_property
    @PETSc.Log.EventDecorator()
    def evaluate_info(self):
        loc = self.locations
        rank = self.mesh.comm.rank
        size = self.mesh.comm.size

        local_info = None

        rank2cells = {}
        n_all = 0
        for i in range(size):
            index = np.where(loc[:, 0] == i)[0]
            cells = loc[index, 1]
            n_all += len(np.where(cells != -1)[0])
            if i == rank:
                local_info = (index, cells)
            elif len(index) > 0:
                rank2cells[i] = (index, cells)

        with timed_region("EvalPointExchange"):
            to_ranks = np.zeros(len(rank2cells), dtype=MPIIntType)
            to_data = np.zeros(len(rank2cells), dtype=MPIIntType)
            for i, (r, pair) in enumerate(rank2cells.items()):
                if r != rank:
                    to_ranks[i] = r
                    to_data[i] = len(pair[0])

            from_ranks, from_data = build_two_sided(self.mesh.comm, 1, MPI.INT, to_ranks,
                                                    to_data)

            # Create dictionary to hold all receive buffers for point requests from
            # each process.
            recv_cells_buffers = {}
            recv_points_buffers = {}

            for i in range(0, len(from_ranks)):
                recv_cells_buffers[from_ranks[i]] = np.empty(from_data[i], dtype=IntType)
                recv_points_buffers[from_ranks[i]] = np.empty(
                    (from_data[i], self.mesh.geometric_dimension()), dtype=RealType)

            # Receive all point requests

            send_cells = {r: np.asarray(cells, dtype=IntType) for r, (index, cells) in rank2cells.items()}
            send_points = {r: np.asarray(self.points[index, :]) for r, (index, cells) in rank2cells.items()}
            self._perform_sparse_communication_round(recv_cells_buffers,
                                                     send_cells)
            self._perform_sparse_communication_round(recv_points_buffers,
                                                     send_points)
        rank2cells[rank] = local_info
        recv_cells_buffers[rank] = local_info[1]
        recv_points_buffers[rank] = self.points[local_info[0], :]

        return (recv_cells_buffers, recv_points_buffers, rank2cells)

    @PETSc.Log.EventDecorator()
    def evaluate(self, function, callback=None):
        """Evaluate a function at the located points.
        :arg function: The function to evaluate.
        """

        if function.ufl_domain() != self.mesh:
            raise('The function must be defined on the mesh of this PointCloud!')

        rank = self.mesh.comm.rank
        size = self.mesh.comm.size

        # must do this!
        from pyop2 import op2
        function.dat.global_to_local_begin(op2.READ)
        function.dat.global_to_local_end(op2.READ)

        recv_cells_buffers, recv_points_buffers, rank2cells = self.evaluate_info

        with timed_region("Eval"):
            values = {}
            for r, cells in recv_cells_buffers.items():
                ps = recv_points_buffers[r]
                values[r] = batch_eval(function, cells, ps, tolerance=self.tolerance)

        with timed_region("PrepareBuffers"):
            n = len(self.points)
            m = np.prod(function.ufl_shape, dtype=IntType)
            array_shape = lambda number: number if m == 1 else [number, m]
            ret = np.zeros(array_shape(n), dtype=ScalarType)
            if m == 1:
                ret[rank2cells[rank][0]] = values[rank]
            else:
                ret[rank2cells[rank][0], :] = values[rank]

            recv_values_buffers = {}
            for r, pair in rank2cells.items():
                if r != rank:
                    recv_values_buffers[r] = np.empty(array_shape(len(pair[0])), dtype=ScalarType)

            values.pop(rank)

        with timed_region("EvalResultExchange"):
            self._perform_sparse_communication_round(recv_values_buffers, values)

        for r, v in recv_values_buffers.items():
            if m == 1:
                ret[rank2cells[r][0]] = v
            else:
                ret[rank2cells[r][0], :] = v

        with timed_region("Callback"):
            # Notes: Make sure every process call callback for parallell case.
            #        How to do it more reasonable?
            if len(self.points_not_found_indices) > 0:
                num_points = len(self.points_not_found_indices)
                points_not_found = self.points[self.points_not_found_indices, :]
                if callback is not None:
                    # TODO: sync to the first process and then print?
                    logger.warning('[%2d/%2d] PointCloud.evaluate: %d points not located, the callback is called!'\
                                    %(rank, size, num_points))
                else:
                    logger.warning('[%2d/%2d] PointCloud.evaluate: %d points not located, the values are set to zero!'\
                                    %(rank, size, num_points))
            else:
                num_points = 0
                points_not_found = np.zeros([0, self.points.shape[1]], dtype=RealType)
            if callback is not None:
                if m == 1:
                    ret[self.points_not_found_indices] = callback(points_not_found)
                else:
                    ret[self.points_not_found_indices, :] = callback(points_not_found)

        return ret


    @cached_property
    @PETSc.Log.EventDecorator()
    def restriction_info(self):
        recv_cells_buffers, recv_points_buffers, rank2cells = self.evaluate_info
        Xs = {}
        for r, cells in recv_cells_buffers.items():
            ps = recv_points_buffers[r]
            Xs[r] = batch_area_coordinates(self.mesh.coordinates, cells, ps, tolerance=self.tolerance)
        return recv_cells_buffers, rank2cells, Xs


    @PETSc.Log.EventDecorator()
    def restrict(self, pvs, function):
        """Restriction point values to fuction.
        :arg pvs: value on points
             function: The function hold result values.
        """

        if function.ufl_domain() != self.mesh:
            raise('The function must be defined on the mesh of this PointCloud!')

        rank = self.mesh.comm.rank
        size = self.mesh.comm.size

        recv_cells_buffers, rank2cells, Xs = self.restriction_info

        with timed_region("PreparePointValues"):
            m = np.prod(function.ufl_shape, dtype=IntType)
            array_shape = lambda number: number if m == 1 else [number, m]
            recv_pvs = {}
            send_pvs = {}
            for r in range(size):
                if r != rank:
                    cells = recv_cells_buffers[r]
                    recv_pvs[r] = np.empty(array_shape(len(cells)), dtype=ScalarType)
                    send_pvs[r] = pvs[rank2cells[r][0]]

        with timed_region("PointValuesExchange"):
            self._perform_sparse_communication_round(recv_pvs, send_pvs)

        recv_pvs[rank] = pvs[rank2cells[rank][0]]
        cell_node_list = function.function_space().cell_node_list
        function.dat.data[:] = 0
        ret = np.zeros_like(function.dat.data_ro_with_halos)
        with timed_region("Restriction"):
            for r in range(size):
                X = Xs[r]
                pv = recv_pvs[r]
                cells = recv_cells_buffers[r]
                if m == 1:
                    for _X, _p, _cell in zip(X, pv, cells):
                        ret[cell_node_list[_cell]] += _p*_X
                else:
                    for i in range(0, m):
                        for _X, _p, _cell in zip(X, pv, cells):
                            ret[cell_node_list[_cell], i] += _p[i]*_X

        function.dat._data[:] = ret
        function.dat.local_to_global_begin(op2.INC)
        function.dat.local_to_global_end(op2.INC)
        function.dat.data[:]
        return function


@PETSc.Log.EventDecorator()
def batch_area_coordinates(function, cells, xs, tolerance=None):
    r"""Helper function to evaluate at points."""

    n = IntType.type(len(cells))
    dim = function.ufl_domain().topological_dimension()
    buf = np.zeros([n, dim+1], dtype=np.double)
    err = _c_evaluate_pointscloud(function, tolerance=tolerance)(function._ctypes,
                                                n,
                                                cells.ctypes.data_as(POINTER(c_petsc_int)),
                                                xs.ctypes.data_as(POINTER(c_double)),
                                                c_void_p(),
                                                buf.ctypes.data_as(POINTER(c_double)))
    if err > 0:
        # PETSc.Sys.syncPrint
        logger.warning('[%d]: %d of %d points are located in wrong place!'%(function.comm.rank, err, n))
    elif err < 0:
        raise PointNotInDomainError('We won\'t be here!')

    return buf


@PETSc.Log.EventDecorator()
def batch_eval(function, cells, xs, tolerance=None):
    r"""Helper function to evaluate at points."""

    n = IntType.type(len(cells))
    m = np.prod(function.ufl_shape, dtype=IntType)
    buf = np.zeros(n if m == 1 else [n, m], dtype=ScalarType)
    # cells = np.ascontiguousarray(np.array(cells, dtype=IntType))
    err = _c_evaluate_pointscloud(function, tolerance=tolerance)(function._ctypes,
                                                n,
                                                cells.ctypes.data_as(POINTER(c_petsc_int)),
                                                xs.ctypes.data_as(POINTER(c_double)),
                                                buf.ctypes.data_as(c_void_p),
                                                POINTER(c_double)())
    if err > 0:
        # PETSc.Sys.syncPrint
        logger.warning('[%d]: %d of %d points are located in wrong place!'%(function.comm.rank, err, n))
    elif err < 0:
        raise PointNotInDomainError('We won\'t be here!')

    return buf

@PETSc.Log.EventDecorator()
def _c_evaluate_pointscloud(function, tolerance=None):
    # Should this cache go to FunctionSpace?
    cache = function.__dict__.setdefault("_c_evaluate_pointscloud_cache", {})
    try:
        return cache[tolerance]
    except KeyError:
        result = make_c_evaluate(function, tolerance=tolerance)
        result.argtypes = [POINTER(_CFunction),
                           c_petsc_int,
                           POINTER(c_petsc_int),
                           POINTER(c_double),
                           c_void_p,
                           POINTER(c_double)]
        result.restype = c_petsc_int
        return cache.setdefault(tolerance, result)

@PETSc.Log.EventDecorator()
def make_c_evaluate(function, c_name="evaluate_points", ldargs=None, tolerance=None):
    r"""Generates, compiles and loads a C function to evaluate the
    given Firedrake :class:`Function` and return the area coordiantes."""

    from os import path

    from pyop2 import compilation, op2
    from pyop2.utils import get_petsc_dir
    # https://github.com/firedrakeproject/firedrake/pull/2235
    # TODO: remove the try
    try:
        from pyop2.sequential import generate_single_cell_wrapper
    except ModuleNotFoundError:
        from pyop2.parloop import generate_single_cell_wrapper

    from firedrake import utils
    import firedrake.pointquery_utils as pq_utils
    import firedrake as fd

    from fdutils.pointeval_utils import compile_element
    from fdutils.meshpatch import src_locate_cell

    mesh = function.ufl_domain()
    src = [src_locate_cell(mesh, tolerance=tolerance)]
    src.append(compile_element(function, mesh.coordinates))

    args = []

    arg = mesh.coordinates.dat(op2.READ, mesh.coordinates.cell_node_map())
    arg.position = 0
    args.append(arg)

    arg = function.dat(op2.READ, function.cell_node_map())
    arg.position = 1
    args.append(arg)

    p_ScalarType_c = f"{utils.ScalarType_c}*"
    src.append(generate_single_cell_wrapper(mesh.cell_set, args,
                                            forward_args=[p_ScalarType_c,
                                                          p_ScalarType_c],
                                            kernel_name="evaluate_kernel",
                                            wrapper_name="wrap_evaluate"))

    src = "\n".join(src)

    if ldargs is None:
        ldargs = []
    ldargs += ["-L%s/lib" % sys.prefix, "-lspatialindex_c", "-Wl,-rpath,%s/lib" % sys.prefix]
    return compilation.load(src, "c", c_name,
                            cppargs= ["-I%s" % p for p in fd.__path__]
                            + ["-I%s/include" % sys.prefix]
                            + ["-I%s/include" % d for d in get_petsc_dir()],
                            ldargs=ldargs,
                            comm=function.comm)
