from collections import OrderedDict

import os
import sys
import numpy as np

from firedrake.mesh import spatialindex
# from firedrake.dmplex import build_two_sided
from firedrake.utils import cached_property
from firedrake.petsc import PETSc
from firedrake import logging
from pyop2.mpi import MPI
from pyop2.datatypes import IntType, ScalarType
from pyop2.profiling import timed_region

from ctypes import POINTER, c_int, c_double, c_void_p

from firedrake.function import _CFunction

try:
    from fdutils.evalpatch import build_two_sided
except ModuleNotFoundError:
     raise
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise

__all__ = ["PointCloud"]


def syncPrint(*args, **kwargs):
    """Perform a PETSc syncPrint operation with given arguments if the logging level is
    set to at least debug.
    """
    if logging.logger.isEnabledFor(logging.DEBUG):
        PETSc.Sys.syncPrint(*args, **kwargs)
        
def Print(*args, **kwargs):
    if logging.logger.isEnabledFor(logging.DEBUG):
        PETSc.Sys.Print(*args, **kwargs)

def syncFlush(*args, **kwargs):
    """Perform a PETSc syncFlush operation with given arguments if the logging level is
    set to at least debug.
    """
    if logging.logger.isEnabledFor(logging.DEBUG):
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
        self.points = np.asarray(points)
        syncPrint('[%d]'%mesh.comm.rank, points)
        self.tolerance = tolerance
        _, dim = points.shape
        if dim != mesh.geometric_dimension():
            raise ValueError("Points must be %d-dimensional, (got %d)" %
                             (mesh.geometric_dimension(), dim))

        # Build spatial index of processes for point location.
        self.processes_index = self._build_processes_spatial_index()

        # Initialise dictionary to store location statistics for evaluation.
        self.statistics = OrderedDict()
        # Initialise counters.
        self.statistics["num_messages_sent"] = 0
        self.statistics["num_points_evaluated"] = 0
        self.statistics["num_points_found"] = 0

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

    def _build_processes_spatial_index(self):
        """Build a spatial index of processes using the bounding boxes of each process.
        This will be used to determine which processes may hold a given point.

        :returns: A libspatialindex spatial index structure.
        """
        min_c = self.mesh.coordinates.dat.data_ro_with_halos.min(axis=0)
        max_c = self.mesh.coordinates.dat.data_ro_with_halos.max(axis=0)
        
        # Format: [min_x, min_y, min_z, max_x, max_y, max_z]
        local = np.concatenate([min_c, max_c])

        global_ = np.empty(len(local) * self.mesh.comm.size, dtype=local.dtype)
        self.mesh.comm.Allgather(local, global_)

        # Create lists containing the minimum and maximum bounds of each process, where
        # the index in each list is the rank of the process.
        min_bounds, max_bounds = global_.reshape(self.mesh.comm.size, 2,
                                                 len(local) // 2).swapaxes(0, 1)

        # Arrays must be contiguous.
        min_bounds = np.ascontiguousarray(min_bounds)
        max_bounds = np.ascontiguousarray(max_bounds)
        
        # Build spatial indexes from bounds.
        return spatialindex.from_regions(min_bounds, max_bounds)

    def _get_candidate_processes(self, point):
        """Determine candidate processes for a given point.

        :arg point: A point on the mesh.

        :returns: A numpy array of candidate processes.
        """
        candidates = spatialindex.bounding_boxes(self.processes_index, point)
        return candidates[candidates != self.mesh.comm.rank]

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

                    syncPrint("[%d] Cell not found locally for point %s. Candidates: %s"
                              % (self.mesh.comm.rank, point, point_candidates),
                              comm=self.mesh.comm)

            syncFlush(comm=self.mesh.comm)

        self.statistics["num_points_found"] += \
            np.count_nonzero(located_elements[:, 0] != -1)
        self.statistics["local_found_frac"] = \
            self.statistics["num_points_found"] / (len(self.points) or 1)  # fix ZeroDivisionError: division by zero
        self.statistics["num_candidates"] = len(local_candidates)

        syncPrint("[%d] Located elements: %s" % (self.mesh.comm.rank,
                                                 located_elements.tolist()),
                  comm=self.mesh.comm)
        syncFlush(comm=self.mesh.comm)
        syncPrint("[%d] Local candidates: %s" % (self.mesh.comm.rank, local_candidates),
                  comm=self.mesh.comm)
        syncFlush(comm=self.mesh.comm)

        # Exchange data for point requests.
        with timed_region("PointExchange"):
            # First get number of points to receive from each rank through sparse
            # communication round.

            # Create input arrays from candidates dictionary.
            to_ranks = np.zeros(len(local_candidates), dtype=IntType)
            to_data = np.zeros(len(local_candidates), dtype=IntType)
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
                    (from_data[i], self.mesh.geometric_dimension()), dtype=ScalarType)

            # Receive all point requests

            local_candidates = {r: np.asarray(p) for r, p in local_candidates.items()}
            self._perform_sparse_communication_round(recv_points_buffers,
                                                     local_candidates)

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
                point_responses[rank] = self.mesh.locate_cells(points_buffers, tolerance=tolerance)
                self.statistics["num_points_evaluated"] += len(points_buffers)
                self.statistics["num_points_found"] += \
                    np.count_nonzero(point_responses[rank] != -1)

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
                        located_elements[i, :] = (rank, result[idx])

        syncPrint("[%d] Located elements: %s" % (self.mesh.comm.rank,
                                                 located_elements.tolist()),
                  comm=self.mesh.comm)
        syncFlush(comm=self.mesh.comm)
        
        np_not_found_index = np.where(located_elements[:, 1] == -1)[0]
        if len(np_not_found_index) > 0:
            logging.warning('[%d] %d points not located!'%(self.mesh.comm.rank, len(np_not_found_index)))
            logging.warning('[%d] points list: %s!'%(self.mesh.comm.rank, self.points[np_not_found_index, :]))
        # PETSc.Sys.syncFlush()
        
        return located_elements

    @cached_property
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
            to_ranks = np.zeros(len(rank2cells), dtype=IntType)
            to_data = np.zeros(len(rank2cells), dtype=IntType)
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
                    (from_data[i], self.mesh.geometric_dimension()), dtype=ScalarType)

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
    def evaluate(self, function):
        """Evaluate a function at the located points.
        :arg function: The function to evaluate.
        """
        
        if function.ufl_domain() != self.mesh:
            raise('The function must be defined on the mesh of this PointCloud!')
        
        rank = self.mesh.comm.rank
        size = self.mesh.comm.rank
        
        # must do this!
        Print('A0'*30)
        from pyop2 import op2
        function.dat.global_to_local_begin(op2.READ)
        function.dat.global_to_local_end(op2.READ)
    
        
        Print('A'*30)
        recv_cells_buffers, recv_points_buffers, rank2cells = self.evaluate_info
        
        values = {}
        
        Print('B'*30)
        for r, cells in recv_cells_buffers.items():
            ps = recv_points_buffers[r]
            values[r] = batch_eval(function, cells, ps, tolerance=self.tolerance)
            
        Print('C'*30)
        n = len(self.points)
        m = np.prod(function.ufl_shape, dtype=np.int64)
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
        
        Print('D'*30)
        with timed_region("EvalResultExchange"):
            self._perform_sparse_communication_round(recv_values_buffers, values)

        Print('E'*30)
        for r, v in recv_values_buffers.items():
            if m == 1:
                ret[rank2cells[r][0]] = v
            else:
                ret[rank2cells[r][0], :] = v
        
        return ret
    
@PETSc.Log.EventDecorator()
def batch_eval(function, cells, xs, tolerance=None):
    r"""Helper function to evaluate at points."""
    
    n = IntType.type(len(cells))
    m = np.prod(function.ufl_shape, dtype=np.int64)
    buf = np.zeros(n if m == 1 else [n, m], dtype=ScalarType)
    cells = np.array(cells, dtype=IntType)
    err = _c_evaluate_pointscloud(function, tolerance=tolerance)(function._ctypes,
                                                n,
                                                cells.ctypes.data_as(POINTER(c_int)),
                                                xs.ctypes.data_as(POINTER(c_double)),
                                                buf.ctypes.data_as(c_void_p))
    if err > 0:
        # PETSc.Sys.syncPrint
        logging.warning('[%d]: %d of %d points are located in wrong place!'%(function.comm.rank, err, n))
        
    if err == -1:
        raise PointNotInDomainError('We won\'t be here!')
        
    return buf
    
@PETSc.Log.EventDecorator()
def _c_evaluate_pointscloud(function, tolerance=None):
    cache = function.__dict__.setdefault("_c_evaluate_pointscloud_cache", {})
    try:
        return cache[tolerance]
    except KeyError:
        result = make_c_evaluate(function, tolerance=tolerance)
        result.argtypes = [POINTER(_CFunction), 
                           c_int,
                           POINTER(c_int),
                           POINTER(c_double), 
                           c_void_p]
        result.restype = c_int
        return cache.setdefault(tolerance, result)
        
@PETSc.Log.EventDecorator()
def make_c_evaluate(function, c_name="evaluate_points", ldargs=None, tolerance=None):
    r"""Generates, compiles and loads a C function to evaluate the
    given Firedrake :class:`Function`."""

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
                            cppargs=["-I%s" % os.path.dirname(__file__),
                                     "-I%s/src/firedrake/firedrake" % sys.prefix,
                                     "-I%s/include" % sys.prefix]
                            + ["-I%s/include" % d for d in get_petsc_dir()],
                            ldargs=ldargs,
                            comm=function.comm)
