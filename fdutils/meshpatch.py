import os
import sys
import ctypes
import numpy as np
import ufl
import firedrake as fd
import firedrake.utils as utils
import firedrake.pointquery_utils as pointquery_utils
from firedrake.mesh import MeshGeometry, spatialindex
from pyop2.datatypes import IntType, RealType, ScalarType

__all__ = []

def src_locate_cell(mesh, tolerance=None):
    if tolerance is None:
        tolerance = 1e-14
    src = ['#include <evaluate.h>']
    src.append(pointquery_utils.compile_coordinate_element(mesh.ufl_coordinate_element(), tolerance))
    src.append(pointquery_utils.make_wrapper(mesh.coordinates,
                            forward_args=["void*", "double*", "int*"],
                            kernel_name="to_reference_coords_kernel",
                            wrapper_name="wrap_to_reference_coords"))

    # with open(path.join(sys.prefix, 'src/firedrake/firedrake', "locate.c")) as f:
    with open(os.path.join(os.path.dirname(__file__), "locate.c")) as f:   # TODO
        src.append(f.read())

    src = "\n".join(src)
    return src

# patch for MeshGeometry

# def locate_cell(self, x, tolerance=None):
#     """Locate cell containg given point.
#     :arg x: point coordinates
#     :kwarg tolerance: for checking if a point is in a cell.
#     :returns: cell number (int), or None (if the point is not in the domain)
#     """
#     raise NotImplementedError("Use locate_cells instead")

def locate_cells(self, points, tolerance=None):
    if self.variable_layers:
        raise NotImplementedError("Cell location not implemented for variable layers")
    points = np.asarray(points, dtype=ScalarType).reshape(-1, self.geometric_dimension())
    if utils.complex_mode:
        if not np.allclose(points.imag, 0):
            raise ValueError("Provided points have non-zero imaginary part")
        points = points.real.copy()
    npoint, _ = points.shape
    cells = np.empty(npoint, dtype=np.intc) # TODO: make the locator use IntType?
    self._c_locators(tolerance=tolerance)(self.coordinates._ctypes,
                                         points.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                         npoint, # TODO: is this right?
                                         cells.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    return cells

def _c_locators(self, tolerance=None):
    from pyop2 import compilation
    from pyop2.utils import get_petsc_dir
    import firedrake.function as function

    cache = self.__dict__.setdefault("_c_locators_cache", {})
    try:
        return cache[tolerance]
    except KeyError:
        src = src_locate_cell(self, tolerance=tolerance)
        src += """
void locator(struct Function *f, double *xs, int npoint, int *cells)
{
    struct ReferenceCoords reference_coords;
    return locate_cells(f, xs, %(geometric_dimension)d, npoint, &to_reference_coords, &to_reference_coords_xtr, &reference_coords, cells);
}
""" % dict(geometric_dimension=self.geometric_dimension())

        locator = compilation.load(src, "c", "locator",
                                   cppargs=["-I%s" % os.path.dirname(__file__),
                                            "-I%s/src/firedrake/firedrake" % sys.prefix,
                                            "-I%s/include" % sys.prefix]
                                   + ["-I%s/include" % d for d in get_petsc_dir()],
                                   ldargs=["-L%s/lib" % sys.prefix,
                                           "-lspatialindex_c",
                                           "-Wl,-rpath,%s/lib" % sys.prefix])

        locator.argtypes = [ctypes.POINTER(function._CFunction),
                            ctypes.POINTER(ctypes.c_double),
                            ctypes.c_int,
                            ctypes.POINTER(ctypes.c_int)]
        locator.restype = None
        return cache.setdefault(tolerance, locator)


@utils.cached_property
def _inner_spatial_index(self):
    """Spatial index to quickly find which cell contains a given point
    and process spatial index to determine which processes may hold a given point.

    :returns: A tuple of two libspatialindex spatial index structure:
              (spatial_index, process_spatial_index)
    """

    from firedrake import function, functionspace
    from firedrake.parloops import par_loop, READ, MIN, MAX

    gdim = self.ufl_cell().geometric_dimension()
    if gdim <= 1:
        info_red("libspatialindex does not support 1-dimension, falling back on brute force.")
        return None

    # Calculate the bounding boxes for all cells by running a kernel
    V = functionspace.VectorFunctionSpace(self, "DG", 0, dim=gdim)
    coords_min = function.Function(V, dtype=RealType)
    coords_max = function.Function(V, dtype=RealType)

    coords_min.dat.data.fill(np.inf)
    coords_max.dat.data.fill(-np.inf)

    if utils.complex_mode:
        if not np.allclose(self.coordinates.dat.data_ro.imag, 0):
            raise ValueError("Coordinate field has non-zero imaginary part")
        coords = function.Function(self.coordinates.function_space(),
                                   val=self.coordinates.dat.data_ro_with_halos.real.copy(),
                                   dtype=RealType)
    else:
        coords = self.coordinates

    cell_node_list = self.coordinates.function_space().cell_node_list
    _, nodes_per_cell = cell_node_list.shape

    domain = "{{[d, i]: 0 <= d < {0} and 0 <= i < {1}}}".format(gdim, nodes_per_cell)
    instructions = """
    for d, i
        f_min[0, d] = fmin(f_min[0, d], f[i, d])
        f_max[0, d] = fmax(f_max[0, d], f[i, d])
    end
    """
    par_loop((domain, instructions), ufl.dx,
             {'f': (coords, READ),
              'f_min': (coords_min, MIN),
              'f_max': (coords_max, MAX)},
             is_loopy_kernel=True)

    # Reorder bounding boxes according to the cell indices we use
    column_list = V.cell_node_list.reshape(-1)
    coords_min = self._order_data_by_cell_index(column_list, coords_min.dat.data_ro_with_halos)
    coords_max = self._order_data_by_cell_index(column_list, coords_max.dat.data_ro_with_halos)

    # set mesh.bbox_relax_factor = 0.01 for high order mesh
    # should be done only when degree != 1 ?
    bbox_relax_factor = getattr(self, 'bbox_relax_factor', 0.01)
    distance = coords_max - coords_min
    coords_min[:] -= bbox_relax_factor*distance
    coords_max[:] += bbox_relax_factor*distance

    # Build spatial index
    spatial_index = spatialindex.from_regions(coords_min, coords_max)

    # Build process spatial index
    min_c = coords_min.min(axis=0)
    max_c = coords_max.max(axis=0)

    # Format: [min_x, min_y, min_z, max_x, max_y, max_z]
    local = np.concatenate([min_c, max_c])

    global_ = np.empty(len(local) * self.comm.size, dtype=local.dtype)
    self.comm.Allgather(local, global_)

    # Create lists containing the minimum and maximum bounds of each process, where
    # the index in each list is the rank of the process.
    min_bounds, max_bounds = global_.reshape(self.comm.size, 2,
                                             len(local) // 2).swapaxes(0, 1)

    # Arrays must be contiguous.
    min_bounds = np.ascontiguousarray(min_bounds)
    max_bounds = np.ascontiguousarray(max_bounds)

    # Build spatial indexes from bounds.
    process_spatial_index = spatialindex.from_regions(min_bounds, max_bounds)

    return spatial_index, process_spatial_index


def clear_spatial_index(self):
    """Reset the :attr:`spatial_index` on this mesh geometry.
    Use this if you move the mesh (for example by reassigning to
    the coordinate field)."""
    try:
        del self.spatial_index
        del self.process_spatial_index
        del self._inner_spatial_index
    except AttributeError:
        pass

@utils.cached_property
def spatial_index(self):
    """Spatial index to quickly find which cell contains a given point."""
    return self._inner_spatial_index[0]


@utils.cached_property
def processes_spatial_index(self):
    """A spatial index of processes using the bounding boxes of each process.
    This will be used to determine which processes may hold a given point.
    """
    return self._inner_spatial_index[1]


# patch all things    
# from firedrake.mesh import MeshGeometry
# import firedrake.pointquery_utils as pointquery_utils
# pointquery_utils.src_locate_cell = meshpatch.src_locate_cell
# MeshGeometry.locate_cell = meshpatch.locate_cell
MeshGeometry.locate_cells = locate_cells
MeshGeometry._c_locators = _c_locators
MeshGeometry._inner_spatial_index = _inner_spatial_index
MeshGeometry.spatial_index = spatial_index
MeshGeometry.processes_spatial_index = processes_spatial_index


# for load high order gmsh file
def getCoordinateFESpaceOrder(dm):
    cdm = dm.getCoordinateDM()
    kls, _ = cdm.getField(0)
    if kls.getClassName() == 'PetscFE':
        p = int(kls.getName()[1:])
    else:
        p = 1
    return p


def callback(mesh):

    """Finish initialisation."""
    del mesh._callback

    mesh.topology.init()

    cell_closure = mesh.cell_closure
    cell_sec = mesh._cell_numbering

    coordinates_fs = fd.functionspace.FunctionSpace(mesh.topology, mesh.ufl_coordinate_element())
    sec = coordinates_fs.dm.getDefaultSection()
    entity_permutations = coordinates_fs.finat_element.entity_permutations

    cell_node_list = coordinates_fs.cell_node_list

    dm = mesh.topology.topology_dm
    cdm = dm.getCoordinateDM()
    dim = dm.getCoordinateDim()
    csec = dm.getCoordinateSection()
    coords_gvec = dm.getCoordinates()
    cs, ce = dm.getHeightStratum(0)

    ncell = ce - cs
    ndof_per_ele = len(entity_permutations[dim][0][0])
    coordinates_data = np.zeros([ncell*ndof_per_ele, dim], dtype=ScalarType)
    #                [0, 1, 2, 3, 23, 13, 12, 03, 02, 01]
    magic = np.array([0, 2, 5, 9,  8,  7,  4,  6,  3,  1], dtype=IntType)
    maps = {
        0: 0,
        1: 2,
        2: 5,
        3: 9,
        (2, 3): 8,
        (1, 3): 7,
        (1, 2): 4,
        (0, 3): 6,
        (0, 2): 3,
        (0, 1): 1,
    }
    perm = np.zeros(len(maps), dtype=IntType)

    offset = 0
    for i in range(cs, ce):
        cell = cell_sec.getOffset(i)
        ccoords = cdm.getVecClosure(csec, coords_gvec, i)
        cl = dm.getTransitiveClosure(i)

        index_a = cl[0][-4:]
        index_a[0], index_a[1] = index_a[1], index_a[0]
        index_b = cell_closure[cell][:4]
        index = [np.where(index_a == _)[0][0] for _ in index_b]

        # index_a[index] == index_b
        for j, key in enumerate(maps.keys()):
            if isinstance(key, tuple):
                if index[key[0]] > index[key[1]]:
                    _i = (index[key[1]], index[key[0]])
                else:
                    _i = (index[key[0]], index[key[1]])
                perm[j] = maps[_i]
            else:
                perm[j] = maps[index[key]]

        cnl = cell_node_list[cell]
        coordinates_data[cnl, :] = ccoords.reshape([-1, dim])[perm, :]

    # Finish the initialisation of mesh topology
    coordinates = fd.function.CoordinatelessFunction(coordinates_fs, val=coordinates_data, name=mesh.name + "_coordinates")

    mesh.__init__(coordinates)


def make_mesh_from_mesh_topology(topology, name):
    # Construct coordinate element
    # TODO: meshfile might indicates higher-order coordinate element
    cell = topology.ufl_cell()
    geometric_dim = topology.topology_dm.getCoordinateDim()
    cell = cell.reconstruct(geometric_dimension=geometric_dim)
    topology_dim = topology.topology_dm.getDimension()

    p = getCoordinateFESpaceOrder(topology.topology_dm)
    if p == 1:
        element = ufl.VectorElement("Lagrange", cell, 1)
    elif p == 2:
        element = ufl.VectorElement("DG", cell, p)
    else:
        Exception('Can not load mesh with order p > 2 for now!')
    # Create mesh object
    mesh = fd.mesh.MeshGeometry.__new__(fd.mesh.MeshGeometry, element)
    mesh._init_topology(topology)
    mesh.name = name
    if p == 2:
        if geometric_dim == 3 and topology_dim == 3:
            mesh._callback = callback
        else:
            Exception('Only can load second order mesh for dim == 3 now!')
    return mesh


fd.mesh.make_mesh_from_mesh_topology = make_mesh_from_mesh_topology
