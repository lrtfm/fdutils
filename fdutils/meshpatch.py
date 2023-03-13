import os
import sys
import ctypes
import numpy as np
from pyop2.datatypes import IntType, RealType, ScalarType

import ufl
from firedrake import utils
from firedrake.mesh import MeshGeometry, spatialindex
from firedrake.logging import info_red
import firedrake.utils as utils
import firedrake.pointquery_utils as pointquery_utils

__all__ = []

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
    import firedrake.pointquery_utils as pq_utils
    import firedrake as fd

    cache = self.__dict__.setdefault("_c_locators_cache", {})
    try:
        return cache[tolerance]
    except KeyError:
        src = pq_utils.src_locate_cell(self, tolerance=tolerance)
        src += """
// TODO: should output Xs?
// void locator(struct Function *f, double *xs, int npoint, int *cells, double *Xs)
void locator(struct Function *f, double *xs, int npoint, int *cells)
{
    /* The type definitions and arguments used here are defined as
        statics in pointquery_utils.py */
    struct ReferenceCoords temp_reference_coords, found_reference_coords;
    int j;
    int dim = %(geometric_dimension)d;
    for (j = 0; j < npoint; j++) {
        cells[j] = locate_cell(f, &xs[j*%(geometric_dimension)d], %(geometric_dimension)d,
                           &to_reference_coords, &to_reference_coords_xtr,
                           &temp_reference_coords, &found_reference_coords);
        // if (cells[j] >= 0) {
        //     for(int i=0; i<%(geometric_dimension)d; i++) {
        //         Xs[j*dim + i] = found_reference_coords.X[i];
        //     }
        // }
    }
}
""" % dict(geometric_dimension=self.geometric_dimension())

        locator = compilation.load(src, "c", "locator",
                                    cppargs=["-I%s" % p for p in fd.__path__]
                                          + ["-I%s/include" % sys.prefix]
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
        return (None, None)

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
