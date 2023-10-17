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


def locate_cells(mesh: MeshGeometry, points, tolerance=None):
    cells, Xs, ref_cell_dists_l1 = mesh.locate_cells_ref_coords_and_dists(points, tolerance)
    return cells


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

    tdim = self.ufl_cell().topological_dimension()
    if gdim == tdim:
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
                'f_max': (coords_max, MAX)})
    else:
        # consider the normal direction for manifold
        from firedrake import CellNormal, CellVolume, CellSize, sqrt
        degree = self.coordinates.function_space().ufl_element().degree()
        V_normal = functionspace.VectorFunctionSpace(self, 'CG', degree)
        normal = function.Function(V_normal, dtype=RealType)
        normal.interpolate(CellNormal(self))

        if degree > 1:
            V_size = functionspace.FunctionSpace(self, 'DG', 0)
            size = function.Function(V_size, dtype=RealType)
            size.interpolate(CellVolume(self))
            size.assign(sqrt(2*size))   # only for surface, it is ok for now
        else:
            # V_size = functionspace.FunctionSpace(self, 'DG', 0)
            # size = function.Function(V_size, dtype=RealType)
            # size.interpolate(CellSize(self))
            size = self.cell_sizes

        cell_node_list = self.coordinates.function_space().cell_node_list
        _, nodes_per_cell = cell_node_list.shape

        domain = "{{[d, i]: 0 <= d < {0} and 0 <= i < {1}}}".format(gdim, nodes_per_cell)
        instructions = """
        for d, i
            <> dist = h[0, 0]*n[i, d]/3.0
            f_min[0, d] = fmin(fmin(f_min[0, d], f[i, d] + dist), f[i, d] - dist)
            f_max[0, d] = fmax(fmax(f_max[0, d], f[i, d] + dist), f[i, d] - dist)
        end
        """
        par_loop((domain, instructions), ufl.dx,
                {'f': (coords, READ),
                'n': (normal, READ),
                'h': (size, READ),
                'f_min': (coords_min, MIN),
                'f_max': (coords_max, MAX)})

    # Reorder bounding boxes according to the cell indices we use
    column_list = V.cell_node_list.reshape(-1)
    coords_min = self._order_data_by_cell_index(column_list, coords_min.dat.data_ro_with_halos)
    coords_max = self._order_data_by_cell_index(column_list, coords_max.dat.data_ro_with_halos)

    # set mesh.bbox_relax_factor = 0.01 for high order mesh
    # should be done only when degree != 1 ?
    if hasattr(self, "tolerance") and self.tolerance is not None:
        coords_diff = coords_max - coords_min
        coords_min -= self.tolerance*coords_diff
        coords_max += self.tolerance*coords_diff

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
MeshGeometry._inner_spatial_index = _inner_spatial_index
MeshGeometry.spatial_index = spatial_index
MeshGeometry.processes_spatial_index = processes_spatial_index
MeshGeometry.clear_spatial_index = clear_spatial_index
