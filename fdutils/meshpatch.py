import os
import sys
import ctypes
import numpy as np

from firedrake.mesh import MeshGeometry
import firedrake.pointquery_utils as pointquery_utils

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
    points = np.asarray(points, dtype=np.float64).reshape(-1, self.geometric_dimension())
    npoint, _ = points.shape
    cells = np.empty(npoint, dtype=np.int32)
    self._c_locators(tolerance=tolerance)(self.coordinates._ctypes,
                                         points.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                         npoint,
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


# patch all things    
# from firedrake.mesh import MeshGeometry
# import firedrake.pointquery_utils as pointquery_utils
# pointquery_utils.src_locate_cell = meshpatch.src_locate_cell
# MeshGeometry.locate_cell = meshpatch.locate_cell
MeshGeometry.locate_cells = locate_cells
MeshGeometry._c_locators = _c_locators
