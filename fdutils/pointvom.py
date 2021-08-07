from firedrake.mesh import VertexOnlyMesh
from firedrake.functionspace import FunctionSpace, VectorFunctionSpace
from firedrake.function import Function
from firedrake.interpolation import Interpolator
import firedrake

class Tolerance(object):
    def __init__(self, tolerance):
        self.tolerance = tolerance
        self.default = firedrake.mesh.MeshGeometry.locate_cell_and_reference_coordinate.__defaults__
    def __enter__(self):
        firedrake.mesh.MeshGeometry.locate_cell_and_reference_coordinate.__defaults__ = (self.tolerance,)
    def __exit__(self, type, value, traceback):
        firedrake.mesh.MeshGeometry.locate_cell_and_reference_coordinate.__defaults__ = self.default

class PointVOM():
    def __init__(self, mesh, points, tolerance=None):
        self.points = points
        self.tolerance = tolerance
        with Tolerance(self.tolerance):
            self.vm = VertexOnlyMesh(mesh, points)
        self._cache = {} # collections.defaultdict(
    
    def _get_cache(self, fun):
        if fun in self._cache.keys():
            return self._cache[fun]
        
        V = fun.ufl_function_space()
        if V.rank == 0:
            V_vm = FunctionSpace(self.vm, 'DG', 0)
        elif V.rank == 1:
            V_vm = FunctionSpace(self.vm, 'DG', 0,
                                 dim=V.ufl_element().value_size())
        else:
            raise NotImplementedError('We currently only support for'
                                      ' functions with rank <= 1!')
        f = Function(V_vm)
        Inter = Interpolator(fun, V_vm)
        self._cache[fun] = Inter, f
        
        return self._cache[fun]
    
    def evaluate(self, fun):
        Inter, f = self._get_cache(fun)
        Inter.interpolate(output=f)
        return f.dat.data_ro.copy()
    