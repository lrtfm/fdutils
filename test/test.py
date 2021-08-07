from firedrake.utility_meshes import UnitSquareMesh
from firedrake.functionspace import FunctionSpace, VectorFunctionSpace
from firedrake.function import Function

from ufl.geometry import SpatialCoordinate 

from fdutils import tools

def get_fun(n, degree=None, fun=None):
    degree = degree or 2
    fun = fun or (lambda x: x[0]**4 + x[1]**4)

    quadrilateral=False
    m = UnitSquareMesh(n, n, quadrilateral=quadrilateral)
    V = FunctionSpace(m, 'CG', degree)
    f = Function(V)
    x = SpatialCoordinate(m)
    f.interpolate(fun(x))
    
    return f

def get_fun_kvm(n, degree=None, fun=None):
    degree = degree or 2
    fun = fun or (lambda x: x[0]**4 + x[1]**4)

    quadrilateral=False
    m = UnitSquareMesh(n, n, quadrilateral=quadrilateral)
    V = FunctionSpace(m, 'KMV', degree)
    f = Function(V)
    x = SpatialCoordinate(m)
    f.interpolate(fun(x))
    
    return f


n = 4
fs_ = [get_fun(2**(_+1)) for _ in range(n)]
fs = [get_fun_kvm(2**(_+1)) for _ in range(n)]


error_handles = {}
sep = '-'
for method in ['at', 'vom', 'pc']:
    for comp in ['Cauchy', 'Reference']:
        error_handles[sep.join([method,comp])] = \
            tools.prepare_errornorm_handle(fs, tolerance=1e-10, eval_method=method, compare_method=comp)

errs = {}
for name, handle in error_handles.items():
    errs[name] = handle()
