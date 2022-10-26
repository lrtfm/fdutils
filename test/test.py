from firedrake.utility_meshes import UnitSquareMesh
from firedrake.functionspace import FunctionSpace, VectorFunctionSpace
from firedrake.function import Function
from firedrake.petsc import PETSc
from ufl.geometry import SpatialCoordinate 
from fdutils import tools
import numpy as np


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

for comp in ['Cauchy', 'Reference']:
    a = np.allclose(errs['at' + sep + comp], errs['vom' + sep + comp])
    b = np.allclose(errs['at' + sep + comp], errs['pc' + sep + comp])
    if a and b:
        PETSc.Sys.Print(f'Test for {comp} OK')
    else:
        if not a:
            PETSc.Sys.Print(f'Test for at-vom Fail:')
            PETSc.Sys.Print("*"*80)
            PETSc.Sys.Print(errs['at' + sep + comp], errs['vom' + sep + comp])
            PETSc.Sys.Print("*"*80)
        if not b:
            PETSc.Sys.Print(f'Test for at-pc Fail')
            PETSc.Sys.Print("*"*80)
            PETSc.Sys.Print(errs['at' + sep + comp], errs['pc' + sep + comp])
            PETSc.Sys.Print("*"*80)

