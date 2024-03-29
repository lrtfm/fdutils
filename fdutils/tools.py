""" some utils for Firedrake
"""
import sys

from firedrake import functionspaceimpl as impl
from firedrake import FiniteElement, TensorProductElement
from firedrake.functionspace import FunctionSpace, VectorFunctionSpace
from firedrake.function import Function
from firedrake.mesh import VertexOnlyMesh
from firedrake import logging
from firedrake import norms

from petsc4py import PETSc
import numpy as np

from fdutils.pointarray import PointArray
from fdutils.pointvom import PointVOM

__all__ = ['PointArray',
           'PointVOM',
           # 'get_nodes_coords',
           # 'get_cg_function_space',
           'errornorm']

eval_method2cls = {'at': PointArray, 'vom': PointVOM, None: PointArray}

try:
    from fdutils.pointcloud import PointCloud
except ImportError:
    logging.warning('PointCloud can not be imported! We will use PointArray!')
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise

if 'PointCloud' in locals():
    __all__.append('PointCloud')
    eval_method2cls['pc'] = PointCloud
    eval_method2cls[None] = PointCloud

def get_nodes_coords(fun):
    ''' Get the coords of the dofs of Function fun or Space'''

    mesh = fun.ufl_domain()
    element = fun.ufl_element()
    degree = element.degree()
    degree_coord = mesh.coordinates.ufl_element().degree()

    eles = [element]
    while len(eles):
        ele = eles.pop(0)
        sub = ele.sub_elements
        if len(sub) > 0:
            eles += sub
        else:
            if not ((ele.family() == 'Lagrange' and ele.degree() > 0) or
                (ele.family() == 'Discontinuous Lagrange' and ele.degree() == 0)):
                raise Exception(f'Do not support element type: {ele}')

    if element.family() == 'TensorProductElement':
        cells = element.cell.sub_cells()
        assert len(degree) == 2
        if degree[0] == 1 and degree[1] == 1 and degree_coord == 1:
            points = mesh.coordinates.dat.data_ro.copy()
        else:
            ele0 = FiniteElement("CG" if degree[0] > 0 else "DG", cells[0], degree[0])
            ele1 = FiniteElement("CG" if degree[1] > 0 else "DG", cells[1], degree[1])
            ele = TensorProductElement(ele0, ele1)
            C = VectorFunctionSpace(mesh, ele)
            interp_coordinates = Function(C)
            interp_coordinates.interpolate(mesh.coordinates)
            points = interp_coordinates.dat.data_ro.copy()
    else:
        if degree == 1 and degree_coord == 1:
            points = mesh.coordinates.dat.data_ro.copy()
        else:
            C = VectorFunctionSpace(mesh, 'CG' if degree > 0 else 'DG', degree)
            interp_coordinates = Function(C)
            interp_coordinates.interpolate(mesh.coordinates)
            points = interp_coordinates.dat.data_ro.copy()

    return points

def get_cg_function_space(u, degree=None):
    '''Get a CG space that including the space of u_ref
    '''

    if isinstance(u, impl.WithGeometry):
        V = u
        if isinstance(V.topological, impl.MixedFunctionSpace):
            raise NotImplementedError('We currently do not support MixedFunctionSpace')
    elif isinstance(u, Function):
        V = u.ufl_function_space()
    else:
        raise NotImplementedError('Only support FunctionSpace and Function')


    if V.ufl_element().family() == 'Lagrange':
        return V

    # TODO for different element
    degree = degree or (V.ufl_element().degree() + 2) # may be just add 2?

    if V.rank == 1:
        V_inter = VectorFunctionSpace(V.ufl_domain(),
                                      'CG',  degree=degree,
                                      dim=V.ufl_element().value_size())
    elif V.rank == 0:
        V_inter = FunctionSpace(V.ufl_domain(), 'CG', degree=degree)
    else:
        raise NotImplementedError('Do not support function space with rank > 1')

    return V_inter


def prepare_for_evaluate(u, u_ref, method=None, tolerance=None):

    eval_cls = eval_method2cls[method]

    V_inter = get_cg_function_space(u_ref)
    coords = get_nodes_coords(V_inter)

    u_inter = Function(V_inter)
    pc = eval_cls(u.ufl_domain(), coords, tolerance=tolerance)

    return pc, u_inter


def errornorm(u, u_ref, method=None, tolerance=None, norm_type="L2"):
    p, u_inter = prepare_for_evaluate(u, u_ref, method=method, tolerance=tolerance)
    u_inter.dat.data[:] = p.evaluate(u)
    return norms.errornorm(u_inter, u_ref, norm_type=norm_type)


compare2handles = {'Cauchy': lambda funs, i: funs[i+1],
                   'Reference': lambda funs, i: funs[-1]}

def prepare_errornorm_handle(funs, compare_method=None, eval_method=None, tolerance=None, norm_type='L2'):

    ref_handle = compare2handles[compare_method or 'Reference']
    eval_cls = eval_method2cls[eval_method]
    n = len(funs)
    pcs = []
    f_inters = []

    for i in range(n-1):
        V_inter = get_cg_function_space(ref_handle(funs, i))
        f_inter = Function(V_inter)

        coords = get_nodes_coords(V_inter)
        pc = eval_cls(funs[i].ufl_domain(), coords, tolerance=tolerance)

        pcs.append(pc)
        f_inters.append(f_inter)

    if isinstance(norm_type, str):
        norm_types = [norm_type,]
    else:
        norm_types = norm_type

    def get_errors():
        errs = {}
        for nt in norm_types:
            errs[nt] = []
        for i in range(n-1):
            f_inters[i].dat.data[:] = pcs[i].evaluate(funs[i])
            for nt in norm_types:
                errs[nt].append(norms.errornorm(f_inters[i], ref_handle(funs, i), norm_type=nt))
        if len(norm_types) == 1:
            errs = errs[norm_types[0]]
        return errs

    return get_errors
