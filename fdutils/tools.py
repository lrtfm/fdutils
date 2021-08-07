""" some utils for Firedrake
"""

from firedrake.functionspace import FunctionSpace, VectorFunctionSpace
from firedrake.function import Function
from firedrake.norms import errornorm
from firedrake.mesh import VertexOnlyMesh

from peval import PointCloud

from petsc4py import PETSc
import numpy as np

def get_nodes_coords(fun):
    ''' Get the coords of the dofs of Function fun or Space'''

    mesh = fun.ufl_domain()
    element = fun.ufl_element()
    
    assert(element.family() == 'Lagrange')

    degree = element.degree()
    if degree == 1:
        coords = mesh.coordinates.dat.data
    else:
        C = VectorFunctionSpace(mesh, 'CG', degree)
        interp_coordinates = Function(C)
        interp_coordinates.interpolate(mesh.coordinates)
        coords = interp_coordinates.dat.data

    return coords

def get_common_function_space(u, u_ref):
    '''Get the CG space for compute errors of u and u_ref
    '''
    element = u.ufl_element()
    element_ref = u_ref.ufl_element()
    assert(element == element_ref)
    
    if element.family() == 'Lagrange':
        V_inter = u_ref.ufl_function_space()
        return V_inter, False
    
    mesh = u_ref.ufl_domain()
    V = u_ref.ufl_function_space()
    degree = element_ref.degree() + 2 # may be just add 2?
    if V.rank == 1:
        V_inter = VectorFunctionSpace(mesh, 'CG', degree=degree, dim=element_ref.value_size())
    elif V.rank == 0:
        V_inter = FunctionSpace(mesh, 'CG', degree=degree)
    else:
        raise NotImplementedError('Do not support space with rank > 1')
        
    return V_inter, True


def evaluate_at(fun, coords):
    comm = fun.comm
    arrays = coords
    for _ in range(n):
        arrays_curr = comm.bcast(arrays, root=_)

        value_ = fun.at(arrays_curr,\
                        dont_raise=True, tolerance=1e-12)

        if comm.rank == _:
            value = value_

    v_size = fun.ufl_element().value_size()
    noneindex =[i for i,v in enumerate(value) if v is None]
    if v_size > 1:
        zeros = [0. for _ in range(v_size)]
        for i in noneindex:
            value[i] = np.array(zeros)
    else:
        for i in noneindex:
            value[i] = 0.  # Should be tested, but we won't goto here now.
    
    return np.array(value)


def compute_errors_at(u, u_ref, tolerance=None, norm_type="L2"):

    V_inter, need_inter_u_ref = get_common_function_space(u, u_ref)
    coords = get_nodes_coords(V_inter)
    
    coords[coords[:, 1] < 0.5 + 1e-12, 1] = 0.5 + 1e-12
    coords[coords[:, 1] > 1.5 - 1e-12, 1] = 1.5 - 1e-12
    coords[coords[:, 0] < 0.5 + 1e-12, 0] = 0.5 + 1e-12
    coords[coords[:, 0] > 1.5 - 1e-12, 0] = 1.5 - 1e-12

    u_inter = Function(V_inter)
    
    if need_inter_u_ref:
        u_ref_inter = Function(V_inter)
    else:
        u_ref_inter = None
    
    u_inter.dat.data[:] = evaluate_at(u, coords) # u.at(coords)
    
    if u_ref_inter is not None:
        u_ref_inter.dat.data[:] = evaluate_at(u_ref, coords) # u_ref.at(coords)
    else:
        u_ref_inter = u_ref
    
    return errornorm(u_ref_inter, u_inter, norm_type=norm_type)


def get_pointclouds(u, u_ref, tolerance):
    V_inter, need_inter_u_ref = get_common_function_space(u, u_ref)
    coords = get_nodes_coords(V_inter)
    u_inter = Function(V_inter)
    
    pc = PointCloud(u.ufl_domain(), coords, tolerance=tolerance)
    
    if need_inter_u_ref:
        pc_ref = PointCloud(u_ref.ufl_domain(), coords, tolerance=tolerance)
        u_ref_inter = Function(V_inter)
    else:
        pc_ref = None
        u_ref_inter = None
    
    return pc, pc_ref, u_inter, u_ref_inter


def compute_errors_pc(u, u_ref, assistant=None, tolerance=None, norm_type="L2"):
    if assistant is None:
        pc, pc_ref, u_inter, u_ref_inter = get_pointclouds(u, u_ref, tolerance)
    else:
        pc, pc_ref, u_inter, u_ref_inter = assistant
        
    u_inter.dat.data[:] = pc.evaluate(u)

    if pc_ref is not None:
        u_ref_inter.dat.data[:] = pc_ref.evaluate(u_ref)
    else:
        u_ref_inter = u_ref
        
    return errornorm(u_ref_inter, u_inter, norm_type=norm_type)









