from fdutils import PointCloud
from firedrake import Function, FunctionSpace, \
                      TrialFunction, TestFunction, \
                      dx, inner, dot, \
                      assemble
from ufl import FiniteElement
import numpy as np


__pc_cache__ = {}
__mass_cache__ = {}


# Transfer a function from coarse space to the fine space
def interpolation(src: Function, dest: Function):
    m_src = src.ufl_domain()
    m_dest = dest.ufl_domain()
    V_dest = dest.funcion_space()
    key = (m_src, V_dest)

    if key in __pc_cache__:
        pc = __pc_cache__[key]
    else:
        points = Function(V_dest).interpolate(m_dest.coordinates)
        pc = PointCloud(m_src, points.dat.data_ro)

    val = pc.evaluate(src)
    dest.dat.data[:] = val[:]


# Transfer the fine solution to the coarse space
def injection(src: Function, dest: Function):
    interpolation(src, dest)


def check_P1_space(V: FunctionSpace):
    ele = V.ufl_element()
    if not (ele.family() == 'Lagrange' and 
            ele.degree() == 1 and 
            isinstance(ele, FiniteElement)):
        raise NotImplementedError


def get_lump_mass_matrix_inner(V: FunctionSpace):
    if V in __mass_cache__:
        M_lump = __mass_cache__[V]
    else:
        u, v = TrialFunction(V), TestFunction(V)
        a = inner(u, v)*dx
        M = assemble(a).petscmat
        s, e = M.getOwnershipRange()

        M_lump = Function(V)
        data = np.zeros_like(M_lump.dat.data_ro)
        for i in range(s, e):
            r = M.getRow(i)
            data[i] = np.sum(r[1])
        
        M_lump.dat.data[:] = data[:]
        __mass_cache__[V] = M_lump

    return M_lump


def get_lump_mass_matrix(src: Function):
    V_src = src.function_space()
    check_P1_space(V_src)
    return get_lump_mass_matrix_inner(V_src)


# Transfer the fine residual to the coarse space
# Here we assume src and dest are P1 Function
def restriction(src: Function, dest: Function):
    M_src = get_lump_mass_matrix(src)
    M_dest = get_lump_mass_matrix(dest)



    pass