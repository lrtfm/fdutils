from fdutils import PointCloud
from firedrake import Function, FunctionSpace, \
                      VectorFunctionSpace, TensorFunctionSpace,\
                      TrialFunction, TestFunction, \
                      dx, inner, dot, \
                      assemble
from firedrake.petsc import PETSc
from ufl import FiniteElement, TensorProductElement
import numpy as np


def check_P1_space(V: FunctionSpace):
    ele = V.ufl_element()
    if not (ele.family() == 'Lagrange' and 
            ele.degree() == 1 and 
            isinstance(ele, FiniteElement)):
        raise NotImplementedError


def get_lump_mass_matrix(V: FunctionSpace):
    check_P1_space(V)
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(u, v)*dx
    M = assemble(a).petscmat

    M_lump = Function(V)
    data = M_lump.dat.data
    s, e = M.getOwnershipRange()
    for i in range(s, e):
        r = M.getRow(i)
        data[i-s] = np.sum(r[1])

    M_lump.dat.data[:] = data[:]

    return M_lump


def get_nodes_coords_space(V):
    ''' Get the coords of the dofs of Function fun or Space'''

    mesh = V.ufl_domain()
    element = V.ufl_element()
    degree = element.degree()
    degree_coord = mesh.coordinates.ufl_element().degree()

    eles = [element]
    while len(eles):
        ele = eles.pop(0)
        sub = ele.sub_elements()
        if len(sub) > 0:
            eles += sub
        else:
            if not ((ele.family() == 'Lagrange' and ele.degree() > 0) or
                (ele.family() == 'Discontinuous Lagrange' and ele.degree() == 0)):
                raise Exception(f'Do not support element type: {ele}')

    if element.family() == 'TensorProductElement':
        cells = element.cell().sub_cells()
        assert len(degree) == 2
        if degree[0] == degree[1] == degree_coord:
            C = mesh.coordinates.function_space()
        else:
            ele0 = FiniteElement("CG" if degree[0] > 0 else "DG", cells[0], degree[0])
            ele1 = FiniteElement("CG" if degree[1] > 0 else "DG", cells[1], degree[1])
            ele = TensorProductElement(ele0, ele1)
            C = VectorFunctionSpace(mesh, ele)
    else:
        if degree == degree_coord:
            C = mesh.coordinates.function_space()
        else:
            C = VectorFunctionSpace(mesh, 'CG' if degree > 0 else 'DG', degree)

    return C


class NonnestedTransferManager(object):
    __pc_caches = {}
    __mat_caches = {}

    def __init__(self, *, native_transfers=None, use_averaging=True):
        PETSc.Sys.Print(f'Init {type(self)}')
        self.pc_caches = NonnestedTransferManager.__pc_caches
        self.mat_caches = NonnestedTransferManager.__mat_caches
        self.tolerance = PETSc.Options().getReal('-ntm_tolerance', 1e-12)
        pass

    def get_pointcloud(self, key):
        m_src, V = key
        if key in self.pc_caches:
            pc = self.pc_caches[key]
        else:
            points = Function(V).interpolate(V.ufl_domain().coordinates)
            pc = PointCloud(m_src, points.dat.data_ro, self.tolerance)
            self.pc_caches[key] = pc

        return pc

    def get_interpolate_matrix(self, src, dest):
        m_src = src.ufl_domain()
        m_dest = dest.ufl_domain()
        V_dest = dest.function_space()
        V = get_nodes_coords_space(V_dest)
        key = (m_src, V)

        if key in self.mat_caches:
            mat = self.mat_caches[key]
        else:
            pc = self.get_pointcloud(key)
            mat = pc.create_interpolate_matrix(src, dest)
            self.mat_caches[key] = mat

        return mat

    def get_lump_mass_matrix(self, V: FunctionSpace):
        if V in self.mat_caches:
            M = self.mat_caches[V]
        else:
            M = get_lump_mass_matrix(V)
            self.mat_caches[V] = M
        return M

    def interpolate(self, src: Function, dest: Function):
        mat = self.get_interpolate_matrix(src, dest)
        with src.dat.vec_ro as src_vec, dest.dat.vec as dest_vec:
                mat.mult(src_vec, dest_vec)

    # Transfer a function from coarse space to the fine space
    # prolong
    def prolong(self, src: Function, dest: Function):
        self.interpolate(src, dest)

    # Transfer the fine solution to the coarse space
    def inject(self, src: Function, dest: Function):
        self.prolong(src, dest)

    # Transfer the fine residual to the coarse space
    # Here we assume src and dest are P1 Function
    def restrict(self, src: Function, dest: Function):
        M_src = self.get_lump_mass_matrix(src.function_space())
        M_dest = self.get_lump_mass_matrix(dest.function_space())

        # when restrict src to dest
        # we use the mesh of dest as base mesh
        pvs = Function(src)
        pvs.dat.data[:] = M_src.dat.data_ro*src.dat.data_ro

        mat = self.get_interpolate_matrix(dest, src)
        with pvs.dat.vec_ro as src_vec, dest.dat.vec as dest_vec:
                mat.multTranspose(src_vec, dest_vec)
        dest.dat.data[:] = dest.dat.data_ro/M_dest.dat.data_ro

        return dest
