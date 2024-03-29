from fdutils import PointCloud
from firedrake import Function, FunctionSpace, \
                      VectorFunctionSpace, TensorFunctionSpace,\
                      TrialFunction, TestFunction, \
                      dx, inner, dot, \
                      assemble
from firedrake.petsc import PETSc
from finat.ufl import FiniteElement, TensorProductElement
import numpy as np


def check_Pn_space(V: FunctionSpace):
    ele = V.ufl_element()
    if not (ele.family() == 'Lagrange' and isinstance(ele, FiniteElement)):
        raise NotImplementedError


def get_lump_mass_matrix(V: FunctionSpace):
    check_Pn_space(V)
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
    __fun_caches = {}

    def __init__(self, *, native_transfers=None, use_averaging=True):
        PETSc.Sys.Print(f'Init {type(self)}')
        self.pc_caches = NonnestedTransferManager.__pc_caches
        self.mat_caches = NonnestedTransferManager.__mat_caches
        self.fun_caches = NonnestedTransferManager.__fun_caches
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

    @PETSc.Log.EventDecorator()
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
    @PETSc.Log.EventDecorator()
    def prolong(self, src: Function, dest: Function):
        self.interpolate(src, dest)

    # Transfer the fine solution to the coarse space
    @PETSc.Log.EventDecorator()
    def inject(self, src: Function, dest: Function):
        self.interpolate(src, dest)

    def get_function(self, V):
        if V in self.fun_caches:
            fun = self.fun_caches[V]
        else:
            fun = Function(V)
            self.fun_caches[V] = fun

        return fun

    # Transfer the fine residual to the coarse space
    # Here we assume src and dest are P1 Function
    @PETSc.Log.EventDecorator()
    def restrict(self, src: Function, dest: Function):
        mat = self.get_interpolate_matrix(dest, src)
        with src.dat.vec_ro as src_vec, dest.dat.vec as val_vec:
                mat.multTranspose(src_vec, val_vec)

        return dest

# patch mg for using galerkin
import firedrake
import firedrake.mg.ufl_utils as ufl_utils
from firedrake.mg.ufl_utils import Injection, Interpolation


def create_interpolation(dmc, dmf):

    cctx = firedrake.dmhooks.get_appctx(dmc)
    fctx = firedrake.dmhooks.get_appctx(dmf)

    manager = firedrake.dmhooks.get_transfer_manager(dmf)

    V_c = cctx._problem.u.function_space()
    V_f = fctx._problem.u.function_space()

    if isinstance(manager, NonnestedTransferManager):
        f_c = Function(V_c)
        f_f = Function(V_f)
        mat = manager.get_interpolate_matrix(f_c, f_f)
        return mat, None

    row_size = V_f.dof_dset.layout_vec.getSizes()
    col_size = V_c.dof_dset.layout_vec.getSizes()

    cfn = firedrake.Function(V_c)
    ffn = firedrake.Function(V_f)
    cbcs = cctx._problem.bcs
    fbcs = fctx._problem.bcs

    ctx = Interpolation(cfn, ffn, manager, cbcs, fbcs)
    mat = PETSc.Mat().create(comm=dmc.comm)
    mat.setSizes((row_size, col_size))
    mat.setType(mat.Type.PYTHON)
    mat.setPythonContext(ctx)
    mat.setUp()
    return mat, None


def create_injection(dmc, dmf):
    cctx = firedrake.dmhooks.get_appctx(dmc)
    fctx = firedrake.dmhooks.get_appctx(dmf)

    manager = firedrake.dmhooks.get_transfer_manager(dmf)

    V_c = cctx._problem.u.function_space()
    V_f = fctx._problem.u.function_space()

    if isinstance(manager, NonnestedTransferManager):
        f_c = Function(V_c)
        f_f = Function(V_f)
        mat_int = manager.get_interpolate_matrix(f_c, f_f)
        mat = PETSc.Mat().createTranspose(mat_int)
        return mat

    row_size = V_f.dof_dset.layout_vec.getSizes()
    col_size = V_c.dof_dset.layout_vec.getSizes()

    cfn = firedrake.Function(V_c)
    ffn = firedrake.Function(V_f)
    cbcs = cctx._problem.bcs

    ctx = Injection(cfn, ffn, manager, cbcs)
    mat = PETSc.Mat().create(comm=dmc.comm)
    mat.setSizes((row_size, col_size))
    mat.setType(mat.Type.PYTHON)
    mat.setPythonContext(ctx)
    mat.setUp()
    return mat


# # TODO: When using injection we should add boundry conditon
# #       How to apply injection and bc within one matrix?
# ufl_utils.create_injection = create_injection
# ufl_utils.create_interpolation = create_interpolation
