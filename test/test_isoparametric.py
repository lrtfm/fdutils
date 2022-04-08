from firedrake import *
from firedrake.petsc import PETSc
import numpy as np
import argparse

from fdutils import PointCloud

def get_coords(mesh, degree=2):
    '''
    Map the points on the boundary to the circle.
    '''
    V_c = VectorFunctionSpace(mesh, 'CG', degree=degree)
    coords = Function(V_c)
    coords.interpolate(SpatialCoordinate(mesh))
    coords_bak = Function(coords)
    x, y = SpatialCoordinate(mesh)
    r = sqrt(x**2 + y**2)
    bc = DirichletBC(V_c, as_vector((x/r, y/r)), 'on_boundary')
    bc.apply(coords)
    
    return coords_bak, coords, bc.nodes

def get_coords_3(mesh):
    '''
    Modified the coordinates according to (M. Lenoir 1986 SINUM)
    '''
    coords2_bak, coords2, bc_nodes2 = get_coords(mesh, 2)
    coords3_bak, coords3, bc_nodes3 = get_coords(mesh, 3)

    cell_node_map3 = coords3.cell_node_map().values
    cell_node_map2 = coords2.cell_node_map().values

    coords3_new = Function(coords3)

    m_idxs = []
    c_idxs = []
    for i, nodes3 in enumerate(cell_node_map3):
        c_index = nodes3[-1]
        nodes2 = cell_node_map2[i]
        m_index = -1
        for k in nodes2[3:]:
            if k in bc_nodes2:
                m_index = k
                break

        if m_index == -1:
            continue
        else:
            m_idxs.append(m_index)
            c_idxs.append(c_index)
            
    m_idxs = np.array(m_idxs, dtype=np.int32)
    c_idxs = np.array(c_idxs, dtype=np.int32)
    
    pc1 = PointCloud(mesh, coords2_bak.dat.data[m_idxs], tolerance=1e-12)
    pc2 = PointCloud(mesh, coords3_bak.dat.data[c_idxs], tolerance=1e-12)
    
    v_e_mid = pc1.evaluate(coords3)
    shift = v_e_mid - coords2.dat.data[m_idxs]
    
    v_c_mid = pc2.evaluate(coords2)
    coords3_new.dat.data[c_idxs] = v_c_mid + 0.5**3*shift
    
    return coords3_new


def solve_possion(n=2, k=2, iso=False, use_Lenoir=False, u_handle=lambda x, y: 1 - x**2 - y**2):
    """ Solve possion problem with/without iso element
    
    iso: use iso element
    use_Lenoir: use Lenoir's method or not. If false, we only move boundy points. 
    """
    mesh = UnitDiskMesh(refinement_level=n)
    if iso and k > 1:
        if k == 2:
            _, coords, _ = get_coords(mesh, degree=2)
        elif k == 3:
            if use_Lenoir:
                coords = get_coords_3(mesh)
            else:
                _, coords, _ = get_coords(mesh, degree=3)
        else:
            NotImplementedError('Unsupport for k = %d'%k)
        
        mesh = Mesh(coords)

    mesh.topology_dm.viewFromOptions('-dm_view')

    x, y = SpatialCoordinate(mesh)
    u_exact = u_handle(x, y) 

    f = - div(grad(u_exact))

    V = FunctionSpace(mesh, 'CG', degree=k)
    u = TrialFunction(V)
    v = TestFunction(V)

    bc = DirichletBC(V, 0, 'on_boundary')


    a = dot(grad(u), grad(v))*dx
    L = f*v*dx(degree=10)

    sol = Function(V)

    solve(a == L, sol, bcs=bc)

    err = errornorm(u_exact, sol, norm_type='L2')
    err_H1 = errornorm(u_exact, sol, norm_type='H1')
    u_int = Function(V).interpolate(u_exact)
    err_int = errornorm(u_exact, u_int, norm_type='L2')
    err_int_H1 = errornorm(u_exact, u_int, norm_type='H1')

    return err, err_H1, err_int, err_int_H1

def main(sdeg=1, edeg=3, u_handle = lambda x, y: 1 - x**2 - y**2):
    for degree in range(sdeg, edeg+1):
        for iso in [False, True]:
            if degree == 3 and iso is True:
                Lenoirs = [False, True]
            else:
                Lenoirs = [False]
            for use_Lenoir in Lenoirs:
                errs = []
                errs_int = []
                errs_H1 = []
                errs_int_H1 = []
                # k = 2
                # iso = True
                for i in range(2, 6):
                    err, err_H1, err_int, err_int_H1 = solve_possion(i, degree, iso=iso, use_Lenoir=use_Lenoir, u_handle=u_handle)
                    errs.append(err)
                    errs_int.append(err_int)
                    errs_H1.append(err_H1)
                    errs_int_H1.append(err_int_H1)

                errs = np.array(errs)
                errs_H1 = np.array(errs_H1)
                errs_int = np.array(errs_int)
                errs_int_H1 = np.array(errs_int_H1)
                PETSc.Sys.Print('FEM space order %d, Isoparametric %s, Lenoir %s'%(degree, iso, use_Lenoir))
                PETSc.Sys.Print('\t' + '-'*80)
                PETSc.Sys.Print('\tH1 error:', format_error(errs_H1))
                PETSc.Sys.Print('\tH1 order:', format_order(np.log(errs_H1[:-1]/errs_H1[1:])/np.log(2)))
                PETSc.Sys.Print('\tH1 error:', format_error(errs))
                PETSc.Sys.Print('\tL2 order:', format_order(np.log(errs[:-1]/errs[1:])/np.log(2)))
                PETSc.Sys.Print('\t' + '-'*80)
                PETSc.Sys.Print('\tInterpolate H1 error:', format_error(errs_int_H1))
                PETSc.Sys.Print('\tInterpolate H1 order:', format_order(np.log(errs_int_H1[:-1]/errs_int_H1[1:])/np.log(2)))
                PETSc.Sys.Print('\tInterpolate L2 error:', format_error(errs_int))
                PETSc.Sys.Print('\tInterpolate L2 order:', format_order(np.log(errs_int[:-1]/errs_int[1:])/np.log(2)))
                PETSc.Sys.Print('\t'+'-'*80)
                
def format_error(error):
    return format_order(error, onefstr='%10.2e', space='')

def format_order(order, onefstr='%10.2f', space=' '*12):
    n = len(order)
    fstr = space + onefstr + (' '*2 + onefstr)*(n-1)
    return fstr%tuple(order)

if __name__ == '__main__':
    test_cases = {
        '1': lambda x, y: 1 - (x**2 + y**2),
        '2': lambda x, y: 1 - (x**2 + y**2)**2, 
        '3': lambda x, y: 1 - sqrt(x**2 + y**2)
    }
    case_strs = {
        '1': '1 - (x^2 + y^2)',
        '2': '1 - (x^2 + y^2)^2', 
        '3': '1 - (x^2 + y^2)^(1/2)'
    }
    
    parser = argparse.ArgumentParser(description='Test Isoparametric')
    parser.add_argument('-s', '--start_degree', metavar='start_degree', type=int, default=3,
                        help='Start degree of element to test. Default: 3.')
    parser.add_argument('-e', '--end_degree', metavar='end_degree', type=int, default=3,
                        help='End degree of element to test. Default: 3.')
    parser.add_argument('-c', '--case', metavar='case', type=str, default='2',
                        help='''Chose test case:
1: u = 1 - (x^2 + y^2);
2: u = 1 - (x^2 + y^2)^2;
3: u = 1 - (x^2 + y^2)^(1/2).
Default value: 2.
''')
    args, _ = parser.parse_known_args()
    sdeg=args.start_degree
    edeg=args.end_degree
    u_handle = test_cases[args.case]
    funstr = case_strs[args.case]
    PETSc.Sys.Print('\n\n')
    PETSc.Sys.Print('Exact solution: ' + funstr)
    main(sdeg=sdeg, edeg=edeg, u_handle=u_handle)
    PETSc.Sys.Print('\n\n')

    

################################################################################
#
# Results for Case 2
#
# Exact solution: 1 - (x^2 + y^2)^2
# FEM space order 3, Isoparametric False, Lenoir False
#         --------------------------------------------------------------------------------
#         H1 error:   1.17e-01    4.13e-02    1.46e-02    5.13e-03
#         H1 order:                   1.50        1.50        1.50
#         H1 error:   2.65e-02    6.58e-03    1.63e-03    4.07e-04
#         L2 order:                   2.01        2.01        2.01
#         --------------------------------------------------------------------------------
#         Interpolate H1 error:   2.86e-03    3.72e-04    4.70e-05    5.90e-06
#         Interpolate H1 order:                   2.95        2.98        2.99
#         Interpolate L2 error:   7.41e-05    4.81e-06    3.04e-07    1.91e-08
#         Interpolate L2 order:                   3.95        3.98        4.00
#         --------------------------------------------------------------------------------
# FEM space order 3, Isoparametric True, Lenoir False
#         --------------------------------------------------------------------------------
#         H1 error:   2.01e-02    3.81e-03    6.96e-04    1.25e-04
#         H1 order:                   2.40        2.45        2.48
#         H1 error:   4.09e-04    3.88e-05    3.56e-06    3.20e-07
#         L2 order:                   3.40        3.45        3.47
#         --------------------------------------------------------------------------------
#         Interpolate H1 error:   2.30e-02    4.28e-03    7.75e-04    1.39e-04
#         Interpolate H1 order:                   2.43        2.47        2.48
#         Interpolate L2 error:   5.14e-04    4.68e-05    4.19e-06    3.73e-07
#         Interpolate L2 order:                   3.46        3.48        3.49
#         --------------------------------------------------------------------------------
# FEM space order 3, Isoparametric True, Lenoir True
#         --------------------------------------------------------------------------------
#         H1 error:   3.11e-03    3.66e-04    4.35e-05    5.27e-06
#         H1 order:                   3.09        3.07        3.05
#         H1 error:   7.97e-05    4.69e-06    2.79e-07    1.69e-08
#         L2 order:                   4.09        4.07        4.05
#         --------------------------------------------------------------------------------
#         Interpolate H1 error:   3.65e-03    4.26e-04    5.07e-05    6.14e-06
#         Interpolate H1 order:                   3.10        3.07        3.05
#         Interpolate L2 error:   8.85e-05    5.28e-06    3.19e-07    1.95e-08
#         Interpolate L2 order:                   4.07        4.05        4.03
#         --------------------------------------------------------------------------------
#
################################################################################

