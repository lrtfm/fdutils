import numpy as np

def evaluate_fun(fun, points, tolerance=None):
    """ evaluate function on points, a wraper for at.
    
    The points can be different on different process.
    """
    comm = fun.comm
    
    for rank in range(comm.size):
        points_curr = comm.bcast(points, root=rank)
        value_ = fun.at(points_curr, dont_raise=True, tolerance=tolerance)

        if comm.rank == rank:
            value = value_

    v_size = fun.ufl_element().value_size()
    noneindex =[i for i,v in enumerate(value) if v is None]
    if v_size > 1:
        zeros = [0. for _ in range(v_size)]
        for i in noneindex:
            value[i] = np.array(zeros)
    else:
        for i in noneindex:
            value[i] = 0  
    
    return np.array(value)


class PointArray():
    """
    A class to help evaluate functions on points.
    
    This is a wrapper for at method of function to present the same
    interface as `PointCloud`.
    
    Example code:
        ```
        from firedrake import *
        from fdutils import *

        mesh = RectangleMesh(10, 10, 1, 1)
        x, y = SpatialCoordinate(mesh)
        V = FunctionSpace(mesh, 'CG', 1)
        f1 = Function(V).interpolate(x**2 + y**2)
        f2 = Function(V).interpolate(sin(x) + y**2)

        mesh2 = RectangleMesh(20, 20, 1, 1)
        points = mesh2.coordinates.dat.data_ro

        pc = PointArray(mesh, points, tolerance=1e-12)
        v1 = pc.evaluate(f1)
        v2 = pc.evaluate(f2)
        ```
    """
    
    def __init__(self, mesh, points, tolerance=None):
        self.points = points
        self.tolerance = tolerance
        
    def evaluate(self, fun):
        return evaluate_fun(fun, self.points, tolerance=self.tolerance)
    