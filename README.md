# Utils for Firedrake

## Requirements 
1. Firedrake: https://github.com/firedrakeproject/firedrake 
2. tqdm: https://pypi.org/project/tqdm 
3. meshio: https://pypi.org/project/meshio/
<!--4. Gmsh: https://pypi.org/project/gmsh -->

## How to use

1. clone the files
    ```
    cd /my/path/to/repos
    git clone https://github.com/lrtfm/fdutils.git
    ```

2. compile the codes in activated firedrake envionment
    ```
    cd /my/path/to/repos/fdutils
    make develop
    ```
    
## List class/functions

### Class for evaluating Functions

1. `PointCloud`: A class to help evaluate functions on points.
    
    This class is used to evaluate functions for many times on 
    a group of points. It works in case that you give different points 
    on different mpi ranks. This is an alternative solution before
    `VertexOnlyMesh` in Firedrake supporting this feature.
    
    Exmaple code:
    ```
    from firedrake import *
    from fdutils import PointCloud

    mesh = RectangleMesh(10, 10, 1, 1)
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, 'CG', 1)
    f1 = Function(V).interpolate(x**2 + y**2)
    f2 = Function(V).interpolate(sin(x) + y**2)

    mesh2 = RectangleMesh(20, 20, 1, 1)
    points = mesh2.coordinates.dat.data_ro

    pc = PointCloud(mesh, points, tolerance=1e-12)
    v1 = pc.evaluate(f1)
    v2 = pc.evaluate(f2)
    ```
    
2. `PointArray`: Another class to help evaluate functions on points.
    
    This is a wrapper for at method of function to present the same
    interface as `PointCloud`.
    
3. `PointVOM`: A wrapper for `VortexOnlyMesh` to provide the same interface
    with `PointCloud`.
    
### Methods to compute errors of functions defined on different meshes

1. `fdutils.tools.errornorm`

### Prograssbar for parallel

1. Class `fdutils.ptqdm`

### Get information of mesh

1. `fdutils.meshutils`

