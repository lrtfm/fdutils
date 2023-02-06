# Utils for Firedrake

## Requirements 
1. Firedrake: https://github.com/firedrakeproject/firedrake 
2. tqdm: https://pypi.org/project/tqdm 
3. meshio: https://pypi.org/project/meshio/
<!--4. Gmsh: https://pypi.org/project/gmsh -->

## How to install

1. Activate the firedrake environment
    ```
    source </path/to/firedrake/bin/activate>
    ```

2. Clone the repo to the source directory of firedrake (or other path)
    ```
    cd $(dirname $(dirname $(which python)))/src
    git clone https://github.com/lrtfm/fdutils.git
    ```

3. Install
    ```
    cd $(dirname $(dirname $(which python)))/src/fdutils
    make develop
    ```
    
## List class/functions

### Class for evaluating Functions

1. `PointCloud`: A class to help evaluate functions on points.
    
    This class is used to evaluate functions for many times on 
    a group of points. It works in case that you give different points 
    on different mpi ranks. This is an alternative solution before
    `VertexOnlyMesh` in Firedrake supporting this feature.
    
    Example code on interpolating function `f1` on mesh `m1` to function `f2` on mesh `m2` 
    ```
    import firedrake as fd
    from fdutils import PointCloud
    from fdutils.tools import get_nodes_coords
    import matplotlib.pyplot as plt

    m1 = fd.RectangleMesh(10, 10, 1, 1)
    V1 = fd.FunctionSpace(m1, 'CG', 2)
    x, y = fd.SpatialCoordinate(m1)
    f1 = fd.Function(V1).interpolate(x**2 + y**2)

    m2 = fd.RectangleMesh(20, 20, 1, 1)
    V2 = fd.FunctionSpace(m2, 'CG', 3)
    f2 = fd.Function(V2)

    points = get_nodes_coords(f2)
    pc = PointCloud(m1, points, tolerance=1e-12)
    f2.dat.data_with_halos[:] = pc.evaluate(f1)

    fig, ax = plt.subplots(1, 2, figsize=[8, 4], subplot_kw=dict(projection='3d'))
    fd.trisurf(f1, axes=ax[0])
    fd.trisurf(f2, axes=ax[1])
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

