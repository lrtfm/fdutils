import numpy as np
import meshio

__all__ = ['get_mesh_size', 'read_msh_file', 'circumradius']

def get_mesh_size(mesh, file=None):
    """
    Compute mesh size (max diameter)
    
    Args:
        mesh: mesh in Firedrake
        file: path to save the diameter function, default is None
        
    Returns:
        max diameter in the mesh
    """
    
    V = FunctionSpace(mesh, 'DG', 0)
    d = CellDiameter(mesh)
    f = Function(V)
    f.interpolate(d)
    
    if file is not None:
        File(file).write(f)
        
    with f.dat.vec_ro as vec:
        _, value = vec.max()
        
    return value

def read_msh_file(file, scale=1):
    """
    Read mesh file and remove orphaned data
    
    Args:
        file: file name
        scale: scale the points value by a number, default is 1.
        
    Returns:
        A tuple (mesh, cells, points)
    """
    
    mesh = meshio.read(file)
    mesh.remove_lower_dimensional_cells()
    mesh.remove_orphaned_nodes()
    mesh.cell_data['gmsh:physical'] = mesh.cell_data['gmsh:geometrical']
    mesh.points = mesh.points/scale
    cells = mesh.cells[0].data
    points = mesh.points
    
    return mesh, cells, points

def circumradius(cells, points):
    """
    Compute the radius of circumscribed circle and inscribed circles for each element.
    
    Args:
        cells: a list of cells, each cell is an array of the index of the vertex
        points: coordinates of every points
        
    Returns:
        A tuple of radius of circumscribed circle and inscribed circles.
        
    Examples:
        Rs, rs = circumradius([[0, 1, 2, 3]], [[-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        Rs, rs = circumradius([[0, 1, 2, 3]], [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        Rs, rs = circumradius([[0, 1, 2]], [[0, 0], [1, 0], [0, 1]])
        Rs, rs = circumradius([[0, 1, 2]], [[-1, 0], [1, 0], [0, 1]])
    """
    
    n = len(cells)
    dim = len(cells[0]) - 1
    
    points = np.array(points)
    Rs = np.zeros(n)
    rs = np.zeros(n)

    if dim == 3:
        idx = np.array([[0, 1], [2, 3], [0, 2], [1, 3], [0, 3], [1, 2]])
        idx2 = np.array([[0, 1, 2], [0, 2, 3], [1, 2, 3], [0, 1, 3]])
        ss = np.zeros(len(idx2))
        for i in range(n):
            ps = points[cells[i]]

            ls = np.linalg.norm(ps[idx[:, 0]] - ps[idx[:, 1]], axis=1)
            a = ls[0]*ls[1]
            b = ls[2]*ls[3]
            c = ls[4]*ls[5]
            p = (a + b + c)/2
            v = np.abs(np.linalg.det(np.hstack([ps, np.array([[1], [1], [1], [1]])])))
            
            for j, _ in enumerate(idx2):
                ss[j] = np.linalg.norm(np.cross(ps[_[2]] - ps[_[0]], ps[_[2]] - ps[_[1]]))/2
            Rs[i] = np.sqrt(p*(p-a)*(p-b)*(p-c))/v
            rs[i] = v/sum(ss)/2
    elif dim == 2:
        idx = np.array([[0, 1], [1, 2], [0, 2]])
        
        for i in range(n):
            ps = points[cells[i]]
            ls = np.linalg.norm(ps[idx[:, 0]] - ps[idx[:, 1]], axis=1)
            a = ls[0]
            b = ls[1]
            c = ls[2]
            p = (a + b + c)/2
            
            Rs[i] = a*b*c/(4*np.sqrt(p*(p-a)*(p-b)*(p-c)))
            rs[i] = np.sqrt((p-a)*(p-b)*(p-c)/p)
    else: 
        raise NotImplementedError('We don\'t support meshes with dim = %d' % dim)

    return Rs, rs

def write2dat(cells, points, file):
    with open(file, "w") as f:
        f.write("""DIM: 3
DIM_OF_WORLD: 3
number of vertices: %d
number of elements: %d\n"""%(len(points), len(cells)))

        f.write('\nvertex coordinates:\n')
        for i in range(len(points)):
            f.write("%s\t%s\t%s\n"%(points[i, 0], points[i, 1], points[i, 2]))
        f.write('\nelement vertices:\n')
        for i in range(len(cells)):
            f.write("%s\t%s\t%s\t%s\n"%(cells[i, 0], cells[i, 1], cells[i, 2], cells[i, 3]))

def read_from_dat(file):
    with open(file, "r") as f:
        data = np.zeros(4, np.int32)
        for i in range(4):
            line = f.readline().rstrip('\n')
            data[i] = int(line.split(':')[1])

        nps = data[2]
        ncs = data[3]

        while 1:
            line = f.readline()# skip blank
            if line.startswith('vertex coordinates:'):
                break

        points = np.zeros([nps, 3])
        for i in range(nps):
            line = f.readline().rstrip('\n')
            points[i] = [float(_) for _ in line.split('\t')]

        while 1:
            line = f.readline()
            if line.startswith('element vertices:'):
                break

        cells = np.zeros([ncs, 4], np.int32)
        for i in range(ncs):
            line = f.readline().rstrip('\n')
            cells[i] = [int(_) for _ in line.split('\t')]
        
    return cells, points


def check_close_points(mesh, tol=None):

    m, n = mesh.points.shape

    remap = np.zeros(m, dtype=np.int32)

    tol = tol or 1e-6

    pidx = []
    for i in range(m):
        d = np.linalg.norm(mesh.points - mesh.points[i], axis=1)
        index = np.where(d < tol)[0]

        if len(index) > 1 and index[0] == i:
            pidx = pidx + list(index)

        remap[i] = index[0]
    
    return remap, pidx

def remap_vertex(mesh, remap):
    cells = mesh.cells[0]
    data = remap[cells.data]

    cellindex = []
    for i in range(len(cells)):
        if len(np.unique(data[i])) == 4:
            cellindex.append(i)

    new_data = data[cellindex]
    vertex = np.unique(new_data.flatten())

    vertex_map = np.zeros(max(vertex)+1, dtype=np.int32)
    vertex_map[vertex] = range(len(vertex))

    cell_node_list = vertex_map[new_data]
    points = mesh.points[vertex]
    
    return cell_node_list, points

def clean_close_points(mesh, tol=1e-4):
    remap, pidx = check_close_points(mesh, tol=1e-4)
    if len(pidx) == 0:
        cell_node_list = mesh.cells[0].data
        points = mesh.points
    else:
        cell_node_list, points = remap_vertex(mesh, remap)
    
    if cell_node_list.shape ==  mesh.cells[0].data.shape and points.shape == mesh.points.shape:
        print('We do not found close points with tol = %s'%tol)
    else:
        print('There are points are close with tol = %s'%tol)
    
    return cell_node_list, points

def get_maxh_3d(cell_node_list, points):
    ds = np.zeros(len(cell_node_list))
    idx = np.array( [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]] ) - 1
    for i, one in enumerate(cell_node_list):
        v = points[one[idx[:, 0]], :] - points[one[idx[:, 1]], :]
        d = np.linalg.norm(v, axis=1)
        ds[i] = d.max()
    return ds


