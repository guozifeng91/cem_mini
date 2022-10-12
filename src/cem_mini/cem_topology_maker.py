import numpy as np
import cem_mini

def get_symmetrical_edges(nx, ny, mode_b=False):
    '''
    this function returns indices of edges that are symmetrical along in both x and y axis.

    for example in a 4 x 4 grid where there are 3 x 4 edges in parallel with x axis (noted as X-EDGES),
    and 4 x 3 edges in parallel with y axis (denoted as Y-EDGES):

    0 - 1 - 2 - 3
    |   |   |   |
    4 - 5 - 6 - 7
    |   |   |   |
    . - . - . - .
    |   |   |   |
    12 -13 -14 -15

    the X-EDGES are
    0  1  2
    3  4  5
    6  7  8
    9  10 11

    the Y-EDGES are (mode a)
    0  1  2  3
    4  5  6  7
    8  9  10  11

    the Y-EDGES are (mode b)
    0  3  6  9
    1  4  7  10
    2  5  8  11

    the symmetric X-EDGES are
    [[0, 2, 9, 11], [1, 10], [3, 5, 6, 8], [4,7]]

    the symmetric Y-EDGES are
    [[0, 3, 8, 11], [1, 2, 9, 10], [4, 7], [5, 6]]
    '''
#     num_edges_x = (nx-1) * ny
#     num_edges_y = nx * (ny-1)

    # get the index of horizontally sym. X-EDGES
    # e.g. in the above example, 0->2, 1->1, 3->5, 4->4, ..., 9->11, 10->10
    sym_horizontal_x = lambda i: i - i%(nx-1) + (nx - 2) - i%(nx - 1) # i - i%(nx-1) is the starting index of each row

    # get the index of horizontally sym. X-EDGES
    # e.g. in the above example, 0->9, 1->10, 3->6, 4->7, ..., 2->11, 5->8
    sym_vertical_x = lambda i: (ny-1)*(nx-1) + i%(nx-1)*2 - i

    if mode_b:
        # get the index of horizontally sym. Y-EDGES
        sym_vertical_y = lambda i: i - i%(ny-1) + (ny - 2) - i%(ny - 1)
        # get the index of horizontally sym. Y-EDGES
        sym_horizontal_y = lambda i: (ny-1)*(nx-1) + i%(ny-1)*2 - i
    else:
        sym_horizontal_y = lambda i: i - i%nx + (nx - 1) - i%nx # i - i%nx is the starting index of each row
        sym_vertical_y = lambda i: (ny-2)*nx + i%nx*2 - i
#         sym_vertical_y = lambda i:i

    # all symmetrical X-EDGES
    indices_x=[i+j*(nx-1) for i in range((nx-1)//2 + (nx-1)%2) for j in range(ny//2 + ny%2)]
    if mode_b:
        indices_y=[i*(ny-1)+j for i in range(nx//2 + nx%2) for j in range((ny-1)//2 + (ny-1)%2)]
    else:
        indices_y=[i+j*nx for i in range(nx//2 + nx%2) for j in range((ny-1)//2 + (ny-1)%2)]

    indices_x=[[i, sym_vertical_x(i), sym_horizontal_x(i), sym_vertical_x(sym_horizontal_x(i))] for i in indices_x]
    indices_y=[[i, sym_vertical_y(i), sym_horizontal_y(i), sym_vertical_y(sym_horizontal_y(i))] for i in indices_y]

    indices_x=[sorted(list(set(i))) for i in indices_x]
    indices_y=[sorted(list(set(i))) for i in indices_y]

    return indices_x, indices_y

def uniform_sampler(low,high,seed=None):
    rnd = np.random.default_rng(seed)
    def call():
        return rnd.uniform(low, high)
    return call

def normal_sampler(mean,std,seed=None):
    rnd = np.random.default_rng(seed)
    def call():
        return rnd.normal(mean, std)
    return call

def discrete_sampler(items,seed=None):
    rnd = np.random.default_rng(seed)
    def call():
        return items[rnd.integers(len(items))]
    return call

def constant_sampler(c):
    def call():
        return c
    return call

def bool_sampler(ratio=0.5, seed=None):
    rnd = np.random.default_rng(seed)
    def call():
        return rnd.uniform(0,1)<ratio
    return call

samplers={uniform_sampler, normal_sampler, discrete_sampler, constant_sampler, bool_sampler}

def cubic_grid_graph(nx,ny,nz, no_bottom_dev=False):
    '''
    create a topology of 3D grid

    ---- parameters ----
    nx: number of nodes in x direction, must >=2
    ny: number of nodes in y direction, must >=2
    nz: number of nodes in z direction, must >=2
    no_bottom_dev: if True, no deviation edges will be generated between the nodes of the bottom 2D grid

    ---- returns ----
    the graph for topology making

    ---- notes ----
    for each 2D grid of the xy plane, the nodes are ordered as

    0 - 1 - 2 - 3
    |   |   |   |
    4 - 5 - 6 - 7
    |   |   |   |
    8 - 9 - 10 -11
    |   |   |   |
    12 -13 -14 -15

    where - are edges in parallel with x axis (denoted as X-EDGES)
    and | are edges in parallel with y axis (denoted as Y-EDGES)

    in this example, X-EDGES will be ordered as

    0 1 2
    3 4 5
    6 7 8
    9 10 11

    and Y-EDGES will be ordered as

    0 1 2 3
    4 5 6 7
    8 9 10 11
    '''
    if nx<2 or ny<2 or nz<2:
        raise Exception("number of nodes must >= 2 in x, y, and z directions")

    n_xy=nx*ny
    # z-axis trail edges (i.e., tower columns)
    trail_paths = [[i+n_xy*z for z in range(nz)] for i in range(n_xy)]
    # x-axis deviation edges
    deviation_edges_x = [[(n_xy*z+y*nx+x, n_xy*z+y*nx+x+1) for y in range(ny) for x in range(nx-1)] for z in range(nz-1 if no_bottom_dev else nz)]
    # y-axis deviation edges
    deviation_edges_y = [[(n_xy*z+y*nx+x, n_xy*z+(y+1)*nx+x) for y in range(ny-1) for x in range(nx)] for z in range(nz-1 if no_bottom_dev else nz)]
    # slabs, sorted by floor order
    deviation_edges = [e for ex,ey in zip(deviation_edges_x, deviation_edges_y) for e in ex+ey] # list of edges

    return {'num_nodes':nx*ny*nz,
            'trail_paths':trail_paths,
            'deviation_edges':deviation_edges,
            'deviation_edges_x':deviation_edges_x,
            'deviation_edges_y':deviation_edges_y}

def cubic_grid_tower_topology(nx, ny, nz,
                              dev_mag_default=constant_sampler(0.01),
                              dev_mag_pattern=discrete_sampler([-0.5, 0.5]),
                              sym_pattern_select=bool_sampler(0.5),
                              trail_len=constant_sampler(-3),
                              sizex=discrete_sampler([3,4,5]),
                              sizey=discrete_sampler([3,4,5]),
                              floorheight=discrete_sampler([3,4,5]),
                              loads=[0,0,-10],
                              no_bottom_dev=True):
    '''
    create a topology of 3D grid that represent a tower.

    ---- parameters ----
    nx: number of nodes in x direction, must >=2
    ny: number of nodes in y direction, must >=2
    nz: number of nodes in z direction, must >=2

    dev_mag_default: a callable which returns a random sampling of the default deviation force magitude
    dev_mag_pattern: a callable which returns a random sampling of the deviation force magitude
    sym_pattern_select: a callable which returns a boolean random sampling whether
                        to accept a group of symmetrical edges for assigning random values
    trail_len: a callable which returns a random sampling of the trail edge length
    sizex: a callable which returns a random sampling of the cell size x
    sizey: a callable which returns a random sampling of the cell size y
    floorheight: a callable which returns a random sampling of the floor height
                 if floorheight is given, it will generate constrained planes to
                 the result topology and potentially override the trail_len specification

    no_bottom_dev: if True, no deviation edges will be generated between the nodes of the bottom 2D grid

    ---- returns ----
    the topology
    '''
    graph = cubic_grid_graph(nx,ny,nz,no_bottom_dev)
    num_nodes=graph['num_nodes']
    trail_paths=graph['trail_paths']
    deviation_edges=graph['deviation_edges']
    dev_force_mag = [dev_mag_default() for _ in range(len(deviation_edges))] # default dev force mag

    # num of X-EDGES (edges in parallel with x axis)
    num_edges_x = (nx-1) * ny
    # num of Y-EDGES (edges in parallel with y axis)
    num_edges_y = nx * (ny-1)

    # get the edge indices of all symmetrical edges of the 2D grid
    indices_x, indices_y = get_symmetrical_edges(nx, ny)
    # ALL-EDGES = X-EDGES + Y-EDGES
    # convert indices of Y-EDGES to indices of ALL-EDGES
    indices_y = [[i+num_edges_x for i in iy] for iy in indices_y]

    # for each group of symmetrical edges in each floor, randomly generate the force magnitudes
    # i.e., specify the values of dev_force_mag[]
    for z in range(nz-1 if no_bottom_dev else nz):
        i_offset = z*(num_edges_x+num_edges_y)
        for ix in indices_x: # for each symmetrical group
            if sym_pattern_select(): # group selected?
                value=dev_mag_pattern() * (z+1) # make a sample
                for i in ix: # assign the sample value to the group
                    dev_force_mag[i + i_offset] = value

        for iy in indices_y: # for each symmetrical group
            if sym_pattern_select(): # group selected?
                value=dev_mag_pattern() * (z+1) # make a sample
                for i in iy: # assign the sample value to the group
                    dev_force_mag[i + i_offset] = value

#     print(dev_force_mag)

    T=cem_mini.create_topology(num_nodes)
    cem_mini.set_trail_paths(T, trail_paths, [[trail_len()] * (len(t)-1) for t in trail_paths])
    cem_mini.set_deviation_edges(T,deviation_edges,dev_force_mag)

    # cell size
    cx=sizex()
    cy=sizey()

    if floorheight is None:
        cem_mini.set_original_node_positions(T,{j*nx+i:[i*cx, j*cy, (nz-1)*3] for i in range(nx) for j in range(ny)})
    else:
        node_z=np.cumsum([floorheight() if z>0 else 0 for z in range(nz)])[::-1]

        cem_mini.set_original_node_positions(T,{j*nx+i:[i*cx, j*cy, node_z[0]] for i in range(nx) for j in range(ny)})
        cem_mini.set_constrained_planes(T,{(k*nx*ny+j*nx+i):[i*cx, j*cy, node_z[k], 0, 0, 1] for i in range(nx) for j in range(ny) for k in range(nz)})

    cem_mini.set_node_loads(T,{i:loads for i in range(nx*ny*nz)})

    return T

def axial_graph(nt, nr, dev_pattern=None, no_outer_dev=False):
    '''
    produce an axial graph like this (a 4 x 2 example):

          4
          0
     5,1     3,7
          2
          6

    ---- parameters ----

    nt: number of trails, must >= 2
    nr: number of nodes per trail, must >= 2
    dev_pattern: list of tuples that indicates the two trail paths connected
                 by deviation edges. If None (default) is given, the deviation
                 edges will be created between adjacent trail paths
                 (i.e., (0, 1), (1, 2), ..., (nt-1, 0))

    no_outer_dev: if True, no deviation edges will be created between the support nodes
    ---- returns ----

    the graph for making the topology
    '''
    if nt < 2 or nr < 2:
        raise Exception('num of trails (nt) or number of nodes per trail (nr) must >= 2')

    if dev_pattern is None:
        #
        dev_pattern=[(i,(i+1)%nt) for i in range(nt)]

    trail_paths = [[j*nt + i for j in range(nr)] for i in range(nt)]
    deviation_edges = [(i*nt+u, i*nt+v) for i in range((nr-1) if no_outer_dev else nr) for u, v in dev_pattern]
#     deviation_edges = [(i*nt+j, i*nt+(j+1)%nt) for i in range((nr-1) if no_outer_dev else nr) for j in range(nt)]

    return {'num_nodes':nt*nr, 'trail_paths':trail_paths, 'deviation_edges':deviation_edges}
