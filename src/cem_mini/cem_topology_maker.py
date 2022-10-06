import numpy as np

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


# how to generate the 2D grid pattern??

def cubic_grid_tower_topology(nx, ny, nz, no_bottom_dev=True):
    '''
    create a topology of 3D grid that represent a tower.
    
    ---- parameters ----
    nx: number of nodes in x direction, must >=2
    ny: number of nodes in y direction, must >=2
    nz: number of nodes in z direction, must >=2
    no_bottom_dev: if True, no deviation edges will be generated between the nodes of the bottom 2D grid
    
    ---- returns ----
    the topology
    
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
    
    size_xy=nx*ny
    # z-axis trail edges (i.e., tower columns)
    trail_paths = [[i+size_xy*z for z in range(nz)] for i in range(size_xy)]
    # x-axis deviation edges
    deviation_edges_x = [[(size_xy*z+y*nx+x, size_xy*z+y*nx+x+1) for y in range(ny) for x in range(nx-1)] for z in range(nz-1 if no_bottom_dev else nz)]
    # y-axis deviation edges
    deviation_edges_y = [[(size_xy*z+y*nx+x, size_xy*z+(y+1)*nx+x) for y in range(ny-1) for x in range(nx)] for z in range(nz-1 if no_bottom_dev else nz)]
    # slabs, sorted by floor order
    deviation_edges = [e for ex,ey in zip(deviation_edges_x, deviation_edges_y) for e in ex+ey] # list of edges
    dev_force_mag = [0.01] * len(deviation_edges)
    
    print(deviation_edges)
    # num of X-EDGES (edges in parallel with x axis)
    num_edges_x = (nx-1) * ny
    # num of Y-EDGES (edges in parallel with y axis)
    num_edges_y = nx * (ny-1)
    
    # get the edge indices of all symmetrical edges of the 2D grid
    indices_x, indices_y = get_symmetrical_edges(nx, ny)
    # ALL-EDGES = X-EDGES + Y-EDGES
    # convert indices of Y-EDGES to indices of ALL-EDGES
    indices_y = [[i+num_edges_x for i in iy] for iy in indices_y]
    
    # debug, validate the symmetrical edges
    print('trails', trail_paths, '\n')
    print('sym dev edges, X-EDGES', [[[deviation_edges[i + z*(num_edges_x+num_edges_y)] for i in ix] for ix in indices_x] for z in range(nz-1 if no_bottom_dev else nz)], '\n')
    print('sym dev edges, Y-EDGES', [[[deviation_edges[i + z*(num_edges_x+num_edges_y)] for i in iy] for iy in indices_y] for z in range(nz-1 if no_bottom_dev else nz)], '\n')
    
    # for each group of symmetrical edges in each floor, randomly generate the force magnitudes
    # i.e., specify the values of dev_force_mag[]
    
    