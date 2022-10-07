'''
the same cem examples with https://github.com/arpastrana/compas_cem/tree/main/examples
'''

import matplotlib.pyplot as plt

# from .cem_mini import *
# from .cem_plots import *
# from .cem_topology_maker import *

import cem_mini
# import cem_plots
import cem_mini.cem_topology_maker as cem_topology_maker

def _make_plots(F,view='2D-XY'):
    plt.figure(figsize=(8,8))
    ax=plt.axes(projection='3d')
    cem_mini.plot_cem_form(ax,F['coords'],F['edges'],F['edge_forces'],F['loads'], view=view, load_len_scale=0.2, thickness_base=0.1,thickness_ratio=2)
    ax.axis('on')
    plt.show()

def quick_start_2d():
    T=cem_mini.create_topology(4)
    
    cem_mini.set_trail_paths(T, [[1,0],[2,3]],[[-1],[-1]])
    cem_mini.set_deviation_edges(T,[[1,2]],[-1])
    
    cem_mini.set_original_node_positions(T,{1:[1,0,0], 2:[2.5,0,0]})
    cem_mini.set_node_loads(T,{1:[0,-1,0],2:[0,-1,0]})
    
    # solve the CEM
    F, Fc= cem_mini.CEM(T,1e-6)
    
    # visualize form diagram
    _make_plots(F)
    
    return T, F, Fc

def braced_tower_2d():
    # topology
    T=cem_mini.create_topology(6)
    
    cem_mini.set_trail_paths(T, [[2,1,0],[5,4,3]],[[-1,-1],[-1,-1]])
    cem_mini.set_deviation_edges(T,[[1,4],[2,5],[1,5],[1,3],[2,4]],[-1,-1, 1, 1, 1])

    cem_mini.set_original_node_positions(T,{2:[0,2,0], 5:[1,2,0]})
    cem_mini.set_node_loads(T,{2:[0,-1,0],5:[0,-1,0]})
    
    # solve the CEM
    F, Fc= cem_mini.CEM(T,1e-7)
    
    # visualize form diagram
    _make_plots(F)
    
    return T, F, Fc

def bridge_2d():
    # topology
    T=cem_mini.create_topology(8)
    
    cem_mini.set_trail_paths(T, [[2,3,0,1],[6,7,4,5]], [[12,12,12],[12,12,12]])
    cem_mini.set_deviation_edges(T,[[7,3],[3,4],[0,4]],[5,-5,5])

    cem_mini.set_original_node_positions(T,{2:[-6.5,0,0], 6:[6.5,0,0]})
    cem_mini.set_node_loads(T,{2:[0,-1,0],6:[0,-1,0]})

    # solve the CEM
    F, Fc= cem_mini.CEM(T,1e-7)
    
    # visualize form diagram
    _make_plots(F)
    
    return T, F, Fc

def tree_2d():
    '''
    the optimization part was not implemented
    '''
    width = 4
    height = width / 2
    
    # topology
    T=cem_mini.create_topology(6)
    
    cem_mini.set_trail_paths(T, [[2,3],[0,4],[1,5]], [[-height/2],[-height/2],[-height/2]])
    cem_mini.set_deviation_edges(T,[[0,1],[0,2],[1,2]],[2,-2,-1.41421])

    cem_mini.set_original_node_positions(T,{0:[-width / 2, height, 0.0],1:[width / 2, height, 0.0], 2:[0.0, height / 2, 0.0]})
    cem_mini.set_node_loads(T,{0:[0.0, -1.0, 0.0],1:[0.0, -1.0, 0.0]})
    
    # solve the CEM
    F, Fc= cem_mini.CEM(T,1e-7)
    
    # visualize form diagram
    _make_plots(F)
    
    return T, F, Fc

def random_tower(nx=4,ny=4,nz=10, floor_heights=[4,8]):
    '''
    the optimization part was not implemented
    '''
    T = cem_topology_maker.cubic_grid_tower_topology(nx,ny,nz,
                                                     floorheight=cem_topology_maker.discrete_sampler(floor_heights))
    F, Fc= cem_mini.CEM(T,1e-7)
    
    # visualize form diagram
    plt.figure(figsize=(16,8))
    ax=plt.axes([0,0,0.5,1], projection='3d')
    cem_mini.plot_cem_form(ax,F['coords'],F['edges'],F['edge_forces'],view='3D-45',thickness_base=0.5,thickness_ratio=0.02)
    plt.axis('on')
    
    ax=plt.axes([0.5,0,0.5,1], projection='3d')
    cem_mini.plot_cem_form(ax,F['coords'],F['edges'],F['edge_forces'],view='2D-XZ',thickness_base=0.5,thickness_ratio=0.02)
    plt.axis('on')
    
    plt.show()
    
    return T, F, Fc
    