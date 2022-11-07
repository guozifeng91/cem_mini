'''
the plotting module of cem_mini

'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

set_view={'2D-XY':(90,-90),
        '2D-XZ':(0, 90),
        '2D-YZ':(0, 0),
        '3D-45':(45, 45),
        '3D-30':(30, 60)}

blue_color=(5/256,120/256,190/256)
red_color=(200/256,20/256,20/256)

# arrow code from https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)#renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def plot_cem_form(ax, coords, edges, forces, loads=None, view='2D-XY', load_len_scale=1, thickness_base=1, thickness_ratio=0.01, ignore_zeros=True, vrange=None):
    '''
    coords: np array of shane (M,3)
    edges: array or np array of shape (N, 2), representing the node indices
    forces: np array of shape (N)
    loads: np array of shane (M,3), external loads at each coord (optional)
    '''
    # make sure the inputs are npy array
    coords=np.asarray(coords)
    edges=np.asarray(edges)
    forces=np.asarray(forces)

    if loads is not None:
        loads=np.asarray(loads)

    ax.set_aspect('auto')
    ax.axis('off')

    # Set view
    ax.set_proj_type('ortho')
    #ax.set_proj_type('persp')

    ax.view_init(elev=set_view[view][0], azim=set_view[view][1])

    if vrange is None:
        # obtain the plotting range automatically
        vrange=coords.max(axis=0) - coords.min(axis=0)
        vrange=vrange.max()/2

        # otherwise stay with user-specified value
        # which would be useful when multiple plots need to have the same scale

    if loads is not None and loads.shape==coords.shape:
        for p2, v in zip(coords, loads):
            p1=p2-v*load_len_scale
            f=np.linalg.norm(v)
            line_width = thickness_base +f*thickness_ratio
            line=np.asarray([p1,p2])

            a = Arrow3D(line[...,0], line[...,1], line[...,2], mutation_scale=20,
                        lw=line_width, arrowstyle="-|>", color="g")

            ax.add_artist(a)
#             ax.plot(line[...,0], line[...,1], line[...,2], color = 'green', linewidth = line_width, antialiased=True, alpha = 1.0)

    for i in range(len(edges)):
        edge=edges[i]
#             edge = [ int(branchNode_arr[2*edge_i]), int(branchNode_arr[2*edge_i+1]) ]
        line_color = blue_color if forces[i] < 0 else red_color

        if ignore_zeros and np.abs(forces[i])<=1e-8:
            continue
        line_width = thickness_base + np.abs(forces[i]*thickness_ratio)
        ax.plot(coords[edge,0], coords[edge,1], coords[edge,2], color = line_color, linewidth = line_width, antialiased=True, alpha = 1.0)

    # make xyz scale equal
    vmean=coords.mean(axis=0)
    ax.set_xlim(vmean[0]-vrange,vmean[0]+vrange)
    ax.set_ylim(vmean[1]-vrange,vmean[1]+vrange)
    ax.set_zlim(vmean[2]-vrange,vmean[2]+vrange)
    return ax
