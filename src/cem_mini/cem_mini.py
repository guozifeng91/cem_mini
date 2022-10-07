'''
the core module of cem_mini

--- intro ---

cem_mini is a python implementation of the Combinatorial Equilibrium Modelling (CEM) method
developed by Ohlbrock, P. O. ETH Zurich, following the chapter 7 of Dr. Ohlbrock's doctoral
thesis that can be downloaded from https://www.research-collection.ethz.ch/handle/20.500.11850/478732

--- concept ---

cem_mini aims to minimize the dependent libraries and attemps to stay with function oriented
programming style (not yet fully, because the modifications are made on the original T rather than the copy of T)

--- implementation details ---

cem_mini implements the topological definition T as a pure graph, where nodes are represented as indices (integers)
and edges are indice pairs (list of integers). there is no object-oriented implementation such as "node" or "edge" classes.

the metric properties are attached to the topological definition as either dictionaries or nd-arrays.
in case of dictionaries, the node indices are used as dictionary keys

this allows the cem definition to be easily exported as json format for data exchange across different libraries.

the solver function relies on numpy's matrix operations as much as possible to increase efficiency and reduce code complexity

--- notes ---

cem_mini is validated using 
 - 1. the compas_cem by Rafael Pastrana (https://github.com/arpastrana/compas_cem), and
 - 2. the cem implementation in grasshopper by Ohlbrock, P. O. and D'Acunto, P. (https://github.com/OleOhlbrock/CEM)

cem_mini is implemented by: Z. Guo

--- variable name scope ---

F: form
Fc: force
T: topology

c: combinatorial states (1 for tension, -1 for compression)
w: topological distance, which is equal to the number of trail edges in-between vertex v_i and its corresponding support vertex.
u: unit directional vector between nodes
rd: resultant deviation vector of nodes

lbd: (i.e., lambda) trail edge length + combinatorial states
mu: absolute force magnitude of the deviation edges + combinatorial states
    
X: design information, which contains:
    p: spatial position of nodes
    q: external force
'''

import numpy as np

def to_edge_indice(i,j,n):
    '''
    compute the edge indice from node indices (i,j) and totoal num of nodes n
    '''
    return i*n+j

def from_edge_indice(ei,n):
    '''
    compute nodes indices (i,j) from edge indice
    '''
    return ei//n,ei%n

def create_topology(num_nodes=8):
    '''
    create a topology diagram with num_nodes nodes
    
    ----- parameters -----
        num_nodes: a positive integer, number of nodes
        
    ----- returns -----
        the topology diagram T (which is a dictionary)
    '''
    n=num_nodes
    
    # number of edges in a complete graph that has n nodes
    # the reason that we use complete graph is that the edges are not yet specified and can be any topology
    # and we need a systematic way to represent these edges
    ne=n**2
    # bug fix: set initial value to 0.0 to prevent initialized as integers
    return {'n':n, 'lbd':[0.0]*ne,'mu':[0.0]*ne,'X':{}}

def set_trail_paths(T, trail_paths, lbd=None, override=False):
    '''
    set trail paths to topology T
    
    ----- parameters -----
        T: the target topology
        trail_paths: trail paths
        lbd: optional, edge length of the trail path edges
        override: True to override exist info (i.e., remove all existing trail path specifications).
        
    ----- returns -----
        T itself
    
    ----- notes -----
        - trail paths are paths along which loads are transferred from original nodes to support nodes
        - trail_paths are directional, i.e., original -> intermidiate -> ... -> support
        - trail_paths should be given as [(i, j, k, ...), (i, j, k, ...), ...] where i j k are indices of nodes
        - trail_paths should not result in any loop
        - trail_paths should contain all nodes of T
        - trail_paths should not intersect, i.e., paths do not share nodes
        
        - lbd are combinatorial states + edge length of the edges of each trail path
        - lbd should be given as list of lists, e.g., [[3,2,-5,...], ...]
        - lbd[i][j] corresponds to the combinatorial states plus the edge length of trail edge trail_paths[i][j:j+2]
        - therefore, len(trail_path[i]) = len(c_path[i])+1
        - lbd indicates tension edges with positive values and compression edges with negative values

    '''
    n=T['n']
    trail_edges=[]
    
    for path in trail_paths:
        for pos in range(len(path)-1):
            trail_edges.append(path[pos:pos+2])
    
    T['trail_edges']=(trail_edges if override or 'trail_edges' not in T.keys() else T['trail_edges']+trail_edges)
    T['trail_paths']=(trail_paths if override or 'trail_paths' not in T.keys() else T['trail_paths']+trail_paths)
    
    T['original_nodes']=[p[0] for p in T['trail_paths']]
    T['support_nodes']=[p[-1] for p in T['trail_paths']]
    
    if lbd is not None:
        if len(lbd) != len(trail_paths):
            print('The lbd should have the same length with trail_path!')
        else:
            if override:
                T['lbd']=[0.0]*(n**2) # fix: change 0 to 0.0
                
            for tp, lbd_ in zip(trail_paths, lbd): # for each trail path
                for edge_i in range(len(lbd_)): # for each trail edge
                    i, j=tp[edge_i:edge_i+2]
                    T['lbd'][to_edge_indice(i,j,n)]=lbd_[edge_i]
                    
    return T

def set_deviation_edges(T, deviation_edges, mu=None, override=False):
    '''
    set direct and indirect deviation edges for T
        
    ----- parameters -----
        T: the target topology
        deviation_edges: should be given as [(i, j), (i, j), ...] where i and j are indices
        mu: optional, force magnitude (kN) of each deviation edge, should be given as list of floats, e.g., [1.0,2.2,-1.5,...]
        override: True to override the exist info.
        
    ----- returns -----
        T itself
        
    ----- note -----
        deviation edges are NOT directional, therefore (1,2) and (2,1) are equivalent
        mu represent combinatorial states with positive and negative values, respectively
        
    '''
    n=T['n']
    
    deviation_edges=[sorted(e) for e in deviation_edges]
    
    if 'deviation_edges' not in T.keys() or override:
        T['deviation_edges'] = sorted(deviation_edges)
    else:
        T['deviation_edges'] = sorted(T['deviation_edges'] + deviation_edges)

    if mu is not None:
        if len(mu) != len(deviation_edges):
            print('The length of mu should be the length of deviation_edges!')
        else:
            if override:
                T['mu']=[0.0]*(n**2) # fix: change 0 to 0.0
                
            for e, mu_ in zip(deviation_edges, mu):
                i,j=e
                T['mu'][to_edge_indice(i,j,n)]=mu_
                T['mu'][to_edge_indice(j,i,n)]=mu_

    return T

def set_deviation_edge_force_magnitude(T, mu):
    '''
    this function is useful if multiple mu are to be generated automatically
    
    specify the tension-compression states as well as the force magnitude for
    all deviation edges for T, if such information was not specified when setting edges
    
    ----- parameters -----
        T: the target topology
        mu: list of floats, the force magnitude + combinatorial states of all
        deviation edges (positive number for tension, negative for compression)
        
    ----- returns -----
        T itself
    '''
    
    for e, mu_ in zip(T['deviation_edges'], mu):
        i,j=e
        T['mu'][to_edge_indice(i,j,n)]=mu_
        T['mu'][to_edge_indice(j,i,n)]=mu_
        
    return T

def set_trail_edge_length(T, lbd):
    '''
    this function is useful if multiple mu are to be generated automatically
    
    specify the tension-compression states as well as the edge length for all
    trail edges for T, if such information was not specified when setting edges
    
    ----- parameters -----
        T: the target topology
        lbd: the edge length + combinatorial states of all deviation edges
        (positive number for tension, negative for compression)
        
    ----- returns -----
        T itself
    '''
    print('Warning, not implemented yet')
    return T # not implemented yet

def _compute_topological_distances(T):
    '''
    compute the topological distances of T
    support_nodes has topological distances of 0
    '''
    n=T['n']
    trail_paths=T['trail_paths']
    w=np.zeros((n,),np.int32)
    
    for path in trail_paths:
        for i in range(len(path)):
            node_id=path[-i-1]
            w[node_id]=i

    return w

def set_original_node_positions(T,p):
    '''
    set original node position for T
    
    ----- parameters -----
        T: the topology
        p: either dictionary {id:[x,y,z],...} or list of coords [(x,y,z),(x,y,z),...].
        Note that if p is given as a list of coords, it should has the same length as T['original_nodes']
    '''
    T['X']['p']=p if type(p) is dict else {i:c for i, c in zip(T['original_nodes'],p)}
    
    return T

def set_constrained_planes(T, cplanes):
    '''
    optional, set contrained planes for the nodes of T
    
    ----- parameters -----
        cplanes: a dictionary of {id:[x,y,z,nx,ny,nz]}, representing the index
        of nodes and the associated constrained planes. Each constrained plane
        should be specified as a 6-dimensional vector [x, y, z, nx, ny, nz] that
        representing the origin and the normal vector of the plane.
        
    ----- returns -----
        the topology diagram T (which is a dictionary)
    '''
    
    T['X']['cp']=cplanes
    return T

def set_node_loads(T,q):
    '''
    set node loads for T
    
    ----- parameters -----
        T: the topology
        q: dictionary {id:[x,y,z],...}
    '''
    T['X']['q']=q

def CEM(T, epsilon=1e-5, load_func=None):
    '''
    the CEM algorithm, implemented based on the chapter 7 of
    Combinatorial Equilibrium Modelling, by Ohlbrock, P. O. ETH Zurich 
    
    ----- parameters -----
        T: topology diagram
        epsilon: threshold value for iterative process, default 1e-5
        load_func: a callable f(i, p) which returns form-dependent loads,
        where the arguments i and p are the index and position of vertex
    
    ----- returns -----
        form and force diagrams, both are dictionaries
        
    ----- notes -----
        it is necessary to specify external loads in X even when load_func is given,
        this is because the spatial position of most vertices are unknown in the initial iteration
    '''
    n=T['n']
    indices=np.arange(n)
        
    w=_compute_topological_distances(T) # topological distances w of each node
    K=w.max()-w # sequence k of each node
    K_max=K.max()
    
    iterative=any([w[ns]!=w[ne] for ns,ne in T['deviation_edges']]) # whether the equilibrium states require iterative solving process
    itr=0 # iteration counter

    trail_paths=T['trail_paths'] # list of lists, representing the trail path, from start (original) to end (support)
    trail_path_down={p[i]:p[i+1] for p in trail_paths for i in range(len(p)-1)} # dict, k nodes to k+1 nodes   
    trail_path_up={p[-i]:p[-i-1] for p in trail_paths for i in range(1, len(p))} # dict, k nodes to k-1 nodes   
    
    # to float64 ?
    
    t_in=np.zeros((n,3),np.float32) # n x 3 matrix, in-bound trail force vector of each node
    t_out=np.zeros((n,3),np.float32) # n x 3 matrix out-going trail force vector of each node
    rd=np.zeros((n,3),np.float32) # n x 3 matrix, resultant deviation vector of each node (i.e., the summation of direct and indirect vectors)
    
    cp_o=np.zeros((n,3), np.float32) # n x 3 matrix, origin of the constrained plane of each node
    cp_n=np.zeros((n,3), np.float32) # n x 3 matrix, normal vector of the constrained plane of each node
    
    u=np.zeros((n,n,3), np.float32) # n x n x 3 matrix, representing the unit vector from node i to node j
    p=np.zeros((n,3),np.float32) # n x 3 matrix, position of nodes
    q=np.zeros((n,3),np.float32) # n x 3 matrix, node loads
    
    for i in T['X']['p'].keys():
        p[i]=T['X']['p'][i]
        
    for i in T['X']['q'].keys():
        q[i]=T['X']['q'][i]
    
    if 'cp' in T['X'].keys():
        for i in T['X']['cp'].keys():
            cp_o[i]=T['X']['cp'][i][:3] # origin of constrained planes
            cp_n[i]=T['X']['cp'][i][3:] # normal vector of constrained planes
    
    mu=np.asarray(T['mu']).reshape((n,n)) # n x n matrix, absolute force magnitudes
    lbd=np.asarray(T['lbd']).reshape((n,n)) # n x n matrix, trail edge length
        
    # compute C after the form-finding because lbd may be overrided
    # C=np.sign(mu)+np.sign(lbd) # symmetric adjacency matrix C, where 1 represents tension, -1 represents compression

    while iterative or itr==0 or load_func is not None:
        p_prev=np.copy(p) # pos of t-1 iteration
        
        # get form-dependent external loads after the inital iteration
        if load_func is not None and itr>0:
            q=np.asarray([load_func(i,p[i]) for i in range(n)])
            
        # run at least one iteration
        for k in range(K_max+1):
            # indicex for sequence k vertices
            indices_k=indices[K==k]
            # indicex for sequence k-1 vertices (where the forces come)
            indices_in=None if k==0 else np.asarray([trail_path_up[i] for i in indices_k])
            # indicex for sequence k+1 vertices (where the forces go)
            indices_out=None if k==K_max else np.asarray([trail_path_down[i] for i in indices_k])
            
            # update the direction matrix of nodes in sequence k in related with other nodes
            for i in indices_k: # nodes in current iteration
                for j in indices: # all other nodes
                    if i!=j:
                        v=p[j]-p[i]
                        v_norm=np.linalg.norm(v)
                        u[i,j] = 0 if v_norm==0 else (v/v_norm)
    
            # step 1. compute resultant direct+indirect deviation vector (formula 7.9, formula 7.13)
            # note that C[indices_k] is integrated in mu[indices_k]
            
            rd[indices_k]=np.sum((mu[indices_k])[...,None] * u[indices_k],axis=1)
            
            # step 2. compute outgoing trail force vector t_i_out (formula 7.8, formula 7.12)
            if k==0: # original nodes
                t_in[indices_out] = rd[indices_k] + q[indices_k] # (formula 7.8)
                t_out[indices_k] = -t_in[indices_out] # (formula 7.8)
            elif k==K_max: # support (final) nodes
                t_out[indices_k] = -(t_in[indices_k] + rd[indices_k] + q[indices_k]) # (formula 7.8)
            else: # k>0, trail-path nodes
                t_in[indices_out] = t_in[indices_k] + rd[indices_k] + q[indices_k] # (formula 7.8)
                t_out[indices_k] = -t_in[indices_out] # (formula 7.8)
            
            # step 3. compute position vector p_i_out
            # if constrained planes are not specified (formula 7.11)
            # or (formula 7.15) if otherwise
            
            # note that C[indices_k,indices_out] is integrated in lbd[indices_k,indices_out]
            # and if the constrained planes are specified, the lbd[indices_k,indices_out] will be overrided (figure 7.7c)

            if k<K_max:
                # u_out is the normalized t_out (formula 7.11)
                u_out = (1.0 / np.linalg.norm(t_out[indices_k], axis=-1))[...,None] * t_out[indices_k]
                
                # a mask vector that indicates whether lbd (i.e., trail edge length) will be overrided
                # mask == 1 if no constrained plane are specified (i.e., cp_n = [0,0,0]),
                # and mask == 0 if otherwise
                mask = 1 - np.sign(np.linalg.norm(cp_n[indices_out], axis=-1))

                # the coefficient r in formula 7.15,
                # r == 0 if no constrained plane are specified (mask == 1, cp_n == [0,0,0])
                # and r != 0 if otherwise (mask == 0, cp_n != [0,0,0])
                r = np.einsum("ij,ij->i",cp_o[indices_out] - p[indices_k], cp_n[indices_out], dtype=np.float64) / (mask + np.einsum("ij,ij->i",u_out, cp_n[indices_out], dtype=np.float64))
                
                # override trail edge length specification if constrained planes are given
                # lbd = lbd * mask + r
                lbd[indices_k, indices_out] = lbd[indices_k, indices_out] * mask + r

                # formula 7.11, with updated lbd
                p[indices_out]=p[indices_k] + lbd[indices_k, indices_out][...,None] * u_out
                
                # previous implementation, formula 7.11 in one line
                # deprecated as we want to have formula 7.15 as well
                # p[indices_out]=p[indices_k] + (lbd[indices_k, indices_out] / np.linalg.norm(t_out[indices_k], axis=-1))[...,None] * t_out[indices_k]
                
        # next iteration
        itr+=1
        error=np.sum((p-p_prev)**2)
        
        if itr%10==1:
            print('iteration',itr,'error', error,end='\r')
        
        # quit itertion if the result converges
        if error <= epsilon:
            print('iteration',itr,'error', error,'finished.')
            break
    
    C=np.sign(mu)+np.sign(lbd) # symmetric adjacency matrix C, where 1 represents tension, -1 represents compression
    T['lbd_cplane_override'] = lbd.flatten() # new lbd computed from the constrained plane specification, in contrast to the original lbd
        
    # deviation force vector from node i to j, note that C is integrated in mu
    force_matrix=mu[...,None]*u
    # trail force vector from node i to j
    i,j = np.asarray(T['trail_edges']).T
    force_matrix[i,j]=t_out[i]
    force_matrix[j,i]=t_in[j]
    
    # force magnitude matrix of edge i-j
    force_mag_matrix=np.sqrt(np.sum(force_matrix**2,axis=-1)) * C

    edges=list(T['trail_edges'])+list(T['deviation_edges'])
    edge_forces=[force_mag_matrix[i,j] for i,j in edges] # force magnitude of edge i-j
    F={'n':n,
       'coords':p,
       'edges':edges,
       'edge_forces':edge_forces,
       'loads':q} # form diagram
    
    Fc={'n':n,
        'edges':edges,
        'node2node_force_vec_matrix':force_matrix,
        'node2node_force_mag_matrix':force_mag_matrix,
        'trail_forces_in':t_in,
        'trail_forces_out':t_out,
        'loads':q} # force diagram

    return F, Fc

def _json_serializable_arr(x):
    '''
    convert a nd-array to list of lists
    '''
    if type(x) is np.ndarray: # numpy array
        return x.tolist()
    elif hasattr(x,'__len__') and (type(x) is not str): # list-liked object, such as the mixture of numpy + list + tuple
        return [_json_serializable_arr(i) for i in x] 
    elif np.issubdtype(type(x) , np.integer): # numpy integer
        return int(x)
    elif np.issubdtype(type(x) , np.inexact): # numpy float
        return float(x)
    else: # other data
        return x

def _json_serializable_dict(x):
    return {str(k):(_json_serializable_dict(x[k]) if type(x[k]) is dict else _json_serializable_arr(x[k])) for k in x.keys()}
    
import json
def export_cem(fname, T):
    '''
    export a CEM definition (T, F, or Fc) to a json file
    '''
    T=_json_serializable_dict(T)
    with open(fname, 'w') as f:
        json.dump(T, f)
        
def import_cem(fname):
    '''
    import a CEM definition from a json file
    '''
    with open(fname, 'r') as f:
        T=json.load(f)
    return T