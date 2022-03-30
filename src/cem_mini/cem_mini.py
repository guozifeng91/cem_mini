'''
cem_mini is a python replica of the CEM method developed by Ohlbrock, P. O. ETH Zurich
cem_mini aims to minimize the dependent libraries and attemps to stay with function oriented programming style
cem_mini is validated by comparing with the compas_cem by Rafael Pastrana (https://github.com/arpastrana/compas_cem)  

the equations mentioned below are from the doctoral thesis Combinatorial Equilibrium Modelling by Ohlbrock, P. O., ETH Zurich
the doctoral thesis can be downloaded in https://www.research-collection.ethz.ch/handle/20.500.11850/478732

cem_mini is implemented by: Z. Guo
---

variable name space:

F: form diagram
Fc: force diagram
T: topology diagram

c: combinatorial states (1 for tension, -1 for compression)
w: topological distance, which is equal to the number of trail edges in-between vertex v_i and its corresponding support vertex.
u: unit directional vector between nodes
rd: resultant deviation vector of nodes

lbd: trail edge length + combinatorial states
mu: absolute force magnitude of the deviation edges + combinatorial states
    
X: design information, which contains:
    p: spatial position of nodes
    q: external force
'''

import numpy as np

def to_edge_indice(i,j,n):
    return i*n+j

def from_edge_indice(ei,n):
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
    ne=n**2
    return {'n':n, 'lbd':[0]*ne,'mu':[0]*ne,'X':{}}

def set_trail_paths(T, trail_paths, lbd=None, override=False):
    '''
    set trail paths to topology T
    
    ----- parameters -----
        T: the target topology
        trail_paths: trail paths
        lbd: optional, edge length of the trail path edges
        override: True to override exist info.
        
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
        - lbd[i][j] corresponds to the combinatorial states as well as the edge length of trail_paths[i][j:j+2]
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
                T['lbd']=[0]*(n**2)
                
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
                T['mu']=[0]*(n**2)
                
            for e, mu_ in zip(deviation_edges, mu):
                i,j=e
                T['mu'][to_edge_indice(i,j,n)]=mu_
                T['mu'][to_edge_indice(j,i,n)]=mu_

    return T

def set_deviation_edge_force_magnitude(T, mu):
    '''
    this function is useful if multiple mu are to be generated automatically
    
    specify the tension-compression states as well as the force magnitude for all deviation edges for T, if such information was not specified when setting edges
    
    ----- parameters -----
        T: the target topology
        mu: list of floats, the force magnitude + combinatorial states of all deviation edges (positive number for tension, negative for compression)
        
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
    
    specify the tension-compression states as well as the edge length for all trail edges for T, if such information was not specified when setting edges
    
    ----- parameters -----
        T: the target topology
        lbd: the edge length + combinatorial states of all deviation edges (positive number for tension, negative for compression)
        
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
        p: either dictionary {id:[x,y,z],...} or list of coords [(x,y,z),(x,y,z),...]
    '''
    T['X']['p']=p if type(p) is dict else {i:c for i, c in zip(T['original_nodes'],p)}
    
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
    the CEM algorithm, implemented based on the chapter 7 of Combinatorial Equilibrium Modelling, by Ohlbrock, P. O. ETH Zurich 
    
    ----- parameters -----
        T: topology diagram
        epsilon: threshold value for iterative process, default 1e-5
        load_func: a callable f(i, p) which returns form-dependent loads, where the arguments i and p are the index and position of vertex
    
    ----- returns -----
        form and force diagrams, both are dictionaries
        
    ----- notes -----
        it is necessary to specify external loads in X even when load_func is given,
        because the spatial position of most vertices are unknown in the initial iteration
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
    
    t_in=np.zeros((n,3),np.float32) # n x 3 matrix, in-bound trail force vector of each node
    t_out=np.zeros((n,3),np.float32) # n x 3 matrix out-going trail force vector of each node
    rd=np.zeros((n,3),np.float32) # n x 3 matrix, resultant deviation vector of each node (i.e., the summation of direct and indirect vectors)
    u=np.zeros((n,n,3), np.float32) # n x n x 3 matrix, representing the unit vector from node i to node j
    p=np.zeros((n,3),np.float32) # n x 3 matrix, position of nodes
    q=np.zeros((n,3),np.float32) # n x 3 matrix, node loads
    
    for i in T['X']['p'].keys():
        p[i]=T['X']['p'][i]
        
    for i in T['X']['q'].keys():
        q[i]=T['X']['q'][i]
    
    mu=np.asarray(T['mu']).reshape((n,n)) # n x n matrix, absolute force magnitudes
    lbd=np.asarray(T['lbd']).reshape((n,n)) # n x n matrix, trail edge length

    C=np.sign(mu)+np.sign(lbd) # symmetric adjacency matrix C, where 1 represents tension, -1 represents compression

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
            if k==0:
                t_in[indices_out] = rd[indices_k] + q[indices_k] # (formula 7.8)
                t_out[indices_k] = -t_in[indices_out] # (formula 7.8)
            elif k==K_max:
                t_out[indices_k] = -(t_in[indices_k] + rd[indices_k] + q[indices_k]) # (formula 7.8)
            else: # k>0
                t_in[indices_out] = t_in[indices_k] + rd[indices_k] + q[indices_k] # (formula 7.8)
                t_out[indices_k] = -t_in[indices_out] # (formula 7.8)
            
            # step 3. compute position vector p_i_out (formula 7.11)
            # note that C[indices_k,indices_out] is integrated in lbd[indices_k,indices_out]
            if k<K_max:
                p[indices_out]=p[indices_k] + (lbd[indices_k, indices_out] / np.linalg.norm(t_out[indices_k], axis=-1))[...,None] * t_out[indices_k]
                
        # next iteration
        itr+=1
        error=np.sum((p-p_prev)**2)
        
        if itr%10==1:
            print('iteration',itr,'error', error,end='\r')
        
        # quit itertion if the result converges
        if error <= epsilon:
            print('iteration',itr,'error', error,'finished.')
            break
    
    # deviation force vector from node i to j, note that C is integrated in mu
    forces_vec=mu[...,None]*u
    # trail force vector from node i to j
    i,j = np.asarray(T['trail_edges']).T
    forces_vec[i,j]=t_out[i]
    forces_vec[j,i]=t_in[j]
    
    # force magnitude matrix of edge i-j
    forces=np.sqrt(np.sum(forces_vec**2,axis=-1)) * C
    Fc={'n':n,'forces_vec':forces_vec, 'trail_forces_out':t_out,'forces':forces,'loads':q} # force diagram

    edges=list(T['trail_edges'])+list(T['deviation_edges'])
    edge_forces=[forces[i,j] for i,j in edges] # force magnitude of edge i-j
    F={'n':n,'coords':p,'edges':edges, 'edge_forces':edge_forces, 'loads':q} # form diagram
    
    return F, Fc

def CEM_old(T, X, epsilon=1e-5, load_func=None):
    '''
    the old function that not in an optimal state
    
    the CEM algorithm, implemented using Ohlbrock, P. O. Combinatorial Equilibrium Modelling, ETH Zurich 
    
    ----- parameters -----
        T: topology diagram
        X: design parameters (i.e., boundary conditions)
        epsilon: threshold value for iterative process, default 1e-5
        load_func: a callable f(i, p) which returns form-dependent loads, where the arguments i and p are the index and position of vertex
    
    ----- returns -----
        form and force diagrams, both are dictionaries
        
    ----- notes -----
        it is necessary to specify external loads in X even when load_func is given,
        because the spatial position of most vertices are unknown in the initial iteration
    '''
    raise Exception('This function has been deprecated')
    
    n=T['n']
    indices=np.arange(n)
    
    topology_matrix=_compute_topological_matrix(T)
    C=np.sign(topology_matrix) # symmetric adjacency matrix C, where 1 represents tension, -1 represents compression
    
    w=_compute_topological_distances(T) # topological distances w of each node
    K=w.max()-w # sequence k of each node
    
    iterative=any([w[ns]!=w[ne] for ns,ne in T['deviation_edges']]) # whether the equilibrium states require iterative solving process
    itr=0 # iteration counter

    trail_paths=T['trail_paths'] # list of lists, representing the trail path, from start (original) to end (support)
    trail_path_down={p[i]:p[i+1] for p in trail_paths for i in range(len(p)-1)} # dict, k nodes to k+1 nodes   
    trail_path_up={p[-i]:p[-i-1] for p in trail_paths for i in range(1, len(p))} # dict, k nodes to k-1 nodes   
    
    original_nodes=T['original_nodes']
    support_nodes=T['support_nodes']

    t_in=np.zeros((n,3),np.float32) # in-bound trail force vector
    t_out=np.zeros((n,3),np.float32) # out-going trail force vector
#     t_ij=np.zeros((n,n,3),np.float32) # n x n x 3 matrix, representing out going trail force vector from node i to node j

    rd_i=np.zeros((n,3),np.float32) # n x 3 matrix, representing resultant deviation vector of node i (i.e., the summation of direct and indirect vectors)
    u_ij=np.zeros((n,n,3), np.float32) # n x n x 3 matrix, representing the unit vector from node i to node j
    
    p=X['p'] # position of nodes
    q=X['q'] # external loads of nodes
    mu=X['mu'] # absolute force magnitudes
    lbd=X['lbd'] # trail edge length

    while iterative or itr==0 or load_func is not None:
        p_prev=np.copy(p) # pos of t-1 iteration
        
        # get form-dependent external loads after the inital iteration
        if load_func is not None and itr>0:
            q=np.asarray([load_func(i,p[i]) for i in range(n)])
            
        # run at least one iteration
        for k in range(K.max()+1):
            # indicex for sequence k vertices
            indices_k=indices[K==k]
            # print(indices_k)
            # indicex for sequence k-1 vertices (where the forces come)
            indices_in=None if k==0 else np.asarray([trail_path_up[i] for i in indices_k])
            # indicex for sequence k+1 vertices (where the forces go)
            indices_out=None if k==K.max() else np.asarray([trail_path_down[i] for i in indices_k])
            
            # print('in', indices_in, 'k',indices_k,'out',indices_out)

            # update the direction matrix of nodes in sequence k in related with other nodes
            for i in indices_k: # nodes in current iteration
                for j in indices: # all other nodes
                    if i!=j:
                        v=p[j]-p[i]
                        v_norm=np.linalg.norm(v)
                        u_ij[i,j]= 0 if v_norm==0 else (v/v_norm)
    
            # step 1. compute resultant direct+indirect deviation vector (formula 7.9, formula 7.13)
            # rd_i = sum(c_ij * mu_ij * u_ij)
            # u_ij for indirect deviation edges are set to 0 for iter=0 (formula 7.12)
            # as C and mu matrices are filled with 0s for all non-connected nodes
            # these edges contribute nothing to the summation and need not to be excluded from the computations
            
            # this for-loop is replaced by the equivalent line below
            # for i in indices_k:
            #     rd_i[i]=np.sum((C[i]*mu[i])[...,None]*u_ij[i],axis=0)

            rd_i[indices_k]=np.sum((C[indices_k]*mu[indices_k])[...,None]*u_ij[indices_k],axis=1)
            
            # step 2. compute outgoing trail force vector t_i_out (formula 7.8, formula 7.12)
            if indices_in is None and indices_out is not None: # k=0
                for i, i_out in zip(indices_k, indices_out):
                    t_out[i]=-rd_i[i]-q[i] # (formula 7.8)
                    t_in[i_out]=-t_out[i] # (formula 7.8)
                    
                    # t_ij[i,i_out]=-rd_i[i]-q[i] # here is an equivalent impl. using matrix t_ij
            elif indices_out is None: # k=K
                for i in indices_k:
                    t_out[i]=-t_in[i]-rd_i[i]-q[i] # (formula 7.8)
            else: # k>0
                for i, i_out in zip(indices_k, indices_out):
                    t_out[i]=-t_in[i]-rd_i[i]-q[i] # (formula 7.8)
                    t_in[i_out]=-t_out[i] # (formula 7.8)
                    
                # below are equivalent implementation, using matrix (from i to j)
                # the downside of using matrix is 1. space complexity and 2. cannot represent t_out for vertices of k=K
                
#                 for i_in, i, i_out in zip(indices_in, indices_k, indices_out):
#                     # t_out(i) = -t_in(i)-rd(i)-q(i), (formula 7.8)
#                     # note that t_ij[a,b] represent vectors from a to b.
#                     # i.e., t_ij[i_in,i] = t_out(i_in) = -t_in(i)
#                     t_ij[i,i_out]=t_ij[i_in,i]-rd_i[i]-q[i]
            
            if indices_out is not None: # k<K
                # step 3. compute position vector p_i_out (formula 7.11)
                for i, i_out in zip(indices_k, indices_out):
                    p[i_out]=p[i]+C[i,i_out]*lbd[i,i_out]*t_out[i] / np.linalg.norm(t_out[i])
#                     p[i_out]=p[i]+C[i,i_out]*lbd[i,i_out]*t_ij[i,i_out] / np.linalg.norm(t_ij[i,i_out])
            
        
        # next iteration
        itr=itr+1
        
        error=np.sum((p-p_prev)**2)
        
        if itr%10==1:
            print('iteration',itr,'error', error,end='\r')
        
        # quit itertion if the result converges
        if error <= epsilon:
            print('iteration',itr,'error', error,'finished.')
            break
    
    # deviation force vector from node i to j
    forces_ij_vec=(C*mu)[...,None]*u_ij
    # trail force vector from node i to j
    for i,j in T['trail_edges']:
        forces_ij_vec[i,j]=t_out[i]
        forces_ij_vec[j,i]=t_in[j]
    
    # force magnitude matrix of edge i-j
    forces_ij=np.sqrt(np.sum(forces_ij_vec**2,axis=-1)) * C
    Fc={'n':n,'forces_ij_vec':forces_ij_vec, 'trail_forces_out':t_out,'forces_ij':forces_ij} # force diagram

    edges=list(T['trail_edges'])+list(T['deviation_edges'])
    edge_forces=[forces_ij[i,j] for i,j in edges] # force magnitude of edge i-j
    F={'n':n,'coords':p,'edges':edges, 'edge_forces':edge_forces} # form diagram
    
    return F, Fc