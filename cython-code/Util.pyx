import numpy as np 
cimport numpy as cnp
from igraph import Graph
from Variable import Variable
cimport cython 
import pandas as pd
from scipy import stats

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int _getDomainSize(list variables):
    cdef int d = 1
    for var in variables:
        d *= var.d
    return d 

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def getDomainSize(list variables):
    return _getDomainSize(variables)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef _getDirectedST(cnp.ndarray[double, ndim=2] adj_mat):
    adj_mat[adj_mat==0] = 1e-12
    g = Graph.Weighted_Adjacency(adj_mat)
    cdef cnp.ndarray[double, ndim=1] wts = adj_mat.flatten()
    tree = g.spanning_tree(weights=wts)
    tree.to_undirected()
    children, parents = tree.dfs(vid=0)
    return children, parents

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def getDirectedST(cnp.ndarray[double, ndim=2] adj_mat):
    return _getDirectedST(adj_mat)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef void _setAddr(list variables, int ind):
    cdef int i 
    for i in range(len(variables)-1, -1, -1):
        variables[i].tval = ind % variables[i].d
        ind /= variables[i].d

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def setAddr(list variables, int ind):
    _setAddr(variables, ind)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int _getAddr(list variables):
    cdef int ind = 0, multiplier = 1
    cdef int i
    for i in range(len(variables)-1, -1, -1):
        ind += variables[i].tval*multiplier
        multiplier *= variables[i].d 
    return ind

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def getAddr(list variables):
    return _getAddr(variables)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def getPairwiseProb(cnp.ndarray[int, ndim=2] data, int x, int y, int x_d, int y_d, cnp.ndarray[double, ndim=1] weights, double laplace):
    cxy = np.ones(shape=(x_d, y_d))*laplace
    for i in range(x_d):
        for j in range(y_d):
            cxy[i][j] += np.sum(weights[(data[:, x] == i) & (data[:, y] == j)])
    cxy /= np.sum(cxy)
    return cxy

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def getProb(cnp.ndarray[int, ndim=2] data, int x, int x_d, cnp.ndarray[double, ndim=1] weights, double laplace):
    cx = np.ones(shape=(x_d))*laplace
    for i in range(x_d):
        cx[i] += np.sum(weights[data[:, x] == i])
    cx /= np.sum(cx)
    return cx

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef list _computeMI(list variables, cnp.ndarray[int, ndim=2] data, cnp.ndarray[double, ndim=1] weights, double laplace):
        cdef int i, j, nvars = len(variables)

        px = [None]*nvars
        for i in range(nvars):
            u = variables[i]
            px[i] = getProb(data, i, u.d, weights, laplace)
        lpx = np.log(px)

        cdef cnp.ndarray[double, ndim = 2] mi = np.zeros((nvars, nvars), dtype=float)
        pxy = [None]*nvars
        for i in range(nvars):
            u = variables[i]
            pxy[i] = [None]*nvars
            for j in range(nvars):
                v = variables[j]
                pxy[i][j] = getPairwiseProb(data, i, j, u.d, v.d, weights, laplace)
                for xi in range(u.d):
                    for xj in range(v.d):
                        mi[i][j] += pxy[i][j][xi][xj]*(np.log(pxy[i][j][xi][xj])-lpx[i][xi]-lpx[j][xj])
        return [mi, pxy, px]

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def computeMI(list variables, cnp.ndarray[int, ndim=2] data, cnp.ndarray[double, ndim=1] weights=np.array([]), double laplace=1.0):
    if weights.shape[0] == 0:
        weights = np.ones(data.shape[0])
    return _computeMI(variables, data, weights, laplace)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef list _computePxy(list variables, cnp.ndarray[int, ndim=2] data, cnp.ndarray[double, ndim=1] weights, double laplace):
    cdef int i, j, nvars = len(variables)
    cdef cnp.ndarray[double, ndim = 2] mi = np.zeros((nvars, nvars), dtype=float)
    pxy = [None]*nvars
    for i in range(nvars):
        u = variables[i]
        pxy[i] = [None]*nvars
        for j in range(nvars):
            v = variables[j]
            pxy[i][j] = getPairwiseProb(data, i, j, u.d, v.d, weights, laplace)
    return pxy

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def computePxy(list variables, cnp.ndarray[int, ndim=2] data, cnp.ndarray[double, ndim=1] weights=np.array([]), double laplace=1.0):
    if weights.shape[0] == 0:
        weights = np.ones(data.shape[0])
    return _computePxy(variables, data, weights, laplace)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def normalizeDim2(cnp.ndarray[int, ndim=2] data):
    norm_const = np.sum(data, axis = 1)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def updateChowLiuCPT(func, cnp.ndarray[int, ndim=2]data, cnp.ndarray[double, ndim=1] weights, double laplace, dict var_id_ind_map):
    cdef int i, d
    cdef int cpt_var_ind
    variables = func.getVars()
    if len(variables) == 1:
        var = variables[0]
        if var_id_ind_map == {}:
            func.setPotential(getProb(data, var.id, var.d, weights, laplace))
        else:
            func.setPotential(getProb(data, var_id_ind_map[var.id], var.d, weights, laplace))
    else:
        cpt_var_ind = func.getCPTVar()
        u = []
        v = []
        if cpt_var_ind == 0:
            u = variables[1]
            v = variables[0]
        else:
            u = variables[0]
            v = variables[1]
        pxy = []
        px = []
        if var_id_ind_map == {}:
            pxy = getPairwiseProb(data, u.id, v.id, u.d, v.d, weights, laplace)
            px = getProb(data, u.id, u.d, weights, laplace)
        else:
            pxy = getPairwiseProb(data, var_id_ind_map[u.id], var_id_ind_map[v.id], u.d, v.d, weights, laplace)
            px = getProb(data, var_id_ind_map[u.id], u.d, weights, laplace)
        d = len(func.getPotential())
        potentials = np.zeros(d)
        for i in range(d):
            setAddr(variables, i)
            u = []
            v = []
            if cpt_var_ind == 0:
                u = variables[1]
                v = variables[0]
            else:
                u = variables[0]
                v = variables[1]
            potentials[i] = pxy[u.tval][v.tval]/px[u.tval]
        func.setPotential(potentials)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double _computeEntropy(list px):
    cdef double entropy = 0.0
    cdef int i, j
    cdef int nvars = len(px)
    for i in range(nvars):
        entropy += stats.entropy(px[i])
    entropy /= nvars   
    return entropy

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def computeEntropy(list pxy):
    return _computeEntropy(pxy)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def printVarVector(variables):
    cdef int i
    #print("id: ")
    for i in range(len(variables)):
        print "{0:d} ".format(variables[i].id),
    print("\n")
    '''
    print("tval:")
    for i in range(len(variables)):
        print "{0:d} ".format(variables[i].tval),
    print("\n")
    '''

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef list _multiplyBucket(list bucket):
    cdef int i, nfunctions, j 
    nfunctions = len(bucket)
    cdef list bucket_vars = []
    cdef double[:] bucket_potential
    if nfunctions == 1:
        bucket_vars = bucket[0].getVars()
        bucket_potential = bucket[0].getPotential()
    else:
        bucket_vars = []
        for i in range(nfunctions):
            func = bucket[i]
            bucket_vars.extend(func.getVars())
        bucket_vars = list(set(bucket_vars))
        d = getDomainSize(bucket_vars)
        bucket_potential = np.zeros(d)
        for i in range(d):
            setAddr(bucket_vars, i)
            for j in range(nfunctions):
                func = bucket[j]
                temp_vars = func.getVars()
                bucket_potential[i] += np.log(func.getPotential()[getAddr(temp_vars)])
        bucket_potential = np.exp(bucket_potential)
    return [bucket_vars, bucket_potential]
            
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def multiplyBucket(list bucket):
    return _multiplyBucket(bucket)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef list _elimVarBucket(list bucket_vars, double[:] bucket_potential, list elim_vars):
    cdef int i, j , marg_d, elim_d
    cdef cnp.ndarray[double, ndim=1] marg_potential
    cdef list marg_vars
    marg_vars = []
    for i in range(len(bucket_vars)):
        marg_vars.append(bucket_vars[i])
    marg_d = bucket_potential.shape[0]
    for i in range(len(elim_vars)):
        var = elim_vars[i]
        marg_vars.remove(var)
        marg_d /= var.d
    marg_potential = np.zeros(marg_d)
    elim_d = int(bucket_potential.shape[0]/marg_d)
    for i in range(elim_d):
        setAddr(elim_vars, i)
        for j in range(marg_d):
            setAddr(marg_vars, j)
            marg_potential[j] += bucket_potential[getAddr(bucket_vars)]
    return [marg_vars, marg_potential]
            
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def elimVarBucket(list bucket_vars, double[:] bucket_potential, list elim_vars):
    return _elimVarBucket(bucket_vars, bucket_potential, elim_vars)