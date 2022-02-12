import numpy as np 
cimport numpy as cnp 
cimport cython 
from libcpp cimport bool


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef class CNode:
    cdef public int id
    cdef public object clt
    cdef public double[:] child_weights
    cdef public list children
    cdef public int node_type
    cdef public int[:] features

    def __init__(self):
        self.id = -1
        self.child_weights = np.array([])
        self.children = []
        self.clt = None
        self.node_type = -1
        self.features = np.array([], dtype=np.int32)