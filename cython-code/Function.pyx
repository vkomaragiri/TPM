import numpy as np 
cimport numpy as cnp
from Variable import Variable
from Util import getDomainSize
cimport cython 

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef class Function:
    cdef list variables
    cdef int cpt_var_ind 
    cdef double[:] potentials 

    def __init__(self, variables_ = [], potentials_ = np.array([])):
        self.variables = variables_
        self.cpt_var_ind = -1
        self.potentials = potentials_
        
    cdef void _setVars(self, list vars):
        self.variables = vars
    
    def setVars(self, list vars):
        self._setVars(vars)

    cdef list _getVars(self):
        return self.variables

    def getVars(self):
        return self._getVars()

    cdef void _setPotential(self, cnp.ndarray[double, ndim=1] potentials_):
        self.potentials = potentials_

    def setPotential(self, cnp.ndarray[double, ndim=1] potentials_):
        self._setPotential(potentials_)

    cdef double[:] _getPotential(self):
        return self.potentials

    def getPotential(self):
        return self._getPotential()

    cdef void _setCPTVar(self, int cvar):
        self.cpt_var_ind = cvar

    def setCPTVar(self, int cvar):
        self._setCPTVar(cvar)

    cdef int _getCPTVar(self):
        return self.cpt_var_ind

    def getCPTVar(self):
        return self._getCPTVar()

        






