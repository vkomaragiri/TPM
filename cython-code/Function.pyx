import numpy as np 
cimport numpy as cnp
from Variable import Variable
from Util import getDomainSize, setAddr, getAddr
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

    cdef object _instantiateEvid(self):
        out = Function()
        cdef int i, j, d_non_evid, d, nvars
        nvars = len(self.variables)
        non_evid_vars = []
        for i in range(nvars):
            if self.variables[i].isEvid():
                self.variables[i].tval = self.variables[i].val 
            else:
                non_evid_vars.append(self.variables[i])
        out.setVars(non_evid_vars)
        if len(non_evid_vars) == nvars:
            out.setPotential(np.asarray(self.potentials))
            out.setCPTVar(self.cpt_var_ind)
        else:
            d = getDomainSize(non_evid_vars)
            temp = -1*np.ones(d)
            for i in range(d):
                setAddr(non_evid_vars, i)
                temp[i] = self.potentials[getAddr(self.variables)]
            out.setPotential(temp)
        return out        


    def instantiateEvid(self):
        return self._instantiateEvid()






