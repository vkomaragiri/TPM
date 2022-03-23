import numpy as np
cimport numpy as cnp
from Variable import Variable
from BN import BN 
cimport cython
from libcpp cimport bool
import sys
from Function import Function
from Util import computePxy

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef class MT:
    cdef double[:] prob_mixture
    cdef list clts
    cdef int ncomponents
    cdef list variables

    def __init__(self):
        self.prob_mixture = np.array([])
        self.clts = []
        self.ncomponents = 0
        self.variables = []

    cdef void _learnEM(self, cnp.ndarray[int, ndim=2] data, int ncomponents, int num_iter):
        cdef cnp.ndarray[int, ndim=1] dsize = np.max(data, axis = 0)+1
        self.ncomponents = ncomponents
        cdef int i, it, j
        cdef int nexamples = data.shape[0]
        cdef int nvars = data.shape[1]
        cdef cnp.ndarray[double, ndim=2] weights = np.random.rand(ncomponents, nexamples)
        weights /= np.sum(weights, axis=0)
        for i in range(nvars):
            var = Variable(i, dsize[i])
            self.variables.append(var)
        for i in range(ncomponents):
            bn = BN() 
            bn.setVars(self.variables)
            self.clts.append(bn)
        cdef double laplace = 1.0/ncomponents
        for it in range(num_iter):
            #M-Step
            self.prob_mixture = np.sum(weights, axis=1)/np.sum(weights)
            #self.prob_mixture /= np.sum(self.prob_mixture)
            for i in range(ncomponents):
                if it % 10 == 0:
                    self.clts[i].learnCLT(data, weights[i, :], True, True, laplace)
                else:
                    self.clts[i].learnCLT(data, weights[i, :], False, True, laplace)

            #E-Step
            for i in range(ncomponents):
                for j in range(nexamples):
                    weights[i][j] = self.prob_mixture[i]*self.clts[i].getProbability(data[j, :])
            weights /= np.sum(weights, axis=0)
    
    def learnEM(self, cnp.ndarray[int, ndim=2] data, int ncomponents=10, int num_iter=100):
        self._learnEM(data, ncomponents, num_iter)

    '''
    cdef void _learnRF(self, cnp.ndarray[int, ndim=2] train, cnp.ndarray[int, ndim=2] valid, int ncomponents, double r):
        cdef cnp.ndarray[int, ndim=1] dsize = np.max(data, axis = 0)+1
        self.ncomponents = ncomponents
        cdef int i, j
        cdef int nexamples = train.shape[0]
        cdef int nvars = train.shape[1]
        cdef cnp.ndarray[int, ndim=2] bootstrap = np.random.randint(low=0, high=nexamples, size=(ncomponents, nexamples))

        for i in range(nvars):
            var = Variable(i, dsize[i])
            self.variables.append(var)
        for i in range(ncomponents):
            bn = BN() 
            bn.setVars(self.variables)
            bn.learnCLTRF(train[bootstrap[i, :], :], r)
            self.clts.append(bn)
            self.prob_mixture.append(bn.getLLScore(valid))
        self.prob_mixture /= np.sum(self.prob_mixture)
        
    
    def learnRF(self, cnp.ndarray[int, ndim=2] train, cnp.ndarray[int, ndim=2] valid, int ncomponents=10, double r=0.2):
        self._learnRF(train, valid, ncomponents, r)
    '''
    
    cdef double _getLogProbability(self, cnp.ndarray[int, ndim=1] example):
        cdef double lprob = 0.0
        cdef int i
        for i in range(self.ncomponents):
            lprob += np.exp(np.log(self.prob_mixture[i])+self.clts[i].getLogProbability(example))
        return np.log(lprob)
        
    def getLogProbability(self, cnp.ndarray[int, ndim=1] example):
        return self._getLogProbability(example)

    def getProbability(self, cnp.ndarray[int, ndim=1] example):
        return np.exp(self.getLogProbability(example))

    cdef double _getLLScore(self, cnp.ndarray[int, ndim=2] data):
        cdef double ll_score = 0.0
        cdef int i
        for i in range(data.shape[0]):
            ll_score += self.getLogProbability(data[i, :])
        return ll_score/data.shape[0]

    def getLLScore(self, cnp.ndarray[int, ndim=2] data):
        return self._getLLScore(data)
    
    cdef void _setVars(self, vars):
        self.variables = vars

    def setVars(self, vars):
        self._setVars(vars)

    cdef list _getVars(self):
        return self.variables

    def getVars(self):
        return self._getVars()

    def write(self, outfilename):
        fw = open(outfilename, "w")
        fw.write("MT\n")
        cdef int nvars = len(self.variables), i, ncomponents = self.ncomponents, j, nfunctions, k
        fw.write(str(ncomponents)+" "+str(nvars)+"\n")
        for i in range(nvars):
            fw.write(str(self.variables[i].d)+" ")
        fw.write("\n")
        for i in range(ncomponents):
            fw.write("{0:.3f} ".format(self.prob_mixture[i]))
        fw.write("\n")
        for i in range(ncomponents):
            clt = self.clts[i]
            clt_functions = clt.getFunctions()
            nfunctions = len(clt_functions)
            fw.write(str(nfunctions)+"\n")
            for j in range(nfunctions):
                vars_ = clt_functions[j].getVars()
                fw.write(str(len(vars_))+" ")
                for k in range(len(vars_)):
                    fw.write(str(vars_[k].id)+" ")
                fw.write(str(clt_functions[j].getCPTVar())+"\n")
            for j in range(nfunctions):
                potentials_ = clt_functions[j].getPotential()
                fw.write(str(len(potentials_))+"\n")
                for k in range(len(potentials_)):
                    fw.write("{0:.3f} ".format(potentials_[k]))
                fw.write("\n")
    
    def read(self, infilename):
        fr = open(infilename, "r")
        line = fr.readline()
        if "MT" != line[:(len(line)-1)]:
            print("Invalid format")
            sys.exit(0)
        cdef int nvars, i, j, k, ncomponents, nfunctions 
        line = fr.readline()
        ncomponents, nvars = np.array(line[:(len(line)-1)].split(" "), dtype=int)
        self.ncomponents = ncomponents
        line = fr.readline()
        dsize = np.array(line[:(len(line)-2)].split(" "), dtype=int)
        for i in range(nvars):
            var = Variable(i, dsize[i])
            self.variables.append(var)
        line = fr.readline()
        self.prob_mixture = np.array(line[:(len(line)-2)].split(" "), dtype=float)
        for i in range(ncomponents):
            bn = BN() 
            bn.setVars(self.variables)
            line = fr.readline()
            nfunctions = int(line[:(len(line)-1)])
            bn_functions = []
            for j in range(nfunctions):
                func = Function()
                line = fr.readline()
                func_var_params = np.array(line[:(len(line)-1)].split(" "), int)
                func.setCPTVar(func_var_params[func_var_params.shape[0]-1])
                func_vars = []
                for k in range(func_var_params[0]):
                    func_vars.append(self.variables[func_var_params[k+1]])
                func.setVars(func_vars)
                bn_functions.append(func)
            for j in range(nfunctions):
                fr.readline()
                line = fr.readline()
                bn_functions[j].setPotential(np.array(line[:(len(line)-2)].split(" "), dtype=float))
            bn.setFunctions(bn_functions)
            self.clts.append(bn)

    def setEvidence(self, int id, int val):
        self.variables[id].setValue(val)

    cdef void _initBTP(self):
        cdef int i 
        for i in range(self.ncomponents):
            self.clts[i].initBTP()

    def initBTP(self):
        self._initBTP()

    cdef double _getPE(self):
        cdef double pe = 0.0 
        cdef int i 
        for i in range(self.ncomponents):
            pe += self.prob_mixture[i]*self.clts[i].getPE()
        return pe 

    def getPE(self):
        return self._getPE()

    cdef double[:, :] _getVarMarginals(self):
        cdef int i, nvars, j 
        cdef double temp
        nvars = len(self.variables)
        marginals = []
        post_prob = np.zeros(self.ncomponents)
        for i in range(self.ncomponents):
            post_prob[i] = self.clts[i].getPE()*self.prob_mixture[i]
            marginals.append(post_prob[i]*self.clts[i].getVarMarginals())
        post_prob /= np.sum(post_prob)
        marginals = np.sum(marginals, axis=0)
        for i in range(nvars):
            temp = np.sum(marginals[i])
            marginals[i] /= temp
        cdef double[:, :] out
        out = marginals
        return out

    def getVarMarginals(self):
        return self._getVarMarginals()

    cdef int[:, :] _generatePriorSamples(self, int n):
        cdef int i, j, k, nvars=len(self.variables)
        cdef int[:, :] out 
        self.prob_mixture = np.asarray(self.prob_mixture)/np.sum(self.prob_mixture)
        cdef cnp.ndarray[int, ndim=1] temp = np.asarray(np.random.choice(a=self.ncomponents, size=(n), p=self.prob_mixture), dtype=np.int32)
        cdef cnp.ndarray[int, ndim=2] samples = -1*np.ones((n, nvars), dtype=np.int32)
        k = 0
        for i in range(self.ncomponents):
            j = np.sum(temp == i)
            samples[k:k+j, :] = self.clts[i].generatePriorSamples(j)
            k += j
        out = samples
        return out

    def generatePriorSamples(self, int n):
        return self._generatePriorSamples(n)
