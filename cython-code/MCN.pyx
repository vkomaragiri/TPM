import numpy as np
cimport numpy as cnp
from Variable import Variable
from CN import CN
cimport cython
from libcpp cimport bool
import sys


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef class MCN:
    cdef double[:] prob_mixture
    cdef list cns
    cdef int ncomponents
    cdef list variables

    cdef bool set_evid

    def __init__(self):
        self.prob_mixture = np.array([])
        self.cns = []
        self.ncomponents = 0
        self.variables = []

        self.set_evid = False

    cdef void _learnEM(self, cnp.ndarray[int, ndim=2] train, cnp.ndarray[int, ndim=2] valid, int ncomponents, int num_iter):
        cdef cnp.ndarray[int, ndim=1] dsize = np.max(train, axis = 0)+1
        self.ncomponents = ncomponents
        cdef int i, it, j
        cdef int nexamples = train.shape[0]
        cdef int nvars = train.shape[1]
        cdef double laplace = 1.0/ncomponents

        cdef cnp.ndarray[double, ndim=2] weights = np.random.rand(ncomponents, nexamples)
        weights /= np.sum(weights, axis=0)
        for i in range(nvars):
            var = Variable(i, dsize[i])
            self.variables.append(var)
        for i in range(ncomponents):
            cn = CN() 
            cn.setVars(self.variables)
            self.cns.append(cn)
        for it in range(num_iter):
            print("EM iteration", it)
            #M-Step
            self.prob_mixture = np.sum(weights, axis=1)/np.sum(weights)
            #self.prob_mixture /= np.sum(self.prob_mixture)
            for i in range(ncomponents):
                if it % 10 == 0:
                    self.cns[i].learn(train, valid, weights[i, :], prune=True, max_depth=3, learn_struct=True, is_component=True, laplace=laplace)
                else:
                    self.cns[i].learn(train, valid, weights[i, :], prune=False, max_depth=3, learn_struct=False, is_component=True, laplace=laplace)
            #E-Step
            for i in range(ncomponents):
                for j in range(nexamples):
                    weights[i][j] = self.prob_mixture[i]*self.cns[i].getProbability(train[j, :])
            weights /= np.sum(weights, axis=0)
    
    def learnEM(self, cnp.ndarray[int, ndim=2] train, cnp.ndarray[int, ndim=2] valid, int ncomponents=10, int num_iter=100):
        self._learnEM(train, valid, ncomponents, num_iter)

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
            lprob += np.exp(np.log(self.prob_mixture[i])+self.cns[i].getLogProbability(example))
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
        fw.write("MCN\n")
        cdef int nvars, i, ncomponents
        nvars = len(self.variables)
        ncomponents = self.ncomponents
        fw.write(str(ncomponents)+" "+str(nvars)+"\n")
        for i in range(nvars):
            fw.write(str(self.variables[i].d)+" ")
        fw.write("\n")
        for i in range(ncomponents):
            fw.write("{0:.3f} ".format(self.prob_mixture[i]))
        fw.write("\n")
        for i in range(ncomponents):
            self.cns[i].writeCNode(self.cns[i].getRoot(), fw)

    def read(self, infilename):
        fr = open(infilename, "r")
        line = fr.readline()
        if "MCN" != line[:(len(line)-1)]:
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
            cn = CN()
            cn.setVars(self.variables)
            cn.setRoot(cn.readCNode(fr))
            self.cns.append(cn)
    
    def setEvidence(self, int id, int val):
        self.variables[id].setValue(val)

    cdef void _instantiateEvidNetwork(self):
        cdef int i 
        for i in range(self.ncomponents):
            self.cns[i].instantiateEvidNetwork()

    def instantiateEvidNetwork(self):
        self._instantiateEvidNetwork()

    cdef double _getPE(self):
        cdef double pe = 0.0 
        cdef int i 
        if self.set_evid == False:
            self._instantiateEvidNetwork()
        for i in range(self.ncomponents):
            pe += self.prob_mixture[i]*self.cns[i].getPE()
        return pe 

    def getPE(self):
        return self._getPE()

    cdef list _getVarMarginals(self):
        cdef int i, nvars, j 
        cdef double temp
        if self.set_evid == False:
            self._instantiateEvidNetwork()
        nvars = len(self.variables)
        marginals = []
        post_prob = np.zeros(self.ncomponents)
        for i in range(self.ncomponents):
            post_prob[i] = self.cns[i].getPE()*self.prob_mixture[i]
            marginals.append(post_prob[i]*self.cns[i].getVarMarginals())
        post_prob /= np.sum(post_prob)
        marginals = np.sum(marginals, axis=0)
        for i in range(nvars):
            temp = np.sum(marginals[i])
            #print(temp)
            marginals[i] /= temp
        return list(marginals)

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
            samples[k:k+j, :] = self.cns[i].generatePriorSamples(j)
            k += j
        out = samples
        return out

    def generatePriorSamples(self, int n):
        return self._generatePriorSamples(n)

    cdef int[:, :] _generatePosteriorSamples(self, int n):
        cdef int i, nvars, j, k
        cdef int[:, :] out 
        nvars = len(self.variables)
        post_prob = np.zeros(self.ncomponents)
        for i in range(self.ncomponents):
            post_prob[i] = self.cns[i].getPE()*self.prob_mixture[i]
        post_prob /= np.sum(post_prob)
        cdef cnp.ndarray[int, ndim=1] temp = np.asarray(np.random.choice(a=self.ncomponents, size=(n), p=post_prob), dtype=np.int32)
        cdef cnp.ndarray[int, ndim=2] samples = -1*np.ones((n, nvars), dtype=np.int32)
        k = 0
        for i in range(self.ncomponents):
            j = np.sum(temp == i)
            samples[k:k+j, :] = self.cns[i].generatePosteriorSamples(j)
            k += j
        out = samples
        return out

    def generatePosteriorSamples(self, int n):
        return self._generatePosteriorSamples(n)