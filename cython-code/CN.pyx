import numpy as np 
cimport numpy as cnp 
cimport cython 
from libcpp cimport bool
from Variable import Variable
from Util import computeMI, computeEntropy, getProb
from BN import BN
from CNode import CNode
import sys 


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef class CN:
    cdef list variables 
    cdef object root

    def __init__(self):
        self.variables = []
        self.root = None

    cdef bool _termination_condition(self, int nexamples, int depth, double entropy, int nfeatures, int max_depth):
        if (nexamples < 10) or (depth >= max_depth) or (entropy < 0.01) or (nfeatures < 1):
            return True
        else:
            return False

    def termination_condition(self, int nexamples, int depth, double entropy, int nfeatures, int max_depth):
        return self._termination_condition(nexamples, depth, entropy, nfeatures, max_depth)

    cdef object _learnCNode(self, cnp.ndarray[int, ndim=2] data, cnp.ndarray[double, ndim=1] data_weights, int depth, cnp.ndarray[int, ndim=1] features, int max_depth, double laplace):
        if features.shape[0] <= 0 or data.shape[0] <= 0:
            return None
        cnode = CNode()
        cnode.features = features
        cdef cnp.ndarray[double, ndim=2] mi 
        cdef list cur_variables = list(np.array(self.variables)[features])
        mi, pxy, px = computeMI(cur_variables, data[:,features], data_weights, laplace)
        cdef int split_ind, split_features_ind, i, j
        cdef cnp.ndarray[int, ndim=1] new_features
        cdef double entropy = computeEntropy(px)
        if self.termination_condition(data.shape[0], depth, entropy, features.shape[0], max_depth):
            cnode.node_type = 1
            cnode.clt = BN()
            cnode.clt.setVars(cur_variables)
            dct = {}
            for j in range(features.shape[0]):
                dct[features[j]] = j
            cnode.clt.setVarIdInd(dct)
            cnode.clt.learnCLT(data=data[:, features], weights=data_weights, learn_struct=True, is_component=True, mi=mi, px=px, pxy=pxy, laplace=laplace)
        else:
            split_ind = np.argmax(np.sum(mi, axis=1))
            split_features_ind = features[split_ind]
            cnode.node_type = 0
            cnode.id = split_features_ind
            cnode.child_weights = px[split_ind]
            new_features = np.delete(features, split_ind)
            for i in range(self.variables[split_features_ind].d):
                indices = data[:, split_features_ind ] == i
                cnode.children.append(self._learnCNode(data[indices, :], data_weights[indices], depth+1, new_features, max_depth, laplace))
        return cnode

    def learnCNode(self, cnp.ndarray[int, ndim=2] data, cnp.ndarray[double, ndim=1] data_weights, int depth, cnp.ndarray[int, ndim=1] features, int max_depth, double laplace):
        return self._learnCNode(data, data_weights, depth, features, max_depth, laplace)

    cdef object _updateCNode(self, object nd, cnp.ndarray[int, ndim=2] data, cnp.ndarray[double, ndim=1] weights, double laplace):
        if nd == None:
            return nd
        if nd.node_type == 0:
            nd.child_weights = getProb(data, nd.id, self.variables[nd.id].d, weights, laplace)
            for i in range(self.variables[nd.id].d):
                indices = data[:, nd.id] == i
                nd.children[i] = self._updateCNode(nd.children[i], data[indices, :], weights[indices], laplace)
        elif nd.node_type == 1:
            nd.clt.learnCLT(data[:, nd.features], weights, learn_struct=False, is_component=True, laplace=laplace)
        return nd

    def updateCNode(self, object nd, cnp.ndarray[int, ndim=2] data, cnp.ndarray[double, ndim=1] weights, double laplace):
        return self._updateCNode(nd, data, weights, laplace)

    cdef object _pruneCNode(self, object nd, cnp.ndarray[int, ndim=2] train, cnp.ndarray[int, ndim=2] valid, double laplace):
        if nd == None or nd.node_type == 1:
            return nd
        cdef int i
        for i in range(len(nd.children)):
            nd.children[i] = self._pruneCNode(nd.children[i], train, valid, laplace)
        cdef double curr_ll, new_ll
        cur_ll = self.getLLScore(valid)
        nd.clt = BN()
        cdef cnp.ndarray[int, ndim=1] cur_features = np.asarray(nd.features)
        cdef list cur_variables = list(np.array(self.variables)[cur_features])
        nd.clt.setVars(cur_variables)
        nd.node_type = 1
        dct = {}
        for j in range(nd.features.shape[0]):
            dct[nd.features[j]] = j
        nd.clt.setVarIdInd(dct)
        nd.clt.learnCLT(train[:,cur_features], learn_struct=True, is_component=True, laplace=laplace)
        new_ll = self.getLLScore(valid)
        if new_ll < cur_ll:
            nd.clt = None
            nd.node_type = 0
        return nd

    def pruneCNode(self, object nd, cnp.ndarray[int, ndim=2] train, cnp.ndarray[int, ndim=2] valid, double laplace):
        return self._pruneCNode(nd, train, valid, laplace)

    cdef void _learn(self, cnp.ndarray[int, ndim=2] train, cnp.ndarray[int, ndim=2] valid, cnp.ndarray[double, ndim=1] train_weights, bool prune, int max_depth, bool is_component, double laplace):
        print("Learning Cutset Network")
        cdef cnp.ndarray[int, ndim=1] dsize = np.max(train, axis = 0)+1
        cdef int nexamples = train.shape[0]
        cdef int nvars = train.shape[1]
        if is_component==False:
            for i in range(nvars):
                var = Variable(i, dsize[i])
                self.variables.append(var)
        nvars = len(self.variables)
        cdef cnp.ndarray[int, ndim=1] features = np.arange(nvars, dtype=np.int32)
        self.root = self.learnCNode(train, train_weights, 0, features, max_depth, laplace)
        print("Log-likelihood score on valid data:", self.getLLScore(valid))
        if prune:
            print("Performing bottom-up pruning on Cutset Network")
            self.root = self.pruneCNode(self.root, train, valid, laplace)
            print("Log-likelihood score on valid data:", self.getLLScore(valid))

    def learn(self, cnp.ndarray[int, ndim=2] train, cnp.ndarray[int, ndim=2] valid, cnp.ndarray[double, ndim=1] train_weights = np.array([]), bool prune=True, int max_depth=10, bool learn_struct=True, bool is_component=False, double laplace=1.0):
        if train_weights.shape[0] == 0:
            train_weights = np.ones(train.shape[0])
        if learn_struct == True:
            self._learn(train, valid, train_weights, prune, max_depth, is_component, laplace)
        else:
            self.root = self.updateCNode(self.root, train, train_weights, laplace)
    
    cdef double _getLogProbabilityCNode(self, object nd, cnp.ndarray[int, ndim=1] example, double out):
        cdef int child_ind
        if nd != None:
            if nd.node_type == 0:
                child_ind = example[nd.id]
                out += np.log(nd.child_weights[child_ind])
                out = self._getLogProbabilityCNode(nd.children[child_ind], example, out)
            else:
                out += nd.clt.getLogProbability(example)
        return out
        
    cdef double _getLogProbability(self, cnp.ndarray[int, ndim=1] example):
        cdef double lprob=0.0
        return self._getLogProbabilityCNode(self.root, example, lprob)
        
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

    def writeCNode(self, nd, fw):
        cdef int nfeatures
        cdef int i 
        fw.write("BEGIN\n")
        if nd == None:
            fw.write("NULL\n")
        else:
            nfeatures = nd.features.shape[0]
            for i in range(nfeatures):
                fw.write(str(nd.features[i])+" ")
            fw.write("\n")
            if nd.node_type == 0:
                fw.write("OR\n")
                fw.write(str(nd.id)+"\n")
                for i in range(len(nd.children)):
                    fw.write("{0:.3f} ".format(nd.child_weights[i]))
                fw.write("\n")
                for i in range(len(nd.children)):
                    self.writeCNode(nd.children[i], fw)
            else:
                fw.write("CLT\n")
                nd.clt.writeCN(fw)
        fw.write("END\n")

    def readCNode(self, fr):
        cdef int i, nfeatures, j
        cdef list cur_variables
        line = fr.readline()
        if "BEGIN" not in line:
            print("Invalid file format")
            sys.exit(0)
        line = fr.readline()
        if "NULL" in line:
            nd = None
        else:
            nd = CNode()
            nd.features = np.array(line[:(len(line)-2)].split(" "), dtype=np.int32)
            line = fr.readline()
            if "OR" in line:
                nd.node_type = 0
                line = fr.readline()
                nd.id = int(line[:(len(line)-1)])
                line = fr.readline()
                nd.child_weights = np.array(line[:(len(line)-2)].split(" "), dtype=float)
                for i in range(nd.child_weights.shape[0]):
                    nd.children.append(self.readCNode(fr))
            elif "CLT" in line:
                nd.node_type = 1
                nd.clt = BN()
                cur_variables = list(np.array(self.variables)[nd.features])
                nd.clt.setVars(cur_variables)
                dct = {}
                for j in range(len(nd.features)):
                    dct[nd.features[j]] = j
                nd.clt.setVarIdInd(dct)
                nd.clt.readCN(fr)
            else:
                print("Error in file")
                sys.exit(0)
        line = fr.readline()
        if "END" not in line:
            print("Invalid format")
            sys.exit(0)
        return nd

    def write(self, outfilename):
        fw = open(outfilename, "w")
        fw.write("CN\n")
        cdef int nvars, i
        nvars = len(self.variables)
        fw.write(str(nvars)+"\n")
        for i in range(nvars):
            fw.write(str(self.variables[i].d)+" ")
        fw.write("\n")
        self.writeCNode(self.root, fw)

    def read(self, infilename):
        fr = open(infilename, "r")
        if "CN" not in fr.readline():
            print("Invalid file format")
            sys.exit(0)
        cdef int nvars, i 
        line = fr.readline()
        nvars = int(line[:(len(line)-1)])
        line = fr.readline()
        dsize = np.array(line[:(len(line)-2)].split(" "), dtype=int)
        for i in range(nvars):
            var = Variable(i, dsize[i])
            self.variables.append(var)
        self.root = self.readCNode(fr)

    cdef void _setVars(self, vars):
        self.variables = vars

    def setVars(self, vars):
        self._setVars(vars)

    cdef list _getVars(self):
        return self.variables

    def getVars(self):
        return self._getVars()

    cdef object _getRoot(self):
        return self.root

    def getRoot(self):
        return self._getRoot()
    
    cdef void _setRoot(self, object nd):
        self.root = nd

    def setRoot(self, object nd):
        self._setRoot(nd)
