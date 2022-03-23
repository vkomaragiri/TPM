from __future__ import print_function
import numpy as np
cimport numpy as cnp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_tree, connected_components
from Variable import Variable
from Function import Function
from Util import getDirectedST, getDomainSize, setAddr, getAddr, getPairwiseProb, getProb, updateChowLiuCPT, computeMI, printVarVector, multiplyBucket, elimVarBucket
from Util import getTopologicalOrder
cimport cython 
from libcpp cimport bool
import sys
import traceback



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef class BN:
    cdef list variables
    cdef list functions
    cdef dict var_id_ind_map
    
    cdef int[:] order
    cdef int[:] var_pos
    cdef double pe
    cdef bool upward_pass 
    cdef bool downward_pass
    cdef list buckets
    cdef int[:, :] messages

    cdef list sampling_distributions 

    def __init__(self):
        self.variables = []
        self.functions = []
        self.var_id_ind_map = None

        self.pe = 0.0
        self.upward_pass = False 
        self.downward_pass = False

        self.sampling_distributions = None


    cdef void _learnCLT(self, cnp.ndarray[int, ndim = 2] data, cnp.ndarray[double, ndim=1] weights, bool learn_struct, bool is_component, double laplace, cnp.ndarray[double, ndim=2] mi, list px, list pxy):
        cdef cnp.ndarray[int, ndim=1] dsize = np.max(data, axis = 0)+1
        cdef int nvars
        cdef int i, d, j, u, v
        if is_component == False:
            for i in range(data.shape[1]):
                var = Variable(i, dsize[i])
                self.variables.append(var)
        nvars = len(self.variables)
        if learn_struct == True:
            if mi.shape[0] == 0 or mi.shape[1] == 0:
                mi, pxy, px = computeMI(self.variables, data, weights, laplace)
            children, parents = getDirectedST(-mi)
            for i in range(nvars):
                u = parents[i]
                v = children[i]
                func = Function()
                if u == v:
                    func.setVars([self.variables[u]])
                    func.setPotential(px[u])
                    func.setCPTVar(0)
                else:
                    func.setVars([self.variables[u], self.variables[v]])
                    func.setCPTVar(1)
                    d = getDomainSize(func.getVars())
                    potentials = np.zeros(d)
                    for j in range(d):
                        setAddr(func.getVars(), j)
                        potentials[j] = pxy[u][v][self.variables[u].tval][self.variables[v].tval]/np.sum(pxy[u][v][self.variables[u].tval])
                    func.setPotential(potentials)
                self.functions.append(func)
            #add chow-liu graphs code later
        else:
            for i in range(len(self.functions)):
                updateChowLiuCPT(self.functions[i], data, weights, laplace, self.var_id_ind_map)
  
    def learnCLT(self, cnp.ndarray[int, ndim = 2] data, cnp.ndarray[double, ndim=1] weights=np.array([]), bool learn_struct=True, bool is_component=False, double laplace=1.0, cnp.ndarray[double, ndim=2] mi = np.array([[]]), list px = [], list pxy = []):
        self._learnCLT(data, weights, learn_struct, is_component, laplace, mi, px, pxy)

    cdef double _getLogProbability(self, cnp.ndarray[int, ndim=1] example):
        cdef double lprob = 0.0
        cdef int i 
        cdef int nvars = len(self.variables)
        for i in range(nvars):
            self.variables[i].tval = example[self.variables[i].id]
        for i in range(nvars):
            func = self.functions[i]
            lprob += np.log(func.getPotential()[getAddr(func.getVars())])
        return lprob
        
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

    cdef void _setVarIdInd(self, dct):
        self.var_id_ind_map = dct

    def setVarIdInd(self, dct):
        self._setVarIdInd(dct)

    cdef dict _getVarIdInd(self):
        return self.var_id_ind_map

    def getVarIdInd(self):
        return self._getVarIdInd()

    cdef list _getFunctions(self):
        return self.functions
    
    def getFunctions(self):
        return self._getFunctions()

    cdef void _setFunctions(self, list functions):
        self.functions = functions
    
    def setFunctions(self, list functions):
        self._setFunctions(functions)

    def write(self, outfilename):
        fw = open(outfilename, "w")
        fw.write("BAYES\n")
        cdef int i, nvars, nfunctions
        nvars = len(self.variables)
        nfunctions = len(self.functions)
        fw.write(str(nvars)+"\n")
        for i in range(nvars):
            fw.write(str(self.variables[i].d)+" ")
        fw.write("\n")
        fw.write(str(nfunctions)+"\n")
        for i in range(nfunctions):
            func = self.functions[i]
            func_variables = func.getVars()
            fw.write(str(len(func_variables))+" ")
            for j in range(len(func_variables)):
                fw.write(str(func_variables[j].id)+" ")
            fw.write(str(func.getCPTVar()))
            fw.write("\n")
        for i in range(nfunctions):
            func = self.functions[i]
            func_potentials = func.getPotential()
            fw.write(str(len(func_potentials))+"\n")
            for j in range(len(func_potentials)):
                fw.write("{0:.3f} ".format(func_potentials[j]))
            fw.write("\n")
    
    def read(self, infilename):
        fr = open(infilename, "r")
        if "BAYES" not in fr.readline():
            print("Invalid input")
            sys.exit(0)
        cdef int nvars, nfunctions, i, j, n
        cdef double k
        line = fr.readline()
        nvars = int(line[:(len(line)-1)])
        line = fr.readline()
        dsize = np.array(line[:(len(line)-2)].split(" "), dtype=int)
        for i in range(nvars):
            var = Variable(i, dsize[i])
            self.variables.append(var)  
        line = fr.readline()
        nfunctions = int(line[:(len(line)-1)])
        for i in range(nfunctions):
            func = Function()
            line = fr.readline()
            vars_ = np.array(line[:(len(line)-1)].split(" "), dtype=int)
            func_variables = []
            for j in range(1, len(vars_)-1):
                func_variables.append(self.variables[vars_[j]])
            func.setVars(func_variables)
            func.setCPTVar(vars_[len(vars_)-1])
            self.functions.append(func)
        for i in range(nfunctions):
            fr.readline()
            line = fr.readline()
            potentials_ = np.array(line[:(len(line)-2)].split(" "), dtype=float)
            self.functions[i].setPotential(potentials_)
        

    def writeCN(self, fw):
        cdef int nfunctions = len(self.functions), i
        fw.write(str(nfunctions)+"\n")
        for i in range(nfunctions):
            func = self.functions[i]
            func_variables = func.getVars()
            fw.write(str(len(func_variables))+" ")
            for j in range(len(func_variables)):
                fw.write(str(func_variables[j].id)+" ")
            fw.write(str(func.getCPTVar()))
            fw.write("\n")
        for i in range(nfunctions):
            func = self.functions[i]
            func_potentials = func.getPotential()
            fw.write(str(len(func_potentials))+"\n")
            for j in range(len(func_potentials)):
                fw.write("{0:.3f} ".format(func_potentials[j]))
            fw.write("\n")

    def readCN(self, fr):
        cdef int nfunctions, i, j, ind
        line = fr.readline()
        nfunctions = int(line[:(len(line)-1)])
        for i in range(nfunctions):
            func = Function()
            line = fr.readline()
            vars_ = np.array(line[:(len(line)-1)].split(" "), dtype=int)
            func_variables = []
            for j in range(1, vars_.shape[0]-1):
                ind = self.var_id_ind_map[vars_[j]]
                func_variables.append(self.variables[ind])
            func.setVars(func_variables)
            func.setCPTVar(vars_[len(vars_)-1])
            self.functions.append(func)
        for i in range(nfunctions):
            fr.readline()
            line = fr.readline()
            self.functions[i].setPotential(np.array(line[:(len(line)-2)].split(" "), dtype=float))

    cdef void _getOrder(self, long order_type):
        cdef int nvars = len(self.variables)
        if order_type == 0: #Random order
            temp = np.arange(nvars, dtype=np.int32)
            np.random.shuffle(temp)
            self.order = temp
        elif order_type == 1: #Min-degree
            self.order = -1*np.ones(nvars, dtype=np.int32)
        elif order_type == 2: #Min-fill
            self.order = -1*np.ones(nvars, dtype=np.int32)
            
    
    def getOrder(self, long order_type):
        self._getOrder(order_type)

    cdef void _initBTP(self):
        self.getOrder(0)
        cdef cnp.ndarray[int, ndim=1] var_pos
        cdef int i, bucket_ind, j, temp_nvars
        cdef int nvars = len(self.variables)
        cdef int nfunctions = len(self.functions)
        self.var_pos = -1*np.ones(nvars, dtype=np.int32)
        self.buckets = []
        for i in range(nvars):
            self.var_pos[self.order[i]] = i
            self.buckets.append([])

        bucket_ind = nfunctions
        for i in range(nfunctions):
            newf = self.functions[i].instantiateEvid()
            newf_vars = newf.getVars()
            temp_nvars = len(newf_vars)
            if temp_nvars == 0:
                self.pe += np.log(newf.getPotential()[0])
                continue
            if self.var_id_ind_map == None:
                bucket_ind = np.min([int(self.var_pos[newf_vars[j].id]) for j in range(temp_nvars)])
            else:
                bucket_ind = np.min([int(self.var_pos[self.var_id_ind_map[newf_vars[j].id]]) for j in range(temp_nvars)])
            self.buckets[bucket_ind].append(newf)

    def initBTP(self):
        self._initBTP()

    cdef void _performUpwardPass(self): #leaves to root
        if self.upward_pass == True:
            return
        edges = []
        self.initBTP()
        
        cdef int i, j
        if self.var_id_ind_map == None:
            for i in range(len(self.order)):
                if len(self.buckets[i]) == 0:
                    continue
                bucket_vars, bucket_potential = multiplyBucket(self.buckets[i])
                marg_vars, marg_potential = elimVarBucket(bucket_vars, bucket_potential, [self.variables[self.order[i]]])
                if len(marg_vars) == 0:
                    self.pe += np.log(marg_potential[0])
                    continue 
                bucket_ind = np.min([self.var_pos[marg_vars[j].id] for j in range(len(marg_vars))])
                edges.append(np.array([i, bucket_ind, len(self.buckets)], dtype=np.int32))
                func = Function()
                func.setVars(marg_vars)
                func.setPotential(marg_potential)
                self.buckets[bucket_ind].append(func)
        else:
            for i in range(len(self.order)):
                if len(self.buckets[i]) == 0:
                    continue
                bucket_vars, bucket_potential = multiplyBucket(self.buckets[i])
                marg_vars, marg_potential = elimVarBucket(bucket_vars, bucket_potential, [self.variables[self.order[i]]])
                if len(marg_vars) == 0:
                    self.pe += np.log(marg_potential[0])
                    continue 
                bucket_ind = np.min([self.var_pos[self.var_id_ind_map[marg_vars[j].id]] for j in range(len(marg_vars))])
                edges.append(np.array([i, bucket_ind, len(self.buckets)], dtype=np.int32))
                func = Function()
                func.setVars(marg_vars)
                func.setPotential(marg_potential)
                self.buckets[bucket_ind].append(func)
        edges = np.array(edges, dtype=np.int32)
        if len(edges) == 0:
            edges = np.zeros((1, 1), dtype=np.int32)
        self.messages = edges
        self.upward_pass = True 
        self.pe = np.exp(self.pe)   
        
    def performUpwardPass(self):
        self._performUpwardPass()

    cdef double _getPE(self):
        if self.upward_pass == True:
            return self.pe 
        self.performUpwardPass()
        #print("InCLT")
        #for i in range(len(self.messages)):
        #    if len(self.messages[i]) < 3:
        #        print(np.array(self.messages[i]))

        return self.pe
        
    def getPE(self):
        return self._getPE()
    
    cdef void _performDownwardPass(self): #root to leaves
        if self.downward_pass == True:
            return 
        if self.upward_pass == False:
            self._performUpwardPass()
        cdef int i, child, par, msg_ind, j, k
        cdef list msgs 
        msgs = list(self.messages)
        for i in range(len(self.messages)-1, -1, -1):
            if self.messages[i].shape[0] == 1:
                continue
            child, par, msg_ind = self.messages[i]
            parent_functions = []
            bucket = self.buckets[par]
            parent_vars = []
            sep_vars = []
            for j in range(len(bucket)):
                if j == msg_ind:
                    sep_vars.append(bucket[j].getVars())
                    continue
                parent_vars.extend(bucket[j].getVars())
                parent_functions.append(bucket[j])
            #print(len(multiplyBucket(parent_functions)))
            temp_parent_vars, temp_parent_potential = multiplyBucket(parent_functions)
            parent_vars = list(set(parent_vars))

            elim_vars = list(set(parent_vars).difference(sep_vars))
            #print(len(elimVarBucket(temp_parent_vars, temp_parent_potential, elim_vars)))
            marg_vars, marg_potential = elimVarBucket(temp_parent_vars, temp_parent_potential, elim_vars)
            func = Function()
            func.setVars(marg_vars)
            func.setPotential(marg_potential)
            msgs[i] = np.array(np.hstack([np.asarray(self.messages[i]), np.array([len(self.buckets[child])])]), dtype=np.int32)
            self.buckets[child].append(func)
        self.downward_pass = True
        self.messages = np.asarray(msgs) 
        #print("$$$$$")

    def performDownwardPass(self):
        self._performDownwardPass()

    cdef double[:, :] _getVarMarginals(self):
        if self.upward_pass == False:
            self._performUpwardPass()
        #edges = np.array(self.messages)
        #print(edges)
        if self.downward_pass == False:
            self._performDownwardPass() 
        cdef int i, nvars, j 
        nvars = len(self.buckets)
        marginals = [np.zeros(self.variables[i].d, dtype=float) for i in range(nvars)]
        for i in range(nvars):
            var = self.variables[self.order[i]]
            if var.isEvid():
                if self.var_id_ind_map == None:
                    marginals[var.id][var.val] = 1.0
                else:
                    marginals[self.var_id_ind_map[var.id]][var.val] = 1.0
            elif len(self.buckets[i]) == 0:
                if self.var_id_ind_map == None:
                    marginals[var.id] = (1.0/var.d)*np.ones(var.d)
                else:
                    marginals[var.id] = (1.0/var.d)*np.ones(var.d)
            else:
                bucket_vars, bucket_potential = multiplyBucket(self.buckets[i])
                elim_vars = [bucket_vars[j] for j in range(len(bucket_vars))]
                elim_vars.remove(var)
                marg_vars, marg_potential = elimVarBucket(bucket_vars, bucket_potential, elim_vars)
                if self.var_id_ind_map == None:
                    marginals[var.id] = marg_potential/np.sum(marg_potential)
                else:
                    marginals[self.var_id_ind_map[var.id]] = marg_potential/np.sum(marg_potential)
        cdef double[:, :] out
        out = np.asarray(list(marginals), dtype=float)
        return out
    
    def getVarMarginals(self):
        return self._getVarMarginals()

    def setEvidence(self, int id, int val):
        self.variables[id].setValue(val)

    def getVars(self):
        return self.variables

    cdef int[:, :] _generatePriorSamples(self, int n):
        cdef int i, j, nvars=len(self.variables)
        cdef int[:, :] out 
        cdef cnp.ndarray[int, ndim=2] samples = -1*np.ones((n, nvars), dtype=np.int32)
        cdef int[:] top_order
        top_order = getTopologicalOrder(self.functions, self.var_id_ind_map)
        if self.var_id_ind_map != None:
            for i in range(nvars):
                top_order[i] = self.variables[top_order[i]].id 
        for i in range(n):
            for j in range(nvars):
                if self.var_id_ind_map == None:
                    samples[i, :] = self.functions[top_order[j]].generateSample(samples[i, :], self.var_id_ind_map)
                else:
                    samples[i, :] = self.functions[self.var_id_ind_map[top_order[j]]].generateSample(samples[i, :], self.var_id_ind_map)
        out = samples
        return out

    def generatePriorSamples(self, int n):
        return self._generatePriorSamples(n)

    cdef void _generatePosteriorSamplingFunctions(self):
        if self.upward_pass == False:
            self._performUpwardPass()
        if self.downward_pass == False:
            self._performDownwardPass() 
        cdef int i, nvars, cpt_var_ind, j
        cdef cnp.ndarray[double, ndim=1] temp_prob, temp
        nvars = len(self.buckets)
        self.sampling_distributions = [[] for i in range(nvars)]
        
        for i in range(nvars):
            var = self.variables[self.order[i]]
            func = Function()
            if var.isEvid():
                continue
            elif len(self.buckets[i]) == 0:
                temp_prob = (1.0/var.d)*np.ones(var.d)
                func.setVars([var])
                func.setPotential(temp_prob)
                func.setCPTVar(0)
            else:
                bucket_vars, bucket_potential = multiplyBucket(self.buckets[i])
                for j in range(len(bucket_vars)):
                    if bucket_vars[j].id == var.id:
                        cpt_var_ind = j
                temp_prob = np.asarray(bucket_potential)/np.sum(bucket_potential)
                func.setVars(bucket_vars)
                func.setPotential(temp_prob)
                func.setCPTVar(cpt_var_ind)
            self.sampling_distributions[var.id] = func 

    cdef int[:, :] _generatePosteriorSamples(self, int n):
        if self.sampling_distributions == None:
            self._generatePosteriorSamplingFunctions()
        cdef int i, nvars, j
        cdef cnp.ndarray[int, ndim=2] samples
        cdef int[:, :] out 
        nvars = len(self.buckets)
        samples = -1*np.ones((n, nvars), dtype=np.int32)
        for i in range(n):
            for j in range(nvars-1, -1, -1):
                var = self.variables[self.order[j]]   
                if var.isEvid() == True:
                    samples[i, var.id] = var.val 
                    print(samples)
                    continue
                samples[i, :] = self.sampling_distributions[var.id].generateSample(samples[i, :], self.var_id_ind_map)
        out = samples
        return out


    def generatePosteriorSamples(self, int n):
        return self._generatePosteriorSamples(n)


