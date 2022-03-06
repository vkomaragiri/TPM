import numpy as np
from BN import BN
from MT import MT
from CN import CN
from MCN import MCN

import time

import pygmlib

dname = "nltcs"

train = np.loadtxt("../data/"+dname+".ts.data", delimiter=',', dtype=np.int32)
valid = np.loadtxt("../data/"+dname+".valid.data", delimiter=',', dtype=np.int32)
test = np.loadtxt("../data/"+dname+".test.data", delimiter=',', dtype=np.int32)
'''
start_time = time.time()
clt = BN()
clt.learnCLT(train)
clt.write("/Users/vasundhara/research/TPM/cython-code/models/"+dname+".clt")
clt2 = BN()
clt2.read("/Users/vasundhara/research/TPM/cython-code/models/"+dname+".clt")
clt2.write("/Users/vasundhara/research/TPM/cython-code/models/"+dname+".clt2")
cython_time = time.time()
ll_score = clt.getLLScore(test)
print("Log-likelhood score =", ll_score)
print("Cython code: %s seconds" % ( cython_time- start_time))
start_time = time.time()
bn = pygmlib.BN_UAI()
bn.learn("../data/"+dname+".ts.data")
cpp_time = time.time()
ll_Score = bn.log_likelihood("../data/"+dname+".test.data")
print("Log-likelhood score =", ll_Score)
print("C++ code: %s seconds" % ( cpp_time- start_time))


start_time = time.time()
mt = MT()
mt.learnEM(train, 10, 10)
mt.write("/Users/vasundhara/research/TPM/cython-code/models/"+dname+".mt")
mt2 = MT()
mt2.read("/Users/vasundhara/research/TPM/cython-code/models/"+dname+".mt")
mt2.write("/Users/vasundhara/research/TPM/cython-code/models/"+dname+".mt2")
cython_time = time.time()
ll_score = mt.getLLScore(test)
print("Log-likelhood score =", ll_score)
print("Cython code: %s seconds" % ( cython_time- start_time))
start_time = time.time()
mtcpp = pygmlib.MT()
mtcpp.learn("../data/"+dname+".ts.data", "../data/"+dname+".valid.data")
cpp_time = time.time()
ll_Score = mtcpp.log_likelihood("../data/"+dname+".test.data")
print("Log-likelhood score =", ll_Score)
print("C++ code: %s seconds" % ( cpp_time- start_time))

start_time = time.time()
cn = CN()
cn.learn(train=train, valid=valid, prune=True, max_depth=10)
cn.write("/Users/vasundhara/research/TPM/cython-code/models/"+dname+".cn")
cn2 = CN()
cn2.read("/Users/vasundhara/research/TPM/cython-code/models/"+dname+".cn")
cn2.write("/Users/vasundhara/research/TPM/cython-code/models/"+dname+".cn2")
cython_time = time.time()
ll_score = cn.getLLScore(test)
print("Log-likelhood score =", ll_score)
print("Cython code: %s seconds" % ( cython_time- start_time))
start_time = time.time()
cncpp = pygmlib.CN()
cncpp.learn("../data/"+dname+".ts.data", "../data/"+dname+".valid.data")
cpp_time = time.time()
ll_Score = cncpp.log_likelihood("../data/"+dname+".test.data")
print("Log-likelhood score =", ll_Score)
print("C++ code: %s seconds" % ( cpp_time- start_time))



start_time = time.time()
mcn = MCN()
mcn.learnEM(train, valid, 10, 100)
mcn.write("/Users/vasundhara/research/TPM/cython-code/models/"+dname+".mcn")
mcn2 = MCN()
mcn2.read("/Users/vasundhara/research/TPM/cython-code/models/"+dname+".mcn")
mcn2.write("/Users/vasundhara/research/TPM/cython-code/models/"+dname+".mcn2")
cython_time = time.time()
ll_score = mcn.getLLScore(test)
print("Log-likelhood score =", ll_score)
ll_score = mcn2.getLLScore(test)
print("Log-likelhood score =", ll_score)
print("Cython code: %s seconds" % ( cython_time- start_time))
'''

'''
clt = BN()
clt.read("/Users/vasundhara/research/TPM/cython-code/models/"+dname+".clt")
clt.setEvidence(0, 0)
clt.setEvidence(5, 0)
clt.setEvidence(6, 1)
pe = clt.getPE()
print("Probability of evidence:", pe)
marginals = np.asarray(clt.getVarMarginals())
print(marginals)


mt = MT()
mt.read("/Users/vasundhara/research/TPM/cython-code/models/"+dname+".mt")
mt.setEvidence(0, 0)
mt.setEvidence(5, 0)
mt.setEvidence(6, 1)
pe = mt.getPE()
print("Probability of evidence:", pe)
marginals = np.asarray(mt.getVarMarginals())
print(marginals)
'''

cn = CN()
#cn.learn(train=train, valid=valid, prune=True, max_depth=10)
#cn.write("/Users/vasundhara/research/TPM/cython-code/models/"+dname+".cn")
cn.read("/Users/vasundhara/research/TPM/cython-code/models/"+dname+".cn")
cn.setEvidence(0, 0)
cn.setEvidence(5, 0)
cn.setEvidence(6, 1)
pe = cn.getPE()
print("Probability of evidence:", pe)
print(cn.getVarMarginals())


'''
mcn = MCN()
mcn.read("/Users/vasundhara/research/TPM/cython-code/models/"+dname+".mcn")
pe = mcn.getPE()
print("Probability of evidence:", pe)
marginals = np.asarray(mcn.getVarMarginals())
print(marginals)
'''