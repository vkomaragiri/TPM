from FBN import FBN, Variable, Function
import pygmlib
import pickle
import sys, os, copy
import numpy as np

fbn = pickle.load(open(sys.argv[1], 'rb'))
mt = pygmlib.MT()
mt.read(sys.argv[2])

evid = open(sys.argv[3], "r").readline().split()[1:]

res = open(sys.argv[4], "w")

evid_var = []
evid_val = []
for i in range(0, len(evid), 2):
    evid_var.append(int(evid[i]))
    evid_val.append(int(evid[i+1]))
    mt.setEvidence(int(evid[i]), int(evid[i+1]))

exact_mt = np.log10(mt.getPE())

max_samples = int(1e6)
num_samples = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]#, 1e7]
imp_mt = []
imp_lw = []

mt_posterior_samples = mt.generateSamples(max_samples)
for n in num_samples:
    n = int(n)
    num = fbn.getProbability(mt_posterior_samples[:n])
    den = mt.generateSampleWeights(n)
    weight = np.log10(np.sum(np.divide(num, den))/n)
    imp_mt.append(weight)


for i in range(len(evid_var)):
    fbn.setValue(evid_var[i], evid_val[i])

fbn_lw_samples, fbn_lw_weights = fbn.generateSamples(max_samples)
for i in range(len(num_samples)):
    n = int(num_samples[i])
    imp_lw.append(np.log10(np.sum(fbn_lw_weights[:n])/n))

    res.write(str(n)+','+str(exact_mt)+','+str(imp_mt[i])+','+str(imp_lw[i])+'\n')

res.close()