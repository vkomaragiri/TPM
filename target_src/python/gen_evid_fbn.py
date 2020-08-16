from FBN import Variable, Function, FBN
import numpy as np
import os, sys, pickle


models_dir = sys.argv[1]#'/nfs/experiments/vasundhara/TMaP/learn_fbn/models_nn/'
evid_dir = sys.argv[2]#'/nfs/experiments/vasundhara/TMaP/gen_evid/NN/evid/'

evids = ['20', '50', '80']

ds = sys.argv[3]
print(ds)
bn = pickle.load(open(models_dir+ds+'.fbn', "rb"))
for ev in evids:
    f = open(evid_dir+ev+'/'+ds+'.evid', "w")
    num_evid = int(int(ev)*len(bn.variables)/100.0)
    #print(num_evid, bn.order)
    f.write(str(num_evid)+" ")
    for i in range(len(bn.order)-1, len(bn.order)-1-num_evid, -1):
        f.write(str(bn.order[i])+" ")
        val = np.random.random_integers(0, bn.variables[bn.order[i]].d-1, 1)[0]
        f.write(str(val)+" ")
    f.close()

