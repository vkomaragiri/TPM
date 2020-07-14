from FBN import Variable, Function, FBN
import numpy as np
import os, sys, pickle


models_dir = '/home/vasundhara/UTD/Research/Proposals/models/FBN/NN/'
#models_dir = '/home/vasundhara/UTD/Research/Proposals/target_src/'
evid_dir = '/home/vasundhara/UTD/Research/Proposals/evid/FBN/NN/'

#datasets = ['nltcs', 'msnbc', 'kdd', 'plants', 'baudio', 'jester', 'bnetflix', 'accidents', 'tretail', 'pumsb_star']#, 'dna', 'kosarek', 'msweb', 'book', 'tmovie', 'cwebkb', 'cr52', 'c20ng', 'bbc', 'ad']
#datasets = ['grid', 'grid_extreme']

evids = ['20', '50', '80']

for fname in os.listdir(models_dir):
    ds = fname.split('.')[0]
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

