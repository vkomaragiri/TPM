import GMLib
import numpy as np
import sys
'''
from FBN import FBN, Variable, Function
import pickle
model_file = '/home/vasundhara/UTD/Research/Proposals/models/FBN/BN_0.fbn'
filehandler = open(model_file, 'rb')
bn = pickle.load(filehandler)
x = np.random.choice([0, 1], size=(1, bn.dsize.shape[0]))
'''

if(len(sys.argv) < 5):
    print("Usage:\npython sampleInf.py <BN (UAI) Model Path> <MT Model Path> <Evid File Path> <P(E) Storage Root Path>")
    exit(0)


def write_mar(fhandle, mar):
    fhandle.write("MAR\n")
    fhandle.write(str(len(mar))+" ")
    for i in range(len(mar)):
        for j in range(mar[i].shape[0]):
            fhandle.write(str(mar[i][j])+" ")

ev = sys.argv[3].split('/')[-2]
resfilename = sys.argv[1].split("/")[-1]+".MAR"

ns = [1e+2, 1e+3, 1e+4, 1e+5, 1e+6]
for num_samples in ns:
    fprop = open(sys.argv[4]+"IMP_MT/"+ev+"/"+str(num_samples)+"_"+resfilename, "w")
    #flw1 = open(sys.argv[4]+"LW1/"+ev+"/"+num_samples+"_"+resfilename, "w")
    flw2 = open(sys.argv[4]+"LW2/"+ev+"/"+str(num_samples)+"_"+resfilename, "w")
    num_samples = int(num_samples)

    bn = GMLib.BN()
    mt = GMLib.MT()
    bn.read(sys.argv[1])
    mt.read(sys.argv[2])

    dsize = mt.getDomains()
    marginals = [np.zeros(dsize[i]) for i in range(dsize.shape[0])]

    evidfilename = sys.argv[3]
    evid = open(evidfilename, "r").readline().split()
    num_evid = int(evid[0])
    evid_var = [int(i) for i in evid[1::2]]
    evid_val = [int(i) for i in evid[2::2]]
    for i in range(num_evid):
        mt.setEvidence(evid_var[i], evid_val[i])

    mt.initializeBTP()

    samples = mt.getPosteriorSamples(num_samples)
    weights_den = np.exp(mt.getWeights())
    bn.readSamples(samples)
    weights_num = np.exp(bn.getWeights(0))
    weights = np.divide(weights_num, weights_den)

    for i in range(samples.shape[0]):
        for j in range(samples.shape[1]):
            marginals[j][samples[i][j]] += weights[i]

    marginals /= np.sum(weights)
    write_mar(fprop, marginals)
    #pe = np.sum(weights)/weights_num.shape[0]
    #fprop.write(str(pe)+"\n")

    for i in range(num_evid):
        bn.setEvidence(evid_var[i], evid_val[i])

    samples = bn.getPriorSamples(num_samples, 1)
    #weights_num = np.exp(bn.getWeights(0))
    #weights_den = np.exp(bn.getWeights(1))
    #weights = np.divide(weights_num, weights_den)
    #pe = np.sum(weights)/weights_num.shape[0]
    #flw1.write(str(pe)+"\n")

    marginals = [np.zeros(dsize[i]) for i in range(dsize.shape[0])]
    weights = np.exp(bn.getEvidWeights(1))
    for i in range(samples.shape[0]):
        for j in range(samples.shape[1]):
            marginals[j][samples[i][j]] += weights[i]

    marginals /= np.sum(weights)
    write_mar(flw2, marginals)
    #pe = np.sum(weights)/samples.shape[0]
    #flw2.write(str(pe)+"\n")

#fprop.close()
#flw1.close()
#flw2.close()