import pickle
import sys
from FBN import Variable, Function, FBN
import GMLib
import numpy as np

num_samples = 10

def main():
    if len(sys.argv) < 6:
        print("Usage:\npython FBNSampleInf.py <FBN Model Path> <MT Model Path> <Evid File Path> <PR IMP_MT Storage Path> <PR LW Storage Path>")
        exit(0)
    fbn = pickle.load(open(sys.argv[1], 'rb'))
    mt = GMLib.MT()
    mt.read(sys.argv[2])

    evidfilename = sys.argv[3]
    evid = open(evidfilename, "r").readline().split()
    num_evid = int(evid[0])
    evid_var = [int(i) for i in evid[1::2]]
    evid_val = [int(i) for i in evid[2::2]]
    for i in range(num_evid):
        mt.setEvidence(evid_var[i], evid_val[i])

    mt.initializeBTP()
    fimp = open(sys.argv[4], "w")
    flw = open(sys.argv[5], "w")
    ns = [1e2, 1e3, 1e4, 1e5, 1e6]
    for num_samples in ns:
        num_samples = int(num_samples)
        samples = mt.getPosteriorSamples(num_samples)
        weights_den = np.exp(mt.getWeights())
        weights_num = fbn.getProbability(samples)
        #weights_num = np.exp(bn.getWeights(0))
        weights = np.divide(weights_num, weights_den)
        pe = np.sum(weights)/num_samples
        fimp.write(str(np.log10(pe))+"\n")

        for i in range(num_evid):
            fbn.setValue(evid_var[i], evid_val[i])

        samples, weights = fbn.generateSamples(num_samples)
        pe = np.sum(weights)/num_samples
        flw.write(str(np.log10(pe))+"\n")
    fimp.close()
    flw.close()

if __name__ == '__main__':
    main()

