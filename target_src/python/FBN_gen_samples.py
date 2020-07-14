import pickle
import sys
from FBN import Variable, Function, FBN
import GMLib
import numpy as np

num_samples = 10

def main():
    '''
    if len(sys.argv) < 3:
        print("Usage:\npython FBN_gen_samples.py <FBN Model Path> <Samples Storage Path>")
    '''
    samples_dir = '../FBN_Samples/'
    models_dir = '../models/FBN/'
    datasets = ['nltcs', 'msnbc', 'plants', 'baudio', 'jester', 'bnetflix', 'accidents', 'tretail', 'pumsb_star', 'dna', 'BN_0', 'BN_1', 'BN_2', 'BN_3', 'BN_4', 'BN_5', 'BN_6', 'BN_7', 'BN_8', 'BN_9', 'BN_10', 'BN_11', 'BN_12', 'BN_13', 'BN_14', 'BN_15', 'BN_28', 'BN_78', 'BN_94', 'BN_96', 'BN_100', 'BN_104', 'BN_106', 'BN_108', 'BN_110', 'BN_112', 'BN_114', 'BN_116', 'BN_118', 'BN_120', 'BN_122', 'BN_124']
    for ds in datasets:
        print(ds)
        fbn = pickle.load(open(models_dir+ds+'.fbn', 'rb'))
        samples, weights = fbn.generateSamples(100000)
        np.savetxt(samples_dir+ds+".ts.data", X = samples[:64000, :], fmt = '%d', delimiter=',')

        #samples, weights = fbn.generateSamples(16000)
        np.savetxt(samples_dir+ds+".valid.data", X = samples[64000:80000, :], fmt = '%d', delimiter=',')

        #samples, weights = fbn.generateSamples(20000)
        np.savetxt(samples_dir+ds+".test.data", X = samples[80000:, :], fmt = '%d', delimiter=',')
if __name__ == '__main__':
    main()

