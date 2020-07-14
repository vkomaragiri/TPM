import pickle
import sys
from FBN import Variable, Function, FBN
import GMLib
import numpy as np

num_samples = 10

def main():
    '''
    if len(sys.argv) < 6:
        print("Usage:\npython FBNSampleInfNN.py <FBN Model Path> <MT Model Path> <Evid File Path> <PR IMP_MT Storage Path> <PR LW Storage Path>")
        exit(0)
    '''

    #datasets = ['nltcs']
    #datasets = ['msnbc', 'kdd', 'plants', 'baudio', 'jester', 'bnetflix', 'accidents', 'tretail', 'pumsb_star']
    #datasets = ['BN_0', 'BN_1', 'BN_2', 'BN_3', 'BN_4', 'BN_5', 'BN_6', 'BN_7', 'BN_8', 'BN_9', 'BN_10', 'BN_11', 'BN_12', 'BN_13', 'BN_14', 'BN_15', 'BN_28', 'BN_78', 'BN_94', 'BN_96']
    datasets = ['BN_100', 'BN_104', 'BN_106', 'BN_108', 'BN_110', 'BN_112', 'BN_114', 'BN_116', 'BN_118', 'BN_120', 'BN_122', 'BN_124']
    evids = ['20', '50', '80']

    fbn_dir = '/home/vasundhara/UTD/Research/Proposals/models/FBN/NN/'
    mt_dir = '/home/vasundhara/UTD/Research/Proposals/models/MT/EM/'
    evid_dir = '/home/vasundhara/UTD/Research/Proposals/evid/FBN/NN/'

    imp_dir = '/home/vasundhara/UTD/Research/Proposals/results/PR/FBN/NN/IMP_MT/'
    lw_dir = '/home/vasundhara/UTD/Research/Proposals/results/PR/FBN/NN/LW/'

    for ds in datasets:
        for ev in evids:
            fbn = pickle.load(open(fbn_dir+ds+'.fbn', 'rb'))
            mt = GMLib.MT()
            mt.read(mt_dir+ds+'.mt')

            evidfilename = evid_dir+ev+'/'+ds+'.evid'
            evid = open(evidfilename, "r").readline().split()
            num_evid = int(evid[0])
            evid_var = [int(i) for i in evid[1::2]]
            evid_val = [int(i) for i in evid[2::2]]
            for i in range(num_evid):
                mt.setEvidence(evid_var[i], evid_val[i])

            mt.initializeBTP()
            fimp = open(imp_dir+ev+'/'+ds+'.PR', "w")
            flw = open(lw_dir+ev+'/'+ds+'.PR', "w")
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

