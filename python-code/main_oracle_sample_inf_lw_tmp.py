import numpy as np
from scipy.special import logsumexp
import os, sys

if(len(sys.argv) < 1):
    print("Usage format: python main_oracle_sample_inf_lw_tmp.py <dataset> <evid_percent>")
    sys.exit(0)

#datasets = ['BN_80', 'bn2o-30-15-150-1a', 'bn2o-30-20-200-1a', 'bn2o-30-25-250-1a']
datasets = ['BN_0']#, 'BN_5', 'BN_65', '50-14-1', '75-18-1', '90-20-1', 'fs-04', 'students_03_02-0000']
#datasets = ['nltcs', 'kdd', 'baudio', 'jester', 'bnetflix', 'accidents', 'tretail', 'pumsb_star']
evids = ['20', '50', '80']

for dataset in datasets:
    for evid_percent in evids:
        #dataset = sys.argv[1]
        #evid_percent = sys.argv[2]

        data_directory = '/Users/vasundharakomaragiri/research/TMap/results/'

        results = []
        for k in range(5):
            #index_wts = -np.ones(500000)/np.loadtxt(fname=data_directory+dataset+'_'+evid_percent+'_percent_'+str(k)+'.wts', delimiter=',', usecols=np.arange(500000))
            #index_wts /= np.sum(index_wts)
            
            #oracle_tpm_estimate = np.loadtxt(fname=data_directory+dataset+'_'+evid_percent+'_percent_'+str(k)+'.tpm', delimiter=',', usecols=np.arange(100000))
            mcn_estimate = np.loadtxt(fname=data_directory+dataset+'_'+evid_percent+'_percent_'+str(k)+'.mcn.wt', delimiter=',', usecols=np.arange(100000))
            lw_estimate = np.loadtxt(fname=data_directory+dataset+'_'+evid_percent+'_percent_'+str(k)+'.lw.wt', delimiter=',', usecols=np.arange(100000))
            oracle_estimate = np.loadtxt(fname=data_directory+dataset+'_'+evid_percent+'_percent_'+str(k)+'.bn.wt', delimiter=',', usecols=np.arange(100000))
            #for nsamples in [10000,20000, 30000, 40000, 50000]:
            for nsamples in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
            #for nsamples in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            
                #tpm_log_wt = 0.0
                #lw_log_wt = 0.0
                #for j in range(5):
                indices = []
                indices = np.random.choice(a=np.arange(100000), size=nsamples, replace=False)#, p=index_wts)
                #oracle_mcn_log_wt = np.exp(logsumexp(oracle_mcn_estimate[indices])-np.log(nsamples))
                mcn_log_wt = np.exp(logsumexp(mcn_estimate[indices])-np.log(nsamples))
                lw_log_wt = np.exp(logsumexp(lw_estimate[indices])-np.log(nsamples))
                oracle_log_wt = np.exp(logsumexp(oracle_estimate[indices])-np.log(nsamples))
            
                #tpm_log_wt /= 5.0
                #lw_log_wt /= 5.0
                results.append([k, nsamples, oracle_log_wt, mcn_log_wt, lw_log_wt])

        np.savetxt(X = np.array(results), fname='/Users/vasundharakomaragiri/research/TMap/results/'+dataset+'_'+evid_percent+'_percent.csv')
