import numpy as np
from scipy.special import logsumexp
import os, sys

if(len(sys.argv) < 1):
    print("Usage format: python main_oracle_sample_inf_lw_tmp.py <dataset> <evid_percent>")
    sys.exit(0)

data_directory = '/Users/vasundharakomaragiri/research/TMap/results/'

datasets = ['BN_80', 'bn2o-30-15-150-1a', 'bn2o-30-20-200-1a', 'bn2o-30-25-250-1a']
#datasets = ['BN_0', 'BN_5', 'BN_65', '50-14-1', '75-18-1', '90-20-1', 'fs-04', 'students_03_02-0000', 'nltcs', 'kdd', 'baudio', 'jester', 'bnetflix', 'accidents', 'tretail', 'pumsb_star']
evids = ['20', '50', '80']

for dataset in datasets:
    for evid_percent in evids:
        results = []
        for nsamples in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
            mcn_log_wt = 0.0
            lw_log_wt = 0.0
            oracle_log_wt = 0.0
            #data_mcn_log_wt = 0.0
            for k in range(5):
                mcn_estimate = np.loadtxt(fname=data_directory+dataset+'_'+evid_percent+'_percent_'+str(k)+'_modified.mcn.wt', delimiter=',', usecols=np.arange(100000))
                lw_estimate = np.loadtxt(fname=data_directory+dataset+'_'+evid_percent+'_percent_'+str(k)+'_modified.lw.wt', delimiter=',', usecols=np.arange(100000))
                oracle_estimate = np.loadtxt(fname=data_directory+dataset+'_'+evid_percent+'_percent_'+str(k)+'_modified.bn.wt', delimiter=',', usecols=np.arange(100000))
                #data_mcn_estimate = np.loadtxt(fname=data_directory+dataset+'_'+evid_percent+'_percent_'+str(k)+'.data_mcn.wt', delimiter=',', usecols=np.arange(100000))
                #for j in range(5):
                indices = np.random.choice(a=np.arange(100000), size=nsamples, replace=False)#, p=index_wts)
            
                mcn_log_wt += np.exp(logsumexp(mcn_estimate[indices])-np.log(nsamples))
                lw_log_wt += np.exp(logsumexp(lw_estimate[indices])-np.log(nsamples))
                oracle_log_wt += np.exp(logsumexp(oracle_estimate[indices])-np.log(nsamples))
                #data_mcn_log_wt += np.exp(logsumexp(data_mcn_estimate[indices])-np.log(nsamples))
            
            mcn_log_wt /= 5.0
            lw_log_wt /= 5.0
            oracle_log_wt /= 5.0
            #data_mcn_log_wt /= 5.0
            results.append([nsamples, oracle_log_wt, mcn_log_wt, lw_log_wt])
            #results.append([nsamples, oracle_log_wt, mcn_log_wt, lw_log_wt, data_mcn_log_wt])

        np.savetxt(X = np.array(results), fname='/Users/vasundharakomaragiri/research/TMap/results/'+dataset+'_'+evid_percent+'_percent_modified.2.csv')


for dataset in datasets:
    for evid_percent in evids:
        results = []
        for k in range(5):            
            mcn_estimate = np.loadtxt(fname=data_directory+dataset+'_'+evid_percent+'_percent_'+str(k)+'_modified.mcn.wt', delimiter=',', usecols=np.arange(100000))
            lw_estimate = np.loadtxt(fname=data_directory+dataset+'_'+evid_percent+'_percent_'+str(k)+'_modified.lw.wt', delimiter=',', usecols=np.arange(100000))
            oracle_estimate = np.loadtxt(fname=data_directory+dataset+'_'+evid_percent+'_percent_'+str(k)+'_modified.bn.wt', delimiter=',', usecols=np.arange(100000))
            #data_mcn_estimate = np.loadtxt(fname=data_directory+dataset+'_'+evid_percent+'_percent_'+str(k)+'.data_mcn.wt', delimiter=',', usecols=np.arange(100000))
            for nsamples in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
                indices = np.random.choice(a=np.arange(100000), size=nsamples, replace=False)#, p=index_wts)

                mcn_log_wt = np.exp(logsumexp(mcn_estimate[indices])-np.log(nsamples))
                lw_log_wt = np.exp(logsumexp(lw_estimate[indices])-np.log(nsamples))
                oracle_log_wt = np.exp(logsumexp(oracle_estimate[indices])-np.log(nsamples))
                #data_mcn_log_wt = np.exp(logsumexp(data_mcn_estimate[indices])-np.log(nsamples))

                results.append([k, nsamples, oracle_log_wt, mcn_log_wt, lw_log_wt])
                #results.append([k, nsamples, oracle_log_wt, mcn_log_wt, lw_log_wt, data_mcn_log_wt])
        np.savetxt(X = np.array(results), fname='/Users/vasundharakomaragiri/research/TMap/results/'+dataset+'_'+evid_percent+'_percent_modified.csv')
