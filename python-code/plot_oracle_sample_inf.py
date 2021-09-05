import numpy as np
import matplotlib.pyplot as plt
import sys, os

data_directory = '/Users/vasundharakomaragiri/research/TMap/results/'

#datasets = ['BN_80', 'bn2o-30-15-150-1a', 'bn2o-30-20-200-1a', 'bn2o-30-25-250-1a']
datasets = ['BN_0']#, 'BN_5', 'BN_65', '50-14-1', '75-18-1', '90-20-1', 'fs-04', 'students_03_02-0000']
#datasets = ['nltcs', 'kdd', 'baudio', 'jester', 'bnetflix', 'accidents', 'tretail', 'pumsb_star']
ns = [1000,2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
#ns = [10000,20000, 30000, 40000, 50000]
evids = ['20', '50', '80']

for dataset in datasets:
    fig, axs = plt.subplots(5, 3)#, sharex='all', sharey='all')
    #dataset = sys.argv[1]
    for i in range(len(evids)):
        ev = evids[i]
        Y = np.loadtxt(data_directory+dataset+'_'+ev+'_percent.csv', delimiter = ' ')
        for k in range(5):
            #oracle_tpm = Y[k*len(ns):(k+1)*len(ns), 2]
            oracle = np.log(Y[k*len(ns):(k+1)*len(ns), 2])
            tpm = np.log(Y[k*len(ns):(k+1)*len(ns), 3])
            lw = np.log(Y[k*len(ns):(k+1)*len(ns), 4])
            l0 = axs[k, i].plot(ns, oracle, color = 'k', label='Oracle')
            l1 = axs[k, i].plot(ns, tpm, color = 'b', label='MCN')
            l2 = axs[k, i].plot(ns, lw, color = 'g', label='LW')
            #axs[k, i].legend((l1, l2), ('MCN', 'LW'), loc='upper right', shadow=True)
            #axs[k, i].set_title(ev+"% evidence")
            axs[0, i].set_title(evids[i]+"% evidence")
            axs[k, 0].set_ylabel("evid "+str(k), rotation=90, size='large')
            
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    fig.suptitle(dataset+'', fontsize=15)
    fig.supxlabel("Number of samples")
    fig.supylabel("Weight")
    plt.savefig('/Users/vasundharakomaragiri/research/TMap/plots/'+dataset+'.png', format='png')
#plt.show()
