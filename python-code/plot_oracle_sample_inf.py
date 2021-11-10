import numpy as np
import matplotlib.pyplot as plt
import sys, os
import matplotlib


def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim




data_directory = '/Users/vasundharakomaragiri/research/TMap/results/'

datasets = ['BN_80', 'bn2o-30-15-150-1a', 'bn2o-30-20-200-1a', 'bn2o-30-25-250-1a']
#datasets = ['BN_0', 'BN_5', 'BN_65', '50-14-1', '75-18-1', '90-20-1', 'fs-04', 'students_03_02-0000', 'nltcs', 'kdd', 'baudio', 'jester', 'bnetflix', 'accidents', 'tretail', 'pumsb_star']
ns = [1000,2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
evids = ['20', '50', '80']

for dataset in datasets:
    fig, axs = plt.subplots(5, 3)#, figsize=set_size(textwidth, fraction=0.4))
    #dataset = sys.argv[1]
    for i in range(len(evids)):
        ev = evids[i]
        Y = np.loadtxt(data_directory+dataset+'_'+ev+'_percent_modified.csv', delimiter = ' ')
        for k in range(5):
            oracle = np.log(Y[k*len(ns):(k+1)*len(ns), 2])
            mcn = np.log(Y[k*len(ns):(k+1)*len(ns), 3])
            lw = np.log(Y[k*len(ns):(k+1)*len(ns), 4])
            #data_mcn = np.log(Y[k*len(ns):(k+1)*len(ns), 5])
            #l0 = axs[k, i].plot(ns, oracle, color = 'k', label='Oracle')
            l1 = axs[k, i].plot(ns, mcn, color = 'b', label='MCN')
            l2 = axs[k, i].plot(ns, lw, color = 'g', label='LW')
            #l3 = axs[k, i].plot(ns, data_mcn, color = 'y', label='Data-MCN')
            axs[0, i].set_title(evids[i]+"% evidence")
            axs[k, 0].set_ylabel("evid "+str(k), rotation=90)
            
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    fig.suptitle(dataset+'_modified')
    fig.supxlabel("Number of samples")
    fig.supylabel("Weight")
    plt.savefig('/Users/vasundharakomaragiri/research/TMap/plots/'+dataset+'_modified.png', format='png')

#plt.show()
textwidth = 238
figdim = set_size(textwidth, fraction=1)
print(figdim)

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

ns = [1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3, 1e4]

for dataset in datasets:
    fig, axs = plt.subplots(1, 3, figsize=figdim)#, sharex='all', sharey='all')
    fig.tight_layout(rect=[0.07, 0.1, 0.9, 0.9])

    #dataset = sys.argv[1]
    for i in range(len(evids)):
        ev = evids[i]
        Y = np.loadtxt(data_directory+dataset+'_'+ev+'_percent_modified.2.csv', delimiter = ' ')
        #for k in range(5):
        #oracle_tpm = Y[k*len(ns):(k+1)*len(ns), 2]
        oracle = np.log(Y[:, 1])
        mcn = np.log(Y[:, 2])
        lw = np.log(Y[:, 3])
        #data_mcn = np.log(Y[:, 4])
        #l0 = axs[i].plot(ns, oracle, color = 'y', label='Oracle')
        l1 = axs[i].plot(ns, mcn, color = 'b', label='BCN')
        l2 = axs[i].plot(ns, lw, color = 'g', label='LW')
        #l3 = axs[i].plot(ns, data_mcn, color = 'y', label='Data-MCN')
        axs[i].set_title(evids[i]+"%")#, fontsize = 10)
        axs[i].ticklabel_format(scilimits= (0, 3))

            
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 6})

    fig.suptitle(dataset+'_modified')
    fig.supxlabel("Number of samples")
    fig.supylabel("Log-likelihood")#, fontsize = 10)
    plt.savefig('/Users/vasundharakomaragiri/research/TMap/plots/avg_kld/'+dataset+'_modified.png', format='png')
    plt.savefig('/Users/vasundharakomaragiri/research/TMap/plots/avg_kld/'+dataset+'_modified.pgf', format='pgf')

#plt.show()
