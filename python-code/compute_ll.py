import numpy as np
import pygmlib
import sys

f = open('/Users/vasundharakomaragiri/research/TMap/results/ll_data_modified.csv', "w")
'''
datasets = ['nltcs', 'kdd', 'baudio', 'jester', 'bnetflix', 'accidents', 'tretail', 'pumsb_star']

ll_scores = []
f.write('Dataset,Oracle,BCN\n')
data_directory = '/Users/vasundharakomaragiri/research/TMap/data/'
models_directory = '/Users/vasundharakomaragiri/research/TMap/models/'
for ds in datasets:
    f.write(ds+',')
    bn = pygmlib.BN_UAI()   
    bn.read(models_directory+ds+'.uai')
    bn_ll = bn.log_likelihood(data_directory+ds+'.test.data')
    f.write(str(bn_ll)+',')
    mcn = pygmlib.MCN()
    mcn.read(models_directory+ds+'-50.mcn')
    mcn_ll = mcn.log_likelihood(data_directory+ds+'.test.data')
    f.write(str(mcn_ll)+'\n')


datasets = ['BN_0', 'BN_5', 'BN_65', '50-14-1', '75-18-1', '90-20-1', 'fs-04', 'students_03_02-0000']
'''
datasets = ['BN_80', 'bn2o-30-15-150-1a', 'bn2o-30-20-200-1a', 'bn2o-30-25-250-1a']
ll_scores = []
f.write('Dataset,Oracle,BCN\n')
data_directory = '/Users/vasundharakomaragiri/research/TMap/data/'
models_directory = '/Users/vasundharakomaragiri/research/TMap/models/'


data_directory = '/Users/vasundharakomaragiri/research/TMap.12.2020/TMap/data/'
for ds in datasets:
    f.write(ds+',')
    bn = pygmlib.BN_UAI()   
    bn.read(models_directory+ds+'.uai')
    bn_ll = bn.log_likelihood(data_directory+ds+'.test.data')
    f.write(str(bn_ll)+',')
    mcn = pygmlib.MCN()
    mcn.read(models_directory+ds+'-oracle-uai-50.mcn')
    mcn_ll = mcn.log_likelihood(data_directory+ds+'.test.data')
    f.write(str(mcn_ll)+'\n')

f.close()