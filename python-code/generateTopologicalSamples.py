import pygmlib
import sys
import numpy as np


if len(sys.argv) < 2:
    print("Usage: python generateTopologicalSamples.py <dataset_name>")
    sys.exit(0)


uai_file_path = '/Users/vasundharakomaragiri/research/TMap.12.2020/TMap/data/'+sys.argv[1]+'.uai'
dataset_file_path = '/Users/vasundharakomaragiri/research/TMap.12.2020/TMap/data/oracle-uai/'+sys.argv[1]

num_samples = [10000, 10000, 10]
suffixes = ['.ts.data', '.valid.data', '.test.data']

for i in range(len(num_samples)):
    bn = pygmlib.BN_UAI()
    bn.read(uai_file_path)
    samples = bn.generatePriorSamples(num_samples[i])
    dname = dataset_file_path+suffixes[i]
    np.savetxt(dname, samples, delimiter=',',fmt='%d')