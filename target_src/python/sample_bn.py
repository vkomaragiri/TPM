import GMLib
import numpy as np
import sys

if(len(sys.argv) < 4):
    print("Usage Format:\npython sample_bn.py <Model File Path> <Number of samples> <Samples Storage Path>")

bn = GMLib.BN()
bn.read(sys.argv[1])
samples = bn.getPriorSamples(int(sys.argv[2]), 0)
bn.writeSamples(sys.argv[3])

