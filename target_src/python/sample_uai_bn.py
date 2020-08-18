import pygmlib
import numpy as np
import sys

if(len(sys.argv) < 4):
    print("Usage Format:\npython sample_bn.py <Model File Path> <Number of samples> <Samples Storage Path>")
    sys.exit(0)

bn = pygmlib.BN_UAI()
bn.read(sys.argv[1])
samples = bn.generateSamples(int(sys.argv[2]))
bn.writeSamples(sys.argv[3])

