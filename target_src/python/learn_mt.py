import pygmlib
import numpy as np
import sys


if(len(sys.argv) < 4):
    print("Usage format:\npython learn_mt.py <Data Path> <Model Storage Path> <Test Likelihood File Path>")
    exit(0)



mt = pygmlib.MT()
mt.learn(sys.argv[1]+".ts.data", sys.argv[1]+".valid.data")

f = open(sys.argv[3], "a")
f.write(sys.argv[1]+" "+str(mt.log_likelihood(sys.argv[1]+'.test.data'))+'\n')
mt.write(sys.argv[2]+'.mt')
