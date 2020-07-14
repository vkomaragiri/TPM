import pygmlib
import numpy as np
import sys


if(len(sys.argv) < 3):
    print("Usage format:\npython learn_mt.py <Data Path> <Model Storage Path>")
    exit(0)



mt = pygmlib.MT()
mt.learn(sys.argv[1]+".ts.data", sys.argv[1]+".valid.data")

print(mt.log_likelihood(sys.argv[1]+'.test.data'))

mt.write(sys.argv[2]+'.mt')
