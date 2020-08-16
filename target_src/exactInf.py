import GMLib
import numpy as np
import sys


if(len(sys.argv) < 5):
    print("Usage:\npython exactInf.py <MT Model Path> <Evid File Path> <MAR Storage Path> <P(E) Storage Root Path>")
    exit(0)


def write_mar(fhandle, mar):
    fhandle.write("MAR\n")
    fhandle.write(str(len(mar))+" ")
    for i in range(len(mar)):
        for j in range(mar[i].shape[0]):
            fhandle.write(str(mar[i][j])+" ")

fmar = open(sys.argv[3], "w")
fpe = open(sys.argv[4], "w")


mt = GMLib.MT()
mt.read(sys.argv[1])


evidfilename = sys.argv[2]
evid = open(evidfilename, "r").readline().split()
num_evid = int(evid[0])
evid_var = [int(i) for i in evid[1::2]]
evid_val = [int(i) for i in evid[2::2]]
for i in range(num_evid):
    mt.setEvidence(evid_var[i], evid_val[i])

mt.initializeBTP()
pe = mt.getPE()
marginals = mt.getVarMarginals()
fpe.write(str(np.log10(pe))+"\n")
write_mar(fmar, marginals)

fpe.close()
fmar.close()