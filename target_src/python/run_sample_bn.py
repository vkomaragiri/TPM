#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Shasha Jin
#
# Created:     03/25/2018
# Copyright:   (c) Shasha 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import sys
import os
import re
import numpy as np
from subprocess import call


def write_PBS_script(path, data_dir, ds, n, dtype):
    program = 'sample_bn'
    prefix = ds
    scriptname =  prefix + '.sh'
    errorname =   prefix + '.err'
    stdoutname =  prefix + '.out'

    out_dir = path+'output/'
    script_dir = path+'scripts/'
    error_dir = path+'error/'
    samples_dir = path+'samples/'

    args = data_dir+ds+'.uai '+n+' '+samples_dir+ds+dtype+'.data'
    command = path+'dist/sample_bn/'+program +' '+ args

    scriptfile = open(script_dir + scriptname,'w')
    scriptfile.write('#!/bin/bash\n')
    scriptfile.write('# SGE Options\n')
    scriptfile.write('#$ -l h_rt=100:00:00\n')
    scriptfile.write('#$ -e ' + error_dir+errorname+' \n')
    scriptfile.write('#$ -o ' + out_dir +stdoutname+' \n')
    scriptfile.write('#$ -l  hostname=uai0[123456789]*|uai1[0123456789]*|uai2[0123456789]*|uai30'+' \n')
    scriptfile.write('export PATH=/home/uaiuser/miniconda2/bin:$PATH' + '\n')
    scriptfile.write('export PATH=/usr/lib/' + '\n')

    scriptfile.write(command)
    print(command)
    scriptfile.close()

    # make script executable
    os.system('chmod 0777 ' + script_dir +'' + scriptname)
    return (scriptname)

def runExperiments():
    #datasets = ['nltcs', 'msnbc', 'kdd', 'plants', 'baudio', 'jester', 'bnetflix', 'accidents', 'tretail', 'pumsb_star', 'dna', 'kosarek', 'msweb', 'book', 'tmovie']
    #datasets = ['cwebkb', 'cr52', 'c20ng', 'bbc', 'ad']
    datasets = ['BN_14', 'BN_112', 'BN_108', 'BN_112', 'BN_100', 'BN_9', 'BN_0', 'BN_10', 'BN_78', 'BN_108', 'BN_12', 'BN_104', 'BN_4', 'BN_102', 'BN_1', 'BN_1', 'BN_94', 'BN_120', 'BN_104', 'BN_13', 'BN_28', 'BN_8', 'BN_2', 'BN_7', 'BN_0', 'BN_12', 'BN_124', 'BN_120', 'BN_94', 'BN_100', 'BN_98', 'BN_11', 'BN_114', 'BN_116', 'BN_102', 'BN_106', 'BN_5', 'BN_5', 'BN_15', 'BN_10', 'BN_7', 'BN_118', 'BN_6', 'BN_122', 'BN_13', 'BN_114', 'BN_96', 'BN_116', 'BN_106', 'BN_2', 'BN_8', 'BN_15', 'BN_110', 'BN_6', 'BN_3', 'BN_11', 'BN_4', 'BN_9', 'BN_110', 'BN_14', 'BN_3', 'BN_96', 'BN_122', 'BN_118', 'BN_28', 'BN_98', 'BN_78', 'BN_124']
    data_dir = r'/nfs/experiments/vasundhara/uai06/pe/'

    num_samples = ['64000', '16000', '20000']
    dtypes = ['.ts', '.valid', '.test']

    base_path = r'/nfs/experiments/vasundhara/Proposals/sample_bn/'
    sdir = base_path+'scripts/'
    for ds in datasets:
        for i in range(len(num_samples)):
            script = write_PBS_script(base_path, data_dir, ds, num_samples[i], dtypes[i])
            script = sdir + script
            call(["qsub",script])

def main():
    runExperiments()

if __name__ == '__main__':
    main()
