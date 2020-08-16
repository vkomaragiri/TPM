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


def write_PBS_script(path, bn_dir, mt_dir, ds, ev_dir, ev, results_dir, dist_dir):
    program = 'sampleInf'
    prefix = ds+'_'+ev
    scriptname = prefix + '.sh'
    errorname = prefix + '.err'
    stdoutname = prefix + '.out'

    out_dir = path+'output/'
    script_dir = path+'scripts/'
    error_dir = path+'error/'
    dsname = ds.split('.')[0]+'.evid'
    args = bn_dir+ds+'.uai '+mt_dir+ds+'.mt '+ev_dir+ev+"/"+dsname+' '+results_dir
    command = dist_dir+'sampleInf/'+program +' '+ args

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
    datasets = ['BN_0', 'BN_1', 'BN_2', 'BN_3', 'BN_4', 'BN_5', 'BN_6', 'BN_7', 'BN_8', 'BN_9', 'BN_10', 'BN_11', 'BN_12', 'BN_13', 'BN_14', 'BN_15', 'BN_28', 'BN_78', 'BN_94', 'BN_96', 'BN_98', 'BN_100', 'BN_102', 'BN_104', 'BN_106', 'BN_108', 'BN_110', 'BN_112', 'BN_114', 'BN_116', 'BN_118', 'BN_120', 'BN_122', 'BN_124']

    bn_dir = r'/nfs/experiments/vasundhara/uai06/pe/'
    mt_dir = r'/nfs/experiments/vasundhara/Proposals/learn_mt/models/MT/EM/'
    evid_dir = r'/nfs/experiments/vasundhara/Proposals/evid/'
    results_dir = r'/nfs/experiments/vasundhara/Proposals/results/MAR/'
    dist_dir = r'/nfs/experiments/vasundhara/Proposals/dist/'

    evids = ['20', '50', '80']

    base_path = r'/nfs/experiments/vasundhara/Proposals/sample_inf_mar/'
    sdir = base_path+'scripts/'
    for ds in datasets:
        for ev in evids:
            script = write_PBS_script(base_path, bn_dir, mt_dir, ds, evid_dir,ev, results_dir, dist_dir)
            script = sdir + script
            call(["qsub",script])

def main():
    runExperiments()

if __name__ == '__main__':
    main()
