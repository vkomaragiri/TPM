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


def write_PBS_script(path, models_dir, ds):
    program = 'gen_evid_fbn'
    prefix = ds
    scriptname = prefix + '.sh'
    errorname = prefix + '.err'
    stdoutname = prefix + '.out'

    out_dir = path+'output/'
    script_dir = path+'scripts/'
    error_dir = path+'error/'
    evid_dir = path+'evid/'

    args = models_dir+' '+evid_dir+' '+ds
    command = '/usr/bin/python3 '+path+program+'.py'+' '+ args

    scriptfile = open(script_dir + scriptname,'w')
    scriptfile.write('#!/bin/bash\n')
    scriptfile.write('# SGE Options\n')
    scriptfile.write('#$ -l h_rt=100:00:00\n')
    scriptfile.write('#$ -e ' + error_dir+errorname+' \n')
    scriptfile.write('#$ -o ' + out_dir +stdoutname+' \n')
    scriptfile.write('#$ -l  hostname=uai0[6789]*|uai1[0123456789]*|uai2[0123456789]*|uai30'+' \n')
    scriptfile.write('export PATH=/home/uaiuser/miniconda2/bin:$PATH' + '\n')
    scriptfile.write('export PATH=/usr/lib/' + '\n')

    scriptfile.write(command)
    print(command)
    scriptfile.close()

    # make script executable
    os.system('chmod 0777 ' + script_dir +'' + scriptname)
    return (scriptname)

def runExperiments():
    models_dir = '/nfs/experiments/vasundhara/TMaP/learn_fbn/models_nn/'
    base_path = '/nfs/experiments/vasundhara/TMaP/gen_evid_fbn/'


    sdir = base_path+'scripts/'
    for fname in os.listdir(models_dir):
        ds = fname.split('.')[0]
    #for ds in ['nltcs']:
        script = write_PBS_script(base_path, models_dir, ds)
        script = sdir + script
        call(["qsub",script])

def main():
    runExperiments()

if __name__ == '__main__':
    main()