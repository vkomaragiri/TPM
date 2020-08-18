import sys
import os
import re
import numpy as np
from subprocess import call


def write_PBS_script(path, fbn_dir, mt_dir, evid_dir, ds, ev):
    program = 'sample_inf_fbn'
    prefix = ds+'_'+ev
    scriptname = prefix + '.sh'
    errorname = prefix + '.err'
    stdoutname = prefix + '.out'

    out_dir = path+'output/'
    script_dir = path+'scripts/'
    error_dir = path+'error/'
    res_dir = path+'pe/'+ev+'/'

    evid_dir = evid_dir+ev+'/'

    args = fbn_dir+ds+'.fbn '+mt_dir+ds+'.mt '+evid_dir+ds+'.evid '+res_dir+ds+'.pe'
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
    fbn_dir = '/nfs/experiments/vasundhara/TMaP/learn_fbn/models_nn/'
    mt_dir = '/nfs/experiments/vasundhara/TMaP/learn_mt/models/'
    evid_dir = '/nfs/experiments/vasundhara/TMaP/gen_evid_fbn/evid/'

    base_path = '/nfs/experiments/vasundhara/TMaP/sample_inf_fbn/'
    sdir = base_path+'scripts/'

    datasets = ['nltcs', 'msnbc', 'kdd', 'plants', 'baudio', 'jester', 'bnetflix', 'accidents', 'tretail', 'pumsb_star', 'dna', 'kosarek', 'msweb', 'book', 'tmovie', 'cwebkb', 'cr52', 'c20ng', 'bbc', 'ad']
    evids = ['20', '50', '80']

    for ds in datasets:
        for ev in evids:
            script = write_PBS_script(base_path, fbn_dir, mt_dir, evid_dir, ds, ev)
            script = sdir + script
            call(["qsub",script])

def main():
    runExperiments()

if __name__ == '__main__':
    main()