#import os, sys, subprocess
'''
datasets = ['nltcs', 'msnbc', 'kdd', 'plants', 'baudio', 'jester', 'bnetflix', 'accidents', 'tretail', 'pumsb_star']
evids = ['0', '20', '50', '80']
#tasks = ['PR', 'MAR']

evid_dir = '/home/vasundhara/UTD/Research/Proposals/evid/CLT/'
for ds in datasets:
    for ev in evids:
        #for task in tasks:
            #command = ['/home/uaiuser/solvers/vec-uai14','/home/uaiuser/vasundhara/vec_check_Proposals/CLT/'+ds+'.clt','/home/uaiuser/vasundhara/vec_check_Proposals/evid/'+ev+'/'+ds+'.evid', '0', task]
        command = ['/home/vasundhara/UTD/Research/Proposals/cmake-build-debug/exactInfCLT', '-m', '/home/vasundhara/UTD/Research/Proposals/models/CLT/'+ds+'.clt', '-e', evid_dir+ev+'/'+ds+'.evid', '-o', '/home/vasundhara/UTD/Research/Proposals/results/CLT/'+ev+'/'+ds+'.MAR', '-p', '/home/vasundhara/UTD/Research/Proposals/results/CLT/'+ev+'/'+ds+'.PR']

        #command = ['/home/vasundhara/UTD/Research/Original/cmake-build-debug/exactInfCLT', '-i', '/home/vasundhara/UTD/Research/Proposals/models/CLT/'+ds+'.clt', '-e', evid_dir+ev+'/'+ds+'.evid', '-o', '/home/vasundhara/UTD/Research/Proposals/results/Original_CLT/'+ev+'/'+ds+'.MAR', '-p', '/home/vasundhara/UTD/Research/Proposals/results/Original_CLT/'+ev+'/'+ds+'.PR']

        subprocess.call(command)

'''
'''
#datasets = ['nltcs', 'msnbc', 'kdd', 'plants', 'baudio', 'jester', 'bnetflix', 'accidents', 'tretail', 'pumsb_star']
datasets = ['BN_0', 'BN_1', 'BN_2', 'BN_3', 'BN_4', 'BN_5', 'BN_6', 'BN_7', 'BN_8', 'BN_9', 'BN_10', 'BN_11', 'BN_12', 'BN_13', 'BN_14', 'BN_15', 'BN_28', 'BN_78', 'BN_94', 'BN_96', 'BN_98', 'BN_100', 'BN_102', 'BN_104', 'BN_106', 'BN_108', 'BN_110', 'BN_112', 'BN_114', 'BN_116', 'BN_118', 'BN_120', 'BN_122', 'BN_124']
evids = ['20']#, '20', '50', '80']
tasks = ['PR', 'MAR']

evid_dir = '/home/vasundhara/UTD/Research/Proposals/evid/CLT/'
for ds in datasets:
    for ev in evids:
        for task in tasks:
            command = ['/home/uaiuser/solvers/vec-uai14','/nfs/experiments/vasundhara/uai06/pe/'+ds+'.uai','/nfs/experiments/vasundhara/Proposals/evid/'+ev+'/'+ds+'.evid', '0', task]
            subprocess.call(command)
'''
import sys
import os
import re
import numpy as np
from subprocess import call


def write_PBS_script(path, ds, ev):
    prefix = 'vec_'+ds
    scriptname = prefix + '.sh'
    errorname = prefix + '.err'
    stdoutname = prefix + '.out'

    out_dir = path+'output/'
    script_dir = path+'scripts/'
    error_dir = path+'error/'

    command = path+'vec-uai14 '+'/nfs/experiments/vasundhara/uai06/pe/'+ds+'.uai '+'/nfs/experiments/vasundhara/Proposals/evid/'+ev+'/'+ds+'.evid '+'0 '+ 'PR'

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
    #datasets = ['nltcs', 'msnbc', 'kdd', 'plants', 'baudio', 'jester', 'bnetflix', 'accidents', 'tretail', 'pumsb_star', 'dna', 'kosarek', 'msweb', 'book', 'tmovie', 'cwebkb', 'cr52', 'c20ng', 'bbc', 'ad']
    datasets = ['BN_0', 'BN_1', 'BN_2', 'BN_3', 'BN_4', 'BN_5', 'BN_6', 'BN_7', 'BN_8', 'BN_9', 'BN_10', 'BN_11', 'BN_12', 'BN_13', 'BN_14', 'BN_15', 'BN_28', 'BN_78', 'BN_94', 'BN_96', 'BN_98', 'BN_100', 'BN_102', 'BN_104', 'BN_106', 'BN_108', 'BN_110', 'BN_112', 'BN_114', 'BN_116', 'BN_118', 'BN_120', 'BN_122', 'BN_124']

    base_path = r'/nfs/experiments/vasundhara/Proposals/vec/'
    evid = ['80']#, '50', '20', '0']
    for ds in datasets:
        for ev in evid:
            script = write_PBS_script(base_path, ds, ev)
            script = base_path+'scripts/' + script
            call(["qsub",script])

def main():
    runExperiments()

if __name__ == '__main__':
    main()
