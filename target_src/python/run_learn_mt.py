import sys
import os
import re
import numpy as np
from subprocess import call


def write_PBS_script(path, data_dir, ds):
	program = 'learn_mt'
	prefix = ds
	scriptname =  prefix + '.sh'
	errorname =   prefix + '.err'
	stdoutname =  prefix + '.out'

	out_dir = path+'output/'
	script_dir = path+'scripts/'
	error_dir = path+'error/'
	models_dir = path+'models/MT/EM/'

	args = data_dir+ds+' '+models_dir+ds
	command = '/usr/bin/python3 '+path+program+'.py '+ args

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
	datasets = ['nltcs', 'msnbc', 'kdd', 'plants', 'baudio', 'jester', 'bnetflix', 'accidents', 'tretail', 'pumsb_star', 'dna', 'kosarek', 'msweb', 'book', 'tmovie', 'cwebkb', 'cr52', 'c20ng', 'bbc', 'ad']
	data_dir = '/nfs/experiments/data/'
	#datasets = ['nltcs', 'msnbc', 'plants', 'baudio', 'jester', 'bnetflix', 'accidents', 'tretail', 'pumsb_star', 'dna', 'BN_0', 'BN_1', 'BN_2', 'BN_3', 'BN_4', 'BN_5', 'BN_6', 'BN_7', 'BN_8', 'BN_9', 'BN_10', 'BN_11', 'BN_12', 'BN_13', 'BN_14', 'BN_15', 'BN_28', 'BN_78', 'BN_94', 'BN_96', 'BN_100', 'BN_104', 'BN_106', 'BN_108', 'BN_110', 'BN_112', 'BN_114', 'BN_116', 'BN_118', 'BN_120', 'BN_122', 'BN_124']
	#data_dir = r'/nfs/experiments/vasundhara/Proposals/FBN_Samples/'

	base_path = r'/nfs/experiments/vasundhara/Proposals/learn_mt/'
	sdir = base_path+'scripts/'
	for ds in datasets:
		script = write_PBS_script(base_path, data_dir, ds)
		script = sdir + script
		call(["qsub",script])
	                       
def main():
	runExperiments()

if __name__ == '__main__':
	main()
