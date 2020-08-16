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
	models_dir = path+'models/'

	args = data_dir+ds+' '+models_dir+ds+" "+'/nfs/experiments/vasundhara/TMaP/learn_mt/MT'
	command = '/usr/bin/python3 '+path+program+'.py '+ args

	scriptfile = open(script_dir + scriptname,'w')
	scriptfile.write('#!/bin/bash\n')
	scriptfile.write('# SGE Options\n')
	scriptfile.write('#$ -l h_rt=100:00:00\n')
	scriptfile.write('#$ -e ' + error_dir+errorname+' \n')
	scriptfile.write('#$ -o ' + out_dir +stdoutname+' \n')
	scriptfile.write('#$ -l  hostname=uai0[23456789]*|uai1[0123456789]*|uai2[0123456789]*|uai30'+' \n')
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
	#data_dir = '/nfs/experiments/data/'
	#datasets = ['nltcs', 'msnbc', 'plants', 'baudio', 'jester', 'bnetflix', 'accidents', 'tretail', 'pumsb_star', 'dna', 'BN_0', 'BN_1', 'BN_2', 'BN_3', 'BN_4', 'BN_5', 'BN_6', 'BN_7', 'BN_8', 'BN_9', 'BN_10', 'BN_11', 'BN_12', 'BN_13', 'BN_14', 'BN_15', 'BN_28', 'BN_78', 'BN_94', 'BN_96', 'BN_100', 'BN_104', 'BN_106', 'BN_108', 'BN_110', 'BN_112', 'BN_114', 'BN_116', 'BN_118', 'BN_120', 'BN_122', 'BN_124']
	#data_dir = r'/nfs/experiments/vasundhara/Proposals/FBN_Samples/'
	datasets = ['BN_36.uai', 'BN_30.uai', 'BN_112.uai', 'BN_74.uai', 'BN_108.uai', 'BN_24.uai', 'BN_100.uai', 'BN_72.uai', 'BN_57.uai', 'BN_9.uai', 'BN_51.uai', 'BN_10.uai', 'BN_73.uai', 'BN_12.uai', 'BN_70.uai', 'BN_104.uai', 'BN_4.uai', 'BN_1.uai', 'BN_45.uai', 'BN_20.uai', 'BN_47.uai', 'BN_94.uai', 'BN_120.uai', 'BN_43.uai', 'BN_40.uai', 'BN_26.uai', 'BN_8.uai', 'BN_55.uai', 'BN_49.uai', 'BN_86.uai', 'BN_0.uai', 'BN_76.uai', 'BN_34.uai', 'BN_82.uai', 'BN_44.uai', 'BN_61.uai', 'BN_32.uai', 'BN_98.uai', 'BN_11.uai', 'BN_114.uai', 'BN_18.uai', 'BN_116.uai', 'BN_102.uai', 'BN_106.uai', 'BN_46.uai', 'BN_5.uai', 'BN_59.uai', 'BN_69.uai', 'BN_63.uai', 'BN_15.uai', 'BN_7.uai', 'BN_80.uai', 'BN_13.uai', 'BN_65.uai', 'BN_53.uai', 'BN_67.uai', 'BN_88.uai', 'BN_92.uai', 'BN_38.uai', 'BN_90.uai', 'BN_2.uai', 'BN_77.uai', 'BN_6.uai', 'BN_110.uai', 'BN_14.uai', 'BN_3.uai', 'BN_96.uai', 'BN_16.uai', 'BN_122.uai', 'BN_22.uai', 'BN_75.uai', 'BN_71.uai', 'BN_42.uai', 'BN_118.uai', 'BN_28.uai', 'BN_84.uai', 'BN_78.uai', 'BN_124.uai']
	data_dir = '/nfs/experiments/vasundhara/TMaP/sample_uai_bn/samples/'
	base_path = r'/nfs/experiments/vasundhara/TMaP/learn_mt/'
	sdir = base_path+'scripts/'
	for ds in datasets:
		script = write_PBS_script(base_path, data_dir, ds)
		script = sdir + script
		call(["qsub",script])
	                       
def main():
	runExperiments()

if __name__ == '__main__':
	main()
