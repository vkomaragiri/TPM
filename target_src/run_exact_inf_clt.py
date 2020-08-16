import os, sys, subprocess

#datasets = ['BN_0', 'BN_1', 'BN_2', 'BN_3', 'BN_4', 'BN_5', 'BN_6', 'BN_7', 'BN_8', 'BN_9', 'BN_10', 'BN_11', 'BN_12', 'BN_13', 'BN_14', 'BN_15', 'BN_28', 'BN_78', 'BN_94', 'BN_96', 'BN_98', 'BN_100', 'BN_102', 'BN_104', 'BN_106', 'BN_108', 'BN_110', 'BN_112', 'BN_114', 'BN_116', 'BN_118', 'BN_120', 'BN_122', 'BN_124']
#datasets = ['nltcs', 'msnbc', 'kdd', 'plants', 'baudio', 'jester', 'bnetflix', 'accidents', 'tretail', 'pumsb_star']
datasets = ['nltcs', 'msnbc', 'kdd', 'plants', 'baudio', 'jester', 'bnetflix', 'accidents', 'tretail', 'pumsb_star', 'BN_0', 'BN_1', 'BN_2', 'BN_3', 'BN_4', 'BN_5', 'BN_6', 'BN_7', 'BN_8', 'BN_9', 'BN_10', 'BN_11', 'BN_12', 'BN_13', 'BN_14', 'BN_15', 'BN_28', 'BN_78', 'BN_94', 'BN_96', 'BN_100', 'BN_104', 'BN_106', 'BN_108', 'BN_110', 'BN_112', 'BN_114', 'BN_116', 'BN_118', 'BN_120', 'BN_122', 'BN_124']
#datasets = ['msnbc', 'kdd']
evids = ['20', '50', '80']
#tasks = ['PR', 'MAR']

evid_dir = '/home/vasundhara/UTD/Research/Proposals/evid/FBN/NN/'
for ds in datasets:
    for ev in evids:
        print(ds, ev)
        #for task in tasks:
            #command = ['/home/uaiuser/solvers/vec-uai14','/home/uaiuser/vasundhara/vec_check_Proposals/CLT/'+ds+'.clt','/home/uaiuser/vasundhara/vec_check_Proposals/evid/'+ev+'/'+ds+'.evid', '0', task]
        command = ['python', '/home/vasundhara/UTD/Research/Proposals/cmake-build-debug/exactInf.py', '/home/vasundhara/UTD/Research/Proposals/models/MT/EM/'+ds+'.mt', evid_dir+ev+'/'+ds+'.evid', '/home/vasundhara/UTD/Research/Proposals/results/MAR/MT/NN/'+ev+'/'+ds+'.MAR', '/home/vasundhara/UTD/Research/Proposals/results/PR/MT/NN/'+ev+'/'+ds+'.PR']

        #command = ['/home/vasundhara/UTD/Research/Original/cmake-build-debug/exactInfCLT', '-i', '/home/vasundhara/UTD/Research/Proposals/models/CLT/'+ds+'.clt', '-e', evid_dir+ev+'/'+ds+'.evid', '-o', '/home/vasundhara/UTD/Research/Proposals/results/Original_CLT/'+ev+'/'+ds+'.MAR', '-p', '/home/vasundhara/UTD/Research/Proposals/results/Original_CLT/'+ev+'/'+ds+'.PR']

        subprocess.call(command)
