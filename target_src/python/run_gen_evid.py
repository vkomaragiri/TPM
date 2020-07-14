import os, sys, subprocess

datasets = ['nltcs', 'msnbc', 'kdd', 'plants', 'baudio', 'jester', 'bnetflix', 'accidents', 'tretail', 'pumsb_star']
evids = ['0', '20', '50', '80']


evid_dir = '/home/vasundhara/UTD/Research/Proposals/evid/CLT/'
for ds in datasets:
    for ev in evids:
        command = ['/home/vasundhara/UTD/Research/Proposals/cmake-build-debug/genEvid', '-i','/home/vasundhara/UTD/Research/Proposals/models/CLT/'+ds+'.clt','-o', evid_dir+ev+'/'+ds+'.evid','-e',ev]
        subprocess.call(command)