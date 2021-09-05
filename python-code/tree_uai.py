import os, sys, re
import numpy as np
import networkx as nx


class OR_Node:
    def __init__(self, val_, type_):
        self.id = -1
        self.val = -1
        if(type_ == 0):
            self.id = val_
        else:
            self.val = val_
        self.type_ = type_
        self.child_nodes = []
    
    def print_obj(self):
        print('(', end=' ')
        if(self.type_ == 0):
            print(self.type_, self.id, end = ' ')
        else:
            if(self.val < 0 or self.val > 1):
                print("Incorrect log!!")
                sys.exit(0)
            print(self.type_, self.val, end = ' ')
        print(len(self.child_nodes), end = ' ')
        for i in range(len(self.child_nodes)):
            self.child_nodes[i].print_obj()
        print(")", end = " ")

    def get_val(self, assignment, parents):
        if self.type_ == 1:
            return self.val
        ind = list(parents).index(self.id)
        return self.child_nodes[assignment[ind]].get_val(assignment, parents)

def read_func(lines, ind, dvar):
    if(ind >= len(lines)):
        return None, ind
    temp = lines[ind]
    nd = OR_Node(int(temp[0].split('_')[0][1:]), 0)
    if(len(temp) > 1):
        for i in range(len(temp)-1):
            nd.child_nodes.append(OR_Node(np.exp(np.float128(temp[i+1])), 1))
    else:
        for i in range(dvar[nd.id]):
            ind += 1
            child, ind = read_func(lines, ind, dvar)
            if(child is not None):
                nd.child_nodes.append(child)
    return nd, ind



def get_parents(nd, parents, id):
    if(nd.type_ == 0):
        if(nd.id != id):
            parents.append(nd.id)
        for child in nd.child_nodes:
            get_parents(child, parents, id)
    return parents

model_directory = sys.argv[1]
inputfilename = sys.argv[2]
fin = open(model_directory+inputfilename+'-bn.bn', "r")
dvar = [int(x) for x in fin.readline()[:-1].split(',')]
nvariables = len(dvar)
text = fin.readline()
if('BN' not in text):
    print("Error!")
    sys.exit(0)
functions = {}
for i in range(nvariables):
    var_id = fin.readline().split(' ')[0][1:]
    func_type = fin.readline()
    if('tree' not in func_type):
        print("Error!!!")
        sys.exit(0)
    func_text = []
    line = fin.readline()
    while(line[0] != '}'):
        line = [x for x in re.split(r"\(|\)| |\n", line) if x!='']
        func_text.append(line)
        line = fin.readline()
    res = read_func(func_text, 0, dvar)
    functions[var_id] = [res[0]]
    fin.readlines(2)
for i in list(functions.keys()):
    parents = [int(x) for x in list(set(get_parents(functions[i][0], [], i)))]
    #parents.sort()
    parents.remove(int(i))
    parents.append(int(i))
    functions[i].append(parents)




outfilename = model_directory+inputfilename+'.uai'
fout = open(outfilename, "w")
fout.write("BAYES\n")
fout.write(str(nvariables)+'\n')
for d in dvar:
    fout.write(str(d)+' ')
fout.write('\n')
fout.write(str(nvariables)+'\n')
for i in range(nvariables):
    parents = functions[str(i)][1]
    fout.write(str(len(parents))+' ')
    for p in parents:
        fout.write(str(p)+' ')
    fout.write('\n')
fout.write('\n')
for i in range(nvariables):
    parents = functions[str(i)][1]
    or_tree = functions[str(i)][0]
    nvals = 2**len(parents)
    fout.write(str(nvals)+'\n')

    for i in range(nvals):
        s = bin(i)[2:]
        assignment = [int(x) for x in s]
        if(len(assignment) < len(parents)):
            padding = list(np.zeros(len(parents)-len(assignment), dtype = int))
            padding.extend(assignment)
            assignment = padding
        val = or_tree.get_val(assignment, parents)
        fout.write(str(val)+' ')
    fout.write('\n\n')




fin.close()
fout.close()




# Verify is generated graph is DAG

G = nx.DiGraph()

for i in list(functions.keys()):
    parents = functions[i][1]
    parents.remove(int(i))
    for p in parents:
        G.add_edge(p, int(i))

print(nx.is_directed_acyclic_graph(G))