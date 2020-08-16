import numpy as np
import scipy.stats as stats
import networkx as nx
import matplotlib.pyplot as plt
import sys


def write_bayes(potentials):
    f = open(sys.argv[1], "w")
    f.write("BAYES\n")
    f.write(str(len(potentials.keys()))+"\n")
    (2*np.ones(len(potentials.keys()), dtype=np.int)).tofile(f, " ")
    f.write(str()+"\n")
    f.write(str(len(potentials.keys()))+"\n")
    for e in potentials.keys():
        f.write(str(len(e))+" ")
        np.array(e, dtype=np.int).tofile(f, " ")
        f.write("\n")

    for e in potentials.keys():
        f.write(str(len(potentials[e]))+"\n")
        np.array(potentials[e]).tofile(f, " ")
        f.write("\n")

def main():
    edges = []
    num_vertices = 50
    for i in range(9):
        edges.append((i, i+1))
    for i in range(10, 19):
        edges.append((i, i+1))
    for i in range(20, 29):
        edges.append((i, i+1))
    for i in range(30, 39):
        edges.append((i, i+1))
    for i in range(40, 49):
        edges.append((i, i+1))
    for i in range(num_vertices-10):
        edges.append((i, i+10))

    print(edges)

    lower, upper = 0, 1
    mu, sigma = 0.5, 0.45
    dist = stats.truncnorm((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma)
    mu, sigma = 0.1, 0.05
    dist1 = stats.truncnorm((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma)
    mu, sigma = 0.9, 0.05
    dist2 = stats.truncnorm((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma)

    dist1 = dist
    dist2 = dist

    potentials = {}
    G = nx.DiGraph()
    G.add_edges_from(edges)

    nx.draw_networkx(G, with_labels=True)
    plt.show()
    for n in G.nodes:
        parents = list(G.predecessors(n))
        pot = []
        if len(parents) == 0:
            r = np.random.ranf()
            if r <= 0.5:
                val = dist1.rvs(1)
            else:
                val = dist2.rvs(1)
            pot.extend([val[0], 1-val[0]])
        else:
            for v in range(2**len(parents)):
                r = np.random.ranf()
                if r <= 0.5:
                    val = dist1.rvs(1)
                else:
                    val = dist2.rvs(1)
                prob = [val[0], 1-val[0]]
                print('prob', prob)
                pot.extend(prob)
        print(pot)
        parents.append(n)
        potentials[tuple(parents)] = pot
    print(potentials)
    write_bayes(potentials)
if __name__ == '__main__':
    main()