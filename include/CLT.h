//
// Created by vasundhara on 1/3/20.
//

#ifndef PROPOSALS_CLT_H
#define PROPOSALS_CLT_H

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include "GM.h"
#include "Utils.h"
#include "HyperParameters.h"

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,boost::property<boost::vertex_index_t, int>,boost::property<boost::edge_weight_t, ldouble>> Graph;
typedef Graph::edge_descriptor Edge;

class CLT: public GM {
public:
    CLT(const CLT& clt) = default;
    CLT(){}
    ldouble getProbability(vector<int> &example);
    ldouble getLogProbability(vector<int> &example);
    ldouble log_likelihood(Data &data);
    void learn(Data &data, vector<ldouble> &weights_, bool isComp=false, bool doStructLearning = true, int r=0, ldouble laplace = HyperParameters::laplace);
    void print();
    void write(string infile);
    void readUAI08(string infile);

    void setEvidence(int var, int val);
    void removeEvidence(int var);
};


#endif //PROPOSALS_CLT_H
