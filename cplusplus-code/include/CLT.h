//
// Created by vasundhara on 1/3/20.
//

#ifndef PROPOSALS_CLT_H
#define PROPOSALS_CLT_H

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include<map>
#include <sstream>


#include "HyperParameters.h"
#include "GM.h"
#include "Utils.h"


typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,boost::property<boost::vertex_index_t, int>,boost::property<boost::edge_weight_t, ldouble>> Graph;
typedef Graph::edge_descriptor Edge;

class CLT: public GM {
public:
    vector<vector<ldouble>> gradients;

    CLT(const CLT& clt) = default;
    CLT(){}
    ldouble getProbability(vector<int> &example);
    ldouble getLogProbability(vector<int> &example);
    ldouble log_likelihood(Data &data);
    void learn(Data &data, bool isComp=false, bool doStructLearning = true, int r=0, ldouble laplace = HyperParameters::laplace, bool part_cn_=false, const vector<vector<ldouble>> &cn_mi = vector<vector<ldouble>>(), const vector<vector<ldouble>> &cn_px = vector<vector<ldouble>>(), const vector<vector<vector<vector<ldouble>>>> &cn_pxy = vector<vector<vector<vector<ldouble>>>>(), const unordered_map<int, int> &cn_varid_ind = unordered_map<int, int>(), const vector<vector<ldouble>> &cn_cx = vector<vector<ldouble>>(), const vector<vector<vector<vector<ldouble>>>> &cn_cxy = vector<vector<vector<vector<ldouble>>>>());
    void print();
    void write(string infile);
    void readUAI08(string infile);

    void setEvidence(int var, int val);
    void removeEvidence(int var);

    void readCN(ifstream &in, const unordered_map<int, int> &cn_varid_ind);
    void writeCN(ofstream &out);

    /*
    GM* clone() const{
        return new CLT(static_cast<const CLT&>(*this)); // call the copy ctor.
    }
    */
    void initGrad();
    void compGrad(vector<vector<int>> &data, vector<int> &partition);
    void doSGDUpdate(ldouble learning_rate);
    ldouble gradSqNorm();

    void readCNCounts(ifstream &in, const unordered_map<int, int> &cn_varid_ind);
    void writeCNCounts(ofstream &out);
    void normalizeParams();

    ldouble getLogLWPostProbability(vector<int> &example);
};


#endif //PROPOSALS_CLT_H
