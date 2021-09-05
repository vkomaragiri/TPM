#ifndef PROPOSALS_CN_H
#define PROPOSALS_CN_H
#include <iostream>
#include <vector>

#include "Data.h"
#include "GM.h"
#include "CLT.h"
#include "BTP.h"

struct CNode{
    vector<ldouble> child_weights;
    vector<ldouble> count_child_weights;
    vector<CNode*> children;
    bool type;
    CLT *clt;

    vector<vector<vector<vector<ldouble>>>> pxy;
    vector<vector<ldouble>> px;
    vector<vector<vector<vector<ldouble>>>> cxy; //counts
    vector<vector<ldouble>> cx; //counts
    ldouble entropy;
    vector<vector<ldouble>> mi;
    unordered_map<int, int> varid_ind;

    Variable* var;
    vector<int> features;
    BTP *btp;
    ldouble val;
    vector<ldouble> child_posterior_weights;
    
    discrete_distribution<int> distribution;
    BN_Sampler *bns;

    vector<ldouble> grad_child_weights;
};

class CN: public GM{
    CNode* root;

    public:
        CN() = default;
        void learn(Data &train_data, Data &valid_data, bool prune, bool randomize, bool isComponent, int max_depth=10);
        CNode* learnCNode(Data &train_data, vector<int> &partition, vector<int> &features, int depth, bool randomize, int max_depth);
        bool termination_condition(int nexamples, ldouble entropy, int depth, int max_depth);
        void pruneCNode(CNode* nd, Data &train_data, Data &valid_data);

        ldouble log_likelihood(Data &data);
        ldouble getLogProb(vector<int> &example);
        ldouble getProb(vector<int> &example);
        void getLogProbCNode(CNode* nd, vector<int> &example, ldouble &out);

        CNode* getRoot(){
            return root;
        }

        void setRoot(CNode* nd){
            root = nd;
        }

        void setEvidence(int var, int val){
            variables[var]->setValue(val);
        }

        void print();
        void printCNNode(CNode* nd);
        void write(string infile);
        void writeCNode(CNode* nd, ofstream &out);
        void read(string infile);
        CNode* readCNode(ifstream &in);

        void initGrad();
        void initGradCNode(CNode* nd);
        void compGrad(vector<vector<int>> &data, vector<int> &partition, CNode* nd);
        void doSGDUpdateCNode(CNode* nd, ldouble learning_rate);
        void doSGDUpdate(ldouble learning_rate);
        ldouble gradSqNorm(CNode* nd);

        void poissonOnlineLearnCNode(CNode* nd, Data &dt, vector<int> &indices, int depth);//, bool paramsOnly, int depth);
        void poissonOnlineLearn(Data &data, vector<int> &indices);//, bool paramsOnly);

        void writeCounts(string infile);
        void writeCNodeCounts(CNode* nd, ofstream &out);
        void readCounts(string infile);
        CNode* readCNodeCounts(ifstream &in);
        void normalizeParams();
        void normalizeParamsCNode(CNode* nd);
};
#endif