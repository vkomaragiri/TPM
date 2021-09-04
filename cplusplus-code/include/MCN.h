//
// Created by Vasundhara Komaragiri on 12/8/20.
//

#ifndef CPLUSPLUS_CODE_MCN_H
#define CPLUSPLUS_CODE_MCN_H

#include "CN.h"
#include "HyperParameters.h"
#include "Data.h"
#include "myRandom.h"
#include <random>

class MCN: public GM{
    public:
    vector<ldouble> prob_mixture;
    vector<ldouble> count_prob_mixture;
    int ncomponents;
    vector<CN> cns;

    vector<ldouble> grad_prob_mixture;

    public:
    MCN():ncomponents(0){}
    void learn(Data &train_data, Data &valid_data, int nbags=10, int max_depth=10);
    
    
    void write(const string &filename);
    void read(const string &filename);

    ldouble log_likelihood(Data &data);
    ldouble getProb(vector<int> &example);
    ldouble getLogProb(vector<int> &example);

    void setEvidence(int var, int val);

    void initGrad();
    void compCNodeGrad(CNode* nd, vector<int> &data, ldouble &total_prob, ldouble &comp_log_prob);
    void compGrad(vector<vector<int>> &data);
    void doSGDUpdate(ldouble learning_rate);
    ldouble gradSqNorm();

    void addComps(vector<vector<vector<int>>> &bags, vector<int> &dsize, Data &valid_data);
    void poissonOnlineLearn(Data &dt, Data &valid_data);//, bool paramsOnly);

    void writeCounts(const string &filename);
    void readCounts(const string &filename);
    void normalizeParams();

    void mergeModel(MCN &new_model);
};

#endif //CPLUSPLUS_CODE_MCN_H