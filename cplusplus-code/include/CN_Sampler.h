#ifndef PROPOSALS_CN_SAMPLER_H
#define PROPOSALS_CN_SAMPLER_H


#include "Variable.h"
#include "BN_Sampler.h"
#include "CN.h"

class CN_Sampler {
    
public:
    CN cns;
    vector<Variable*> variables;
    CN_Sampler(CN &cn_);
    CN_Sampler()=default;
    CNode* getSamplerCNode(CNode* nd);

    void generateSamples(int n, vector<vector<int>> &samples);
    void generateCNodeSample(CNode* nd, vector<int> &sample);
    ldouble getProbability(vector<int> &sample);
    ldouble getCNodeLogProbability(CNode* nd, vector<int> &example);
};


#endif //PROPOSALS_CN_SAMPLER_H