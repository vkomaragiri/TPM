//
// Created by vasundhara on 1/30/20.
//

#ifndef PROPOSALS_BN_SAMPLER_H
#define PROPOSALS_BN_SAMPLER_H


#include "SamplingFunction.h"
#include "CLT.h"
#include "Utils.h"

class BN_Sampler {

public:
    vector<Variable*> variables;
    vector<int> evidence_variables;
    vector<SamplingFunction> sampling_functions;
    bool sampler_type;

    BN_Sampler(CLT &clt, const unordered_map<int, int> &varid_ind = unordered_map<int, int>());
    BN_Sampler()=default;
    void setEvidence();
    void generateSamples(int n, vector<vector<int>> &samples);
    void generateSample(vector<int> &sample);
    ldouble getProbability(vector<int> &sample);
    ldouble getLogWeight(vector<int> &sample);
};


#endif //PROPOSALS_BN_SAMPLER_H
