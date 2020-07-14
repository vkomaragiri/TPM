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

    BN_Sampler(CLT &clt);
    BN_Sampler(){};
    void setEvidence();
    void generateSamples(int n, vector<vector<int>> &samples);
    ldouble getProbability(vector<int> &sample);
    ldouble getWeight(vector<int> &sample);
};


#endif //PROPOSALS_BN_SAMPLER_H
