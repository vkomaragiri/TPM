//
// Created by vasundhara on 1/30/20.
//

#ifndef MCN_SAMPLER_H
#define MCN_SAMPLER_H


#include "Variable.h"
#include "CN_Sampler.h"
#include "MCN.h"

class MCN_Sampler {

public:
    discrete_distribution<int> mixture_distribution;
    vector<CN_Sampler> samplers;
    vector<Variable*> variables;


    MCN_Sampler(MCN &mt);
    MCN_Sampler(){}
    void generateSamples(int n, vector<vector<int>> &samples);
    ldouble getProbability(vector<int> &sample);
};


#endif //PROPOSALS_MCN_SAMPLER_H
