//
// Created by vasundhara on 1/30/20.
//

#ifndef PROPOSALS_MT_SAMPLER_H
#define PROPOSALS_MT_SAMPLER_H


#include "Variable.h"
#include "BN_Sampler.h"
#include "MT.h"

class MT_Sampler {

public:
    discrete_distribution<int> mixture_distribution;
    vector<BN_Sampler> samplers;
    vector<Variable*> variables;


    MT_Sampler(MT &mt);
    MT_Sampler(){}
    void generateSamples(int n, vector<vector<int>> &samples);
    ldouble getProbability(vector<int> &sample);
};


#endif //PROPOSALS_MT_SAMPLER_H
