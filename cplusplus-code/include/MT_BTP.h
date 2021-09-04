//
// Created by vasundhara on 1/23/20.
//

#ifndef PROPOSALS_MT_BTP_H
#define PROPOSALS_MT_BTP_H


#include "InferenceEngine.h"
#include "MT.h"
#include "BTP.h"
#include "MT_Sampler.h"

class MT_BTP: MarginalInferenceEngine, PosteriorSamplerCreator<MT_Sampler>{//, PosteriorDistCreator<MT> {
    MT mt;
    vector<BTP> btps;

public:
    MT_BTP(){}
    MT_BTP(const MT &mt_);

    ldouble getPE();
    void getVarMarginals(vector<vector<ldouble>> &var_marginals);

    void getPosteriorSampler(MT_Sampler& mts);
    //void getPosteriorDist(MT &other_mt);
};


#endif //PROPOSALS_MT_BTP_H
