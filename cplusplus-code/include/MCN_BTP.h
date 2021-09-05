#ifndef MCN_BTP_H
#define MCN_BTP_H

#include "MCN.h"
#include "CN_BTP.h"
#include "InferenceEngine.h"
#include "MCN_Sampler.h"

class MCN_BTP: public MarginalInferenceEngine, PosteriorSamplerCreator<MCN_Sampler>{
    MCN mcn;
    vector<CN_BTP> cn_btps;

    public:
        MCN_BTP(MCN &mcn_);
        MCN_BTP(){}
        ldouble getPE();
        void getVarMarginals(vector<vector<ldouble>> &var_marginals);

        void getPosteriorSampler(MCN_Sampler& mcns);
};

#endif