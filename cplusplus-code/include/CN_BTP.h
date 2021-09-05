#ifndef PROPOSALS_CN_BTP_H
#define PROPOSALS_CN_BTP_H

#include "CN.h"
#include "InferenceEngine.h"
#include "CN_Sampler.h"

class CN_BTP: public MarginalInferenceEngine, PosteriorSamplerCreator<CN_Sampler>{
        CN cn_btp;
        ldouble pe;
        bool upward;//, downward;
        
    public:
        CN_BTP()=default;
        CN_BTP(CN &cn_);
        void setEvid(CNode* nd);
        ldouble getPE();
        void doUpwardPass();
        void upwardPass(CNode* nd);
        void getVarMarginals(vector<vector<ldouble>> &var_marginals);
        void computeVarMarginals(CNode* nd, vector<vector<ldouble>> &marg);
        void getPosteriorSampler(CN_Sampler &cns);
        //void doDownwardPass();
        //void downwardPass(CNode* nd);
        CNode* getPosteriorSamplerCNode(CNode* nd);
};

#endif