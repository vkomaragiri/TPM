//
// Created by vasundhara on 1/15/20.
//

#ifndef PROPOSALS_BTP_H
#define PROPOSALS_BTP_H


#include "InferenceEngine.h"
#include "Variable.h"
#include "Function.h"
#include "CLT.h"
#include "BN_Sampler.h"

struct Message{


    int u; //child bucket index
    int v; //parent bucket index
    int muv; //bucket index in parent that contains message from child;
    int mvu; //bucket index in child that contains message from parent;
    Message(int i, int i1, int i2): u(i), v(i1), muv(i2){}
    void setParentMsg(int i){
        mvu = i;
    }
};

class BTP: public MarginalInferenceEngine, PosteriorSamplerCreator<BN_Sampler>{//, PosteriorDistCreator<CLT> {
    vector<Variable*> variables;
    vector<vector<Function>> buckets;
    vector<int> order;
    vector<int> var_pos;
    vector<Message> edges;
    ldouble pe;
    bool downward_performed;
    bool upward_performed;

public:
    BTP():pe(1.0), downward_performed(false), upward_performed(false){}
    BTP(CLT &clt);

    void downward_pass();
    void upward_pass();
    void propagate();

    ldouble getPE();
    void getVarMarginals(vector<vector<ldouble>> &var_marginals);
    void getPosteriorSampler(BN_Sampler &bns);
    //void getPosteriorDist(CLT &clt);
};


#endif //PROPOSALS_BTP_H
