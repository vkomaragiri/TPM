//
// Created by vasundhara on 1/15/20.
//

#ifndef PROPOSALS_INFERENCEENGINE_H
#define PROPOSALS_INFERENCEENGINE_H

#include <MyTypes.h>
#include <vector>

using namespace std;
class MarginalInferenceEngine{
public:
    // Get Probability of Evidence or Partition Function
    virtual ldouble getPE()=0;
    virtual void getVarMarginals(vector<vector<ldouble>> &var_marginals)=0;
};

template <class T>
class PosteriorSamplerCreator{
public:
    virtual void getPosteriorSampler(T& sampler)=0;
};

/*
template <class T>
class PosteriorDistCreator{
public:
    virtual void getPosteriorDist(T& sampler)=0;
};
*/
#endif //PROPOSALS_INFERENCEENGINE_H
