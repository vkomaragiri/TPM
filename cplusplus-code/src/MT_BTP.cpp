//
// Created by vasundhara on 1/23/20.
//

#include "../include/MT_BTP.h"

MT_BTP::MT_BTP(const MT &mt_) {
    mt = mt_;
    for(int i = 0; i < mt.ncomponents; i++){
        btps.push_back(BTP(mt.trees[i]));
    }
}

ldouble MT_BTP::getPE() {
    ldouble res = 0.0;
    for(int i = 0; i < mt.ncomponents; i++){
        res += mt.prob_mixture[i]*btps[i].getPE();
    }
    return res;
}

void MT_BTP::getVarMarginals(vector<vector<ldouble>> &var_marginals){
    vector<ldouble> post_prob_mixture(mt.ncomponents);
    var_marginals = vector<vector<ldouble>>(mt.variables.size());
    for(int i = 0; i < mt.variables.size(); i++){
        var_marginals[i] = vector<ldouble> (mt.variables[i]->d, 0.0);
    }
    vector<vector<ldouble>> comp_marginals;
    for(int i = 0; i < mt.ncomponents; i++){
        comp_marginals.clear();
        btps[i].getVarMarginals(comp_marginals);
        post_prob_mixture[i] = mt.prob_mixture[i]*btps[i].getPE();
        for(int j = 0; j < comp_marginals.size(); j++){
            for(int k = 0; k < comp_marginals[j].size(); k++){
                var_marginals[j][k] += post_prob_mixture[i]*comp_marginals[j][k];
            }
        }
    }
    for(int i = 0; i < var_marginals.size(); i++){
        Utils::normalize1d(var_marginals[i]);
    }
}

void MT_BTP::getPosteriorSampler(MT_Sampler& mts) {
    mts = MT_Sampler();
    vector<ldouble> post_prob_mixture(mt.ncomponents);
    mts.variables = mt.variables;
    mts.samplers = vector<BN_Sampler> (mt.ncomponents);

    for(int i = 0; i < mt.ncomponents; i++){
        post_prob_mixture[i] = mt.prob_mixture[i]*btps[i].getPE();
        btps[i].getPosteriorSampler(mts.samplers[i]);
    }
    Utils::normalize1d(post_prob_mixture);
    mts.mixture_distribution = discrete_distribution<int> (post_prob_mixture.begin(), post_prob_mixture.end());
}

/*
void MT_BTP::getPosteriorDist(MT &other_mt) {
    other_mt = MT();
    other_mt.variables = mt.variables;
    other_mt.ncomponents = mt.ncomponents;
    other_mt.prob_mixture = vector<ldouble>(mt.ncomponents);
    other_mt.trees = vector<CLT>(mt.ncomponents);

    for(int i = 0; i < mt.ncomponents; i++){
        btps[i].getPosteriorDist(other_mt.trees[i]);
        other_mt.prob_mixture[i] = mt.prob_mixture[i]*btps[i].getPE();
    }
    Utils::normalize1d(other_mt.prob_mixture);
}
*/
