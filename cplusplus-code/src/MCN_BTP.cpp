#include <MCN_BTP.h>

MCN_BTP::MCN_BTP(MCN &mcn_){
    mcn = MCN();
    mcn = mcn_;
    cn_btps = vector<CN_BTP>(mcn.ncomponents);
    for(int i = 0; i < mcn.ncomponents; i++){
        cn_btps[i] = CN_BTP(mcn.cns[i]);
    }
}

ldouble MCN_BTP::getPE(){
    ldouble res = 0.0;
    for(int i = 0; i < mcn.ncomponents; i++){
        res += mcn.prob_mixture[i]*cn_btps[i].getPE();
    }
    return res;
}

void MCN_BTP::getVarMarginals(vector<vector<ldouble>> &var_marginals){
    vector<ldouble> post_prob_mixture(mcn.ncomponents);
    var_marginals = vector<vector<ldouble>>(mcn.variables.size());
    for(int i = 0; i < mcn.variables.size(); i++){
        var_marginals[i] = vector<ldouble> (mcn.variables[i]->d, 0.0);
    }
    vector<vector<ldouble>> comp_marginals;
    for(int i = 0; i < mcn.ncomponents; i++){
        comp_marginals.clear();
        cn_btps[i].getVarMarginals(comp_marginals);
        post_prob_mixture[i] = mcn.prob_mixture[i]*cn_btps[i].getPE();
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

void MCN_BTP::getPosteriorSampler(MCN_Sampler& mcns) {
    mcns = MCN_Sampler();
    vector<ldouble> post_prob_mixture(mcn.ncomponents);
    mcns.variables = mcn.variables;
    mcns.samplers = vector<CN_Sampler> (mcn.ncomponents);

    for(int i = 0; i < mcn.ncomponents; i++){
        post_prob_mixture[i] = mcn.prob_mixture[i]*cn_btps[i].getPE();
        cn_btps[i].getPosteriorSampler(mcns.samplers[i]);
    }
    Utils::normalize1d(post_prob_mixture);
    mcns.mixture_distribution = discrete_distribution<int> (post_prob_mixture.begin(), post_prob_mixture.end());
}