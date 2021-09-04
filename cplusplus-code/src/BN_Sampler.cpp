//
// Created by vasundhara on 1/30/20.
//

#include "../include/BN_Sampler.h"

BN_Sampler::BN_Sampler(CLT &clt, const unordered_map<int, int> &varid_ind) {
    variables = clt.variables;
    vector<int> order;
    Utils::getTopologicalOrder(variables, clt.functions, order, varid_ind);
    vector<int> var_func(order.size());
    for(int i = 0; i < clt.functions.size(); i++){
        if(varid_ind.empty())
            var_func[clt.functions[i].cpt_var->id] = i;
        else{
            var_func[varid_ind.at(clt.functions[i].cpt_var->id)] = i;
        }
    }
    for(int var : order){
        sampling_functions.emplace_back(clt.functions[var_func[var]]);
    }
}

void BN_Sampler::setEvidence(){
    for(int i = 0; i < variables.size(); i++){
        Variable* var = variables[i];
        if(var->isEvid()){
            var->t_val = var->val;
            evidence_variables.push_back(i);
        }
    }
}

void BN_Sampler::generateSamples(int n, vector<vector<int>> &samples) {
    samples = vector<vector<int>> (n);
    for(int i = 0; i < n; i++) {
        samples[i] = vector<int> (variables.size());
        for (auto &func: sampling_functions) {
            func.generateSample();
            samples[i][func.var->id] = func.var->t_val;
        }
        for(auto &j: evidence_variables){
            if(variables[j]->isEvid()){
                samples[i][variables[j]->id] = variables[j]->val;
            }
        }
    }
}

void BN_Sampler::generateSample(vector<int> &sample){
    for (auto &func: sampling_functions) {
            func.generateSample();
            sample[func.var->id] = func.var->t_val;
        }
    for(auto &j: evidence_variables){
        if(variables[j]->isEvid()){
            sample[variables[j]->id] = variables[j]->val;
        }
    }
}

ldouble BN_Sampler::getProbability(vector<int> &sample) {
    ldouble p = 0.0;
    for(auto &func: sampling_functions){
        if(func.var->isEvid()) continue;
        p += log(func.distributions[Utils::getAddr(func.other_variables)].probabilities()[func.var->t_val]);
    }
    return exp(p);
}


ldouble BN_Sampler::getLogWeight(vector<int> &sample) {
    ldouble wt = 0.0;
    for(auto &var: variables){
        var->t_val = sample[var->id];
    }
    for(auto &func: sampling_functions){
        if(func.var->isEvid()){
            wt += log(func.distributions[Utils::getAddr(func.other_variables)].probabilities()[func.var->t_val]);
        }
    }
    return wt;
}

