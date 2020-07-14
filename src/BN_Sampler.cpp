//
// Created by vasundhara on 1/30/20.
//

#include "../include/BN_Sampler.h"

BN_Sampler::BN_Sampler(CLT &clt) {
    //cout << 0 << endl;
    variables = clt.variables;
    //Utils::printVarVector(variables);
    vector<int> order;
    Utils::getTopologicalOrder(variables, clt.functions, order);
    //cout << 1 << endl;
    vector<int> var_func = vector<int> (order.size());
    for(int i = 0; i < clt.functions.size(); i++){
        var_func[clt.functions[i].cpt_var->id] = i;
    }
    //cout << 2 << endl;
    for(int var : order){
        sampling_functions.emplace_back(clt.functions[var_func[var]]);
    }
    //cout << 3 << endl;
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

ldouble BN_Sampler::getProbability(vector<int> &sample) {
    ldouble p = 1.0;
    for(auto &func: sampling_functions){
        if(func.var->isEvid()) continue;
        p *= func.distributions[Utils::getAddr(func.other_variables)].probabilities()[func.var->t_val];
    }
    return p;
}

ldouble BN_Sampler::getWeight(vector<int> &sample) {
    ldouble wt = 1.0;
    for(auto &var: variables){
        var->t_val = sample[var->id];
    }
    for(auto &func: sampling_functions){
        if(func.var->isEvid()){
            wt *= func.distributions[Utils::getAddr(func.other_variables)].probabilities()[func.var->t_val];
        }
    }
    return wt;
}



