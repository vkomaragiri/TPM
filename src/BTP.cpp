//
// Created by vasundhara on 1/15/20.
//

#include "../include/BTP.h"

BTP::BTP(CLT &clt) {
    pe = 1.0;
    variables = clt.variables;
    Utils::getOrder(variables, clt.functions, order);

    buckets = vector<vector<Function>> (order.size());
    var_pos = vector<int>(order.size());
    for(int i = 0; i < order.size(); i++){
        var_pos[order[i]] = i;
    }
    for(auto &func: clt.functions){
        int bucket_func = order.size();
        Function newf;
        func.instantiateEvid(newf);
        for(auto var: newf.variables){
            if(bucket_func > var_pos[var->id]){
                bucket_func = var_pos[var->id];
            }
        }
        if(newf.variables.size() == 0){
            pe *= newf.potentials[0];
            continue;
        }
        buckets[bucket_func].push_back(newf);
    }
    upward_performed = false;
    downward_performed = false;
}

ldouble BTP::getPE() {
    if(upward_performed){
        return pe;
    }
    upward_pass();
    return pe;
}

void BTP::getVarMarginals(vector<vector<ldouble>> &var_marginals) {
    propagate();
    var_marginals = vector<vector<ldouble>> (order.size());
    for(int i = 0; i < order.size(); i++){
        if(variables[order[i]]->isEvid()){
            var_marginals[order[i]] = vector<ldouble> (variables[order[i]]->d, 0.0);
            var_marginals[order[i]][variables[order[i]]->val] = 1.0;
            continue;
        }
        if(buckets[i].empty()){
            var_marginals[order[i]] = vector<ldouble> (variables[order[i]]->d, ldouble(1.0)/(ldouble ) variables[order[i]]->d);
            continue;
        }
        vector<Variable*> bucket_variables;
        for(auto &func: buckets[i]){
            Utils::doUnion(bucket_variables, func.variables);
        }
        vector<ldouble> mult, sum;
        Utils::multiplyBucket(buckets[i], bucket_variables, mult);
        vector<Variable*> mar_variables;
        mar_variables.push_back(variables[order[i]]);
        Utils::elimVariables(bucket_variables, mult, mar_variables, sum);
        Utils::normalize1d(sum);
        var_marginals[order[i]] = sum;
    }
}

void BTP::propagate() {
    if(upward_performed && downward_performed){
        return;
    }
    if(!upward_performed)
        upward_pass();
    if(!downward_performed)
        downward_pass();
}

void BTP::upward_pass(){ //leaves to root
    if(upward_performed) return;
    else{
        for(int i = 0; i < buckets.size(); i++){
            if(buckets[i].empty()){
                continue;
            }
            vector<Variable*> bucket_variables;
            for(auto &func: buckets[i]){
                Utils::doUnion(bucket_variables, func.variables);
            }
            vector<ldouble> mult, sum;
            Utils::multiplyBucket(buckets[i], bucket_variables, mult);

            vector<Variable*> elim_variables, mar_variables;
            elim_variables.push_back(variables[order[i]]);
            Utils::doDifference(bucket_variables, elim_variables, mar_variables);
            Utils::elimVariables(bucket_variables, mult, mar_variables, sum);
            if(mar_variables.empty()){
                pe *= sum[0];
                continue;
            }
            pe *= Utils::normalize1d(sum);
            Function newf(mar_variables, sum);
            int bucket_loc = buckets.size();
            for(auto var: newf.variables){
                if(var_pos[var->id] < bucket_loc){
                    bucket_loc = var_pos[var->id];
                }
            }
            if(bucket_loc < buckets.size()){
                buckets[bucket_loc].push_back(newf);
                edges.push_back(Message(i, bucket_loc, buckets[bucket_loc].size()-1));
            }
            else{
                cout << "Error!" << endl;
            }
        }
        upward_performed = true;
    }
}

void BTP::downward_pass(){ //root to leaves
    if(downward_performed) return;
    else{
        if(edges.empty())
            return;
        for(auto it = edges.end()-1; it >= edges.begin(); it--){
            auto &msg = *(it);
            if(buckets[msg.v].empty())
                continue;
            vector<Function> re_organized_bucket;
            vector<Variable*> bucket_variables;
            for(int i = 0; i < buckets[msg.v].size(); i++){
                if(i == msg.muv){
                    continue;
                }
                auto func = buckets[msg.v][i];
                re_organized_bucket.push_back(func);
                Utils::doUnion(bucket_variables, func.variables);
            }
            vector<ldouble> mult, sum;
            Utils::multiplyBucket(re_organized_bucket, bucket_variables, mult);
            vector<Variable*> mar_variables;
            Utils::doIntersection(bucket_variables, buckets[msg.v][msg.muv].variables, mar_variables);
            Utils::elimVariables(bucket_variables, mult, mar_variables, sum);

            Function newf(mar_variables, sum);
            buckets[msg.u].push_back(newf);
            msg.setParentMsg(buckets[msg.u].size()-1);
        }
        downward_performed = true;
    }
}


void BTP::getPosteriorSampler(BN_Sampler &bns) {
    propagate();
    bns = BN_Sampler();
    bns.variables = variables;

    for(int i = 0; i < order.size(); i++){
        int var = order[i];
        if(variables[var]->isEvid()){
            continue;
        }

        if(buckets[i].empty()){
            Function func = Function();
            func.variables.push_back(variables[var]);
            func.cpt_var = variables[var];
            func.potentials = vector<ldouble>(variables[var]->d, (ldouble) 1.0/ (ldouble)variables[var]->d);
            bns.sampling_functions.emplace_back(func);
        }
        else{

            vector<Variable*> bucket_variables;
            for(auto &func: buckets[i]){
                Utils::doUnion(bucket_variables, func.variables);
            }
            vector<ldouble> mult;
            Utils::multiplyBucket(buckets[i], bucket_variables, mult);
            Utils::normalize1d(mult);
            Function func = Function();
            func.variables = bucket_variables;
            func.potentials = mult;
            func.cpt_var = variables[var];
            bns.sampling_functions.emplace_back(func);
        }
    }
    bns.setEvidence();
}

/*
void BTP::getPosteriorDist(CLT &clt) {
    propagate();

    clt.variables = variables;
    clt.functions = vector<Function>();
    for(int i = 0; i < order.size(); i++){
        int var = order[i];
        if(variables[var]->isEvid()) continue;
        if(buckets[i].empty()){
            Function func = Function();;
            func.variables = vector<Variable*>(1);
            func.variables[0] = variables[var];
            func.potentials = vector<ldouble> (variables[var]->d, (ldouble)1.0/(ldouble)variables[var]->d);
            clt.functions.emplace_back(func);
        }
        else{
            vector<int> var_ind;
            vector<Variable*> bucket_variables;
            for(auto &func: buckets[i]){
                Utils::doUnion(bucket_variables, func.variables);

            }
            vector<ldouble> mult;
            Utils::multiplyBucket(buckets[i], bucket_variables, mult);
            Function func = Function();
            func.variables = bucket_variables;
            func.cpt_var = variables[var];
            func.potentials = mult;
            Utils::functionToCPT(func);
            clt.functions.emplace_back(func);
        }

    }
}
*/
