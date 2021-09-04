//
// Created by vasundhara on 12/23/19.
//

#include "../include/Function.h"
#include "../include/Utils.h"

Function::Function() = default;

Function::Function(vector<Variable*> &variables_ ) : variables(variables_ ), cpt_var(nullptr){
    potentials = vector<ldouble> (Utils::getDomainSize(variables_), 1);
    count_potentials = vector<ldouble> (Utils::getDomainSize(variables_), 1);
}

Function::Function(vector<Variable*> &vars, vector<ldouble> &potentials_) : variables(vars), potentials(potentials_), cpt_var(
        nullptr){}

void Function::instantiateEvid(Function &out) {
    out = Function();
    for(auto var: variables){
        if(var->is_evid){
            var->t_val = var->val;
        }
        else{
            out.variables.push_back(var);
        }
    }
    if(out.variables.size() == variables.size()){
        out.potentials = potentials;
        out.cpt_var = cpt_var;
    }
    else{
        int dom_size = Utils::getDomainSize(out.variables);
        out.potentials = vector<ldouble>(dom_size, 1.0);
        for(int i = 0; i < dom_size; i++){
            Utils::setAddr(out.variables, i);
            out.potentials[i] = potentials[Utils::getAddr(variables)];
        }
    }
}
