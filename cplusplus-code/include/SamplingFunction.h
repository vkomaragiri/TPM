//
// Created by vasundhara on 1/30/20.
//

#ifndef PROPOSALS_SAMPLINGFUNCTION_H
#define PROPOSALS_SAMPLINGFUNCTION_H


#include <random>
#include "Variable.h"
#include "Function.h"
#include "myRandom.h"
#include "Utils.h"

struct SamplingFunction{
    vector<Variable*> other_variables;
    Variable* var;
    vector<discrete_distribution<int> > distributions;

    SamplingFunction(Function &func){
        var = func.cpt_var;
        vector<Variable*> vars2 = vector<Variable*>(1);
        vars2[0] = var;
        Utils::doDifference(func.variables, vars2, other_variables);
        int other_domain = Utils::getDomainSize(other_variables);
        distributions = vector<discrete_distribution<int>>(other_domain);
        for(int i = 0; i < other_domain; i++){
            Utils::setAddr(other_variables, i);
            vector<ldouble> prob = vector<ldouble>(var->d);
            for(int j = 0; j < var->d; j++){
                var->t_val = j;
                prob[j] = func.potentials[Utils::getAddr(func.variables)];
            }
            Utils::normalize1d(prob);
            distributions[i] = discrete_distribution<int>(prob.begin(), prob.end());
        }
    }

    void generateSample(){
        var->t_val = distributions[Utils::getAddr(other_variables)](myRandom::m_g);
    }
};

#endif //PROPOSALS_SAMPLINGFUNCTION_H
