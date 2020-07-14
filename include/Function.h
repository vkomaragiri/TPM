//
// Created by vasundhara on 12/23/19.
//

#ifndef PROPOSALS_FUNCTION_H
#define PROPOSALS_FUNCTION_H


#include "Variable.h"
#include "MyTypes.h"
#include <vector>

class Function {
public:
    vector<Variable*> variables; //variables in the function
    Variable* cpt_var; //The CPT variable if function is a BN
    vector<ldouble> potentials; //The potential functions/cpts for a MN/BN
    vector<ldouble> gradients; //Gradient

    Function();
    Function(vector<Variable*> &vars);
    Function(vector<Variable*> &vars, vector<ldouble> &potentials_);
    Function(const Function &func_) = default;
    void instantiateEvid(Function &out);

};


#endif //PROPOSALS_FUNCTION_H
