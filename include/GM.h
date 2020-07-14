//
// Created by vasundhara on 12/27/19.
//

#ifndef PROPOSALS_GM_H
#define PROPOSALS_GM_H


#include "Variable.h"
#include "Function.h"
#include "Data.h"
#include <vector>

class GM {
public:
    vector<Variable*> variables;
    vector<Function> functions;

    virtual ldouble log_likelihood(Data &data){};
    virtual void print(){};
};


#endif //PROPOSALS_GM_H
