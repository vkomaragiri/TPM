//
// Created by vasundhara on 12/23/19.
//

#ifndef PROPOSALS_VARIABLE_H
#define PROPOSALS_VARIABLE_H

#include <iostream>

using namespace std;

class Variable {
public:
    int id, d, val, t_val, is_evid;

    Variable(): id(-1), d(-1), val(-1), t_val(-1), is_evid(0){}

    Variable(int i, int d): id(i), d(d), t_val(-1), is_evid(0), val(-1){}
    Variable(const Variable &var) = default;

    int isEvid(){
        return is_evid;
    }

    int getValue(){
        return val;
    }

    void setValue(int v){
        if(v < d) {
            val = v;
            is_evid = 1;
        }
        else
            cout << "Value to be assigned invalid" << endl;
    }
};


#endif //PROPOSALS_VARIABLE_H
