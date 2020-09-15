//
// Created by vasundhara on 12/27/19.
//

#ifndef PROPOSALS_DATA_H
#define PROPOSALS_DATA_H

#include <iostream>
#include <vector>
#include <string>
#include "MyTypes.h"

using namespace std;
struct Data {

public:
    int nexamples, nfeatures;
    vector<int> dsize;
    vector<vector<int>> data_matrix;
    vector<vector<vector<vector<ldouble>>>> pxy;
    vector<vector<ldouble>> px;
    vector<vector<ldouble>> lpx;
    vector<vector<ldouble>> mi;
    vector<ldouble> weights;

    void setWeights(vector<ldouble> &wts){
        weights.clear();
        weights = wts;
    }
    bool readCSVData(string &filename);
    void append(Data &new_matrix);
    void computeMI(ldouble &laplace);
};


#endif //PROPOSALS_DATA_H
