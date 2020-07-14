//
// Created by vasundhara on 1/7/20.
//

#ifndef PROPOSALS_MT_H
#define PROPOSALS_MT_H

#include "MyTypes.h"
#include "CLT.h"
#include "HyperParameters.h"
#include "myRandom.h"
#include "Utils.h"

class MT: public GM {
public:
    int ncomponents;
    vector<ldouble> prob_mixture;
    vector<CLT> trees;

    MT(): ncomponents(0){}
    ~MT(){}
    MT(const MT &mt): ncomponents(mt.ncomponents), prob_mixture(mt.prob_mixture), trees(mt.trees){}
    void learnEM(Data &data);
    void learnRF(Data &data, int r, Data &valid_data);
    void learnSEM(Data &data, int m = 1);

    void learnGD(Data &data, Data &test_data);

    void write(const string& filename);
    void read(const string& filename);

    void print();
    ldouble log_likelihood(Data &data);

    ldouble getLogProbability(vector<int> example);
    ldouble getProbability(vector<int> example);

    void setEvidence(int var, int val);
};


#endif //PROPOSALS_MT_H
