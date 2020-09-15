//
// Created by vasundhara on 12/27/19.
//

#include <fstream>
#include <sstream>
#include <Utils.h>
#include <cmath>
#include "../include/Data.h"


bool Data::readCSVData(string &dataset){
    ifstream in(dataset);
    if(!in.good())
        return 0;
    string line;
    data_matrix.clear();
    while(getline(in, line)){
        stringstream inss(line);
        vector<int> row;
        int m;
        while(inss >> m){
            row.push_back(m);
            if (inss.peek() == ',')
                inss.ignore();
        }
        data_matrix.push_back(row);
    }
    nexamples = data_matrix.size();
    if(data_matrix.size() > 0){
        nfeatures = data_matrix[0].size();
    }
    dsize = vector<int>(nfeatures, 1);
    for(int i = 0; i < data_matrix.size(); i++){
        for(int j = 0; j < data_matrix[i].size(); j++){
            if(data_matrix[i][j] > (dsize[j]-1)){
                dsize[j] = data_matrix[i][j]+1;
            }
        }
    }
    return 1;
}

void Data::append(Data &new_data){
    if(nfeatures == new_data.nfeatures && nexamples > 0) {
        data_matrix.insert(data_matrix.end(), new_data.data_matrix.begin(), new_data.data_matrix.end());
        nexamples = data_matrix.size();
        for(int i = 0; i < new_data.data_matrix.size(); i++){
            for(int j = 0; j < new_data.data_matrix[i].size(); j++){
                if(new_data.data_matrix[i][j] > (dsize[j]-1)){
                    dsize[j] = new_data.data_matrix[i][j]+1;
                }
            }
        }
    }
    else{
        cout << "Inconsistent dimension sizes" << endl;
    }
}

void Data::computeMI(ldouble &laplace){
    pxy = vector<vector<vector<vector<ldouble>>>> (nfeatures, vector<vector<vector<ldouble>>> (nfeatures));
    px = vector<vector<ldouble>> (nfeatures);
    for(int i = 0; i < nfeatures; i++){
        for(int j = 0; j < nfeatures; j++){
            pxy[i][j] = vector<vector<ldouble>> (dsize[i]);
            for(int k = 0; k < dsize[i]; k++){
                pxy[i][j][k] = vector<ldouble> (dsize[j], laplace); //laplace count
            }
        }
        px[i] = vector<ldouble> (dsize[i], laplace); //laplace 1-addition
    }
    for(int i = 0; i < nexamples; i++){
        for(int j = 0; j < nfeatures; j++){
            px[j][data_matrix[i][j]]+=weights[i];
            for(int k = 0; k < nfeatures; k++){
                pxy[j][k][data_matrix[i][j]][data_matrix[i][k]] += weights[i];
            }
        }
    }
    for(int i = 0; i < nfeatures; i++){
        for(int j = 0; j < nfeatures; j++){
            Utils::normalize2d(pxy[i][j]);
        }
        Utils::normalize1d(px[i]);
    }

    //Store log px for faster implementation.
    lpx = vector<vector<ldouble>> (nfeatures);
    for(int i = 0; i < px.size(); i++){
        lpx[i] = vector<ldouble> (dsize[i]);
        for(int xi = 0; xi < px[i].size(); xi++){
            lpx[i][xi] = log(px[i][xi]);
        }
    }

    mi = vector<vector<ldouble>> (nfeatures, vector<ldouble> (nfeatures));
    for(int i = 0; i < nfeatures; i++){
        for(int j = i+1; j < nfeatures; j++){
            mi[i][j] = 0;
            for(int xi = 0; xi < dsize[i]; xi++){
                for(int xj = 0; xj < dsize[j]; xj++){
                    mi[i][j] += pxy[i][j][xi][xj]*(log(pxy[i][j][xi][xj])-lpx[i][xi]-lpx[j][xj]);
                }
            }
        }
    }
}