//
// Created by vasundhara on 1/7/20.
//

#include <fstream>
#include "../include/MT.h"

void MT::learnEM(Data &data, Data &valid_data) {
    //Initialization
    ncomponents = HyperParameters::num_components;
    prob_mixture = vector<ldouble> (ncomponents);
    trees = vector<CLT> (ncomponents);
    vector<vector<ldouble>> weights(ncomponents, vector<ldouble> (data.nexamples));
    vector<int> partition(data.nexamples);
    vector<int> features(data.nfeatures);

    for (int i = 0; i < data.nfeatures; i++) {
        Variable* var = new Variable(i, data.dsize[i]);
        variables.push_back(var);
        features.push_back(i);
    }
    for(int i = 0; i < data.nexamples; i++){
        partition.push_back(i);
    }
    for(int i = 0; i < ncomponents; i++){
        trees[i].variables = variables;
        for(ldouble & j : weights[i]){
            j = myRandom::getDouble();
        }
    }

    //EM
    Utils::normalizeDim2(weights);
    for (int iter = 0; iter < HyperParameters::num_iterations_em ; iter++) {
        //M-Step
        for (int i = 0; i < ncomponents; i++) {
            data.setWeights(weights[i]);
            prob_mixture[i] = Utils::sum1d(weights[i]);
            trees[i].learn(data, true, iter % HyperParameters::interval_for_structure_learning == 0, 0, HyperParameters::laplace/ncomponents);
            /*
            if (iter == 1)
                trees[i].learn(data, weights[i], true, true);
            else
                trees[i].learn(data, weights[i], true, false);
            */
        }
        Utils::normalize1d(prob_mixture);
        //E-Step
        for (int j = 0; j < data.nexamples; j++) {
            for (int i = 0; i < ncomponents; i++) {
                weights[i][j] = prob_mixture[i] * trees[i].getProbability(data.data_matrix[j]);
            }
        }
        Utils::normalizeDim2(weights);
                cout << iter << " " << log_likelihood(valid_data) << endl;

    }
}

/*
void MT::learnGD(Data &data, Data &test_data) {
    //Initialization
    ncomponents = HyperParameters::num_components;
    prob_mixture = vector<ldouble> (ncomponents);
    trees = vector<CLT> (ncomponents);
    vector<vector<ldouble>> weights = vector<vector<ldouble>> (ncomponents);

    for (int i = 0; i < data.nfeatures; i++) {
        Variable* var = new Variable(i, data.dsize[i]);
        variables.push_back(var);
    }
    for(int i = 0; i < ncomponents; i++){
        trees[i].variables = variables;
        weights[i] = vector<ldouble> (data.nexamples);
        for(ldouble & j : weights[i]){
            j = myRandom::getDouble();
        }
    }


    for (int i = 0; i < ncomponents; i++) {
        prob_mixture[i] = Utils::sum1d(weights[i]);
        trees[i].learn(data, weights[i], true, true);
        for (auto &func: trees[i].functions){
            func.gradients = vector<ldouble>(func.potentials.size(), 0.0);
        }
    }
    Utils::normalize1d(prob_mixture);
    for (int iter = 0; iter < 100 ; iter++) {
        ldouble gamma = 2.0/(2.0+iter);
        for(int j = 0; j < data.nexamples; j++){
            for(int i = 0; i < data.data_matrix[j].size(); i++){
                variables[i]->t_val = data.data_matrix[j][i];
            }
            ldouble norm_const = 0;
            for(int i = 0; i < ncomponents; i++){
                weights[i][j] = prob_mixture[i]*trees[i].getProbability(data.data_matrix[j]);
                norm_const += weights[i][j];
            }
            for(int i = 0; i < ncomponents; i++){
                for (auto &func: trees[i].functions){
                    for(int k = 0; k < func.gradients.size(); k++){
                        func.gradients[k] += (weights[i][j]/(norm_const*func.potentials[Utils::getAddr(func.variables)]));
                    }
                }
                weights[i][j] /= (norm_const*prob_mixture[i]);
                weights[i][j] *= gamma;
            }
        }
        for (int i = 0; i < ncomponents; i++) {
            prob_mixture[i] += Utils::sum1d(weights[i]);
            for (auto &func: trees[i].functions) {
                //Utils::print1d(func.potentials);
                //Utils::print1d(func.gradients);
                for (int k = 0; k < func.potentials.size(); k++){
                    func.potentials[k] += gamma * func.gradients[k];
                    func.gradients[k] = 0.0;
                    Utils::functionToCPT(func);
                }
                //Utils::print1d(func.potentials);
            }
        }
        Utils::normalize1d(prob_mixture);
        cout << iter << " " << log_likelihood(test_data) << endl;
    }
}
*/

ldouble MT::getProbability(vector<int> &example){
    /*
    for(int i = 0; i < example.size(); i++){
        variables[i]->t_val = example[i];
    }
    ldouble prob = 1.0, comp_prob = 0;
    for(int i = 0; i < ncomponents; i++){
        comp_prob += prob_mixture[i]*trees[i].getProbability(example);
    }
    prob *= comp_prob;
    return prob;
     */
    return exp(getLogProbability(example));
}

ldouble MT::getLogProbability(vector<int> &example){
    for(int i = 0; i < example.size(); i++){
        variables[i]->t_val = example[i];
    }
    ldouble prob = 0.0, comp_prob = 0;
    for(int i = 0; i < ncomponents; i++){
        comp_prob += prob_mixture[i]*trees[i].getProbability(example);
    }
    prob += log(comp_prob);
    return prob;
}

void MT::setEvidence(int var, int val)
{
    variables[var]->setValue(val);
}

ldouble MT::log_likelihood(Data &data) {
    ldouble res = 0.0;
    for(auto example: data.data_matrix){
        res += getLogProbability(example);
    }
    res /= data.nexamples;
    return res;
}

void MT::print() {
    for(int i = 0; i < ncomponents; i++){
        cout << "Tree-" << i << endl;
        trees[i].print();
    }
    cout << "prob_mix: ";
    Utils::print1d(prob_mixture);
}

/*
void MT::learnRF(Data &data, int r, Data &valid_data) {
    //Initialization
    ncomponents = HyperParameters::num_components;
    prob_mixture = vector<ldouble> (ncomponents);
    trees = vector<CLT> (ncomponents);
    vector<Data> bags = vector<Data> (ncomponents);
    for (int i = 0; i < data.nfeatures; i++) {
        auto *var = new Variable(i, data.dsize[i]);
        variables.push_back(var);
    }
    for(int i = 0; i < ncomponents; i++){
        trees[i].variables = variables;
        vector<int> indices = vector<int> (data.nexamples);
        for(int i = 0; i < data.nexamples; i++){
            indices[i] = myRandom::getInt(data.nexamples);
        }
        bags[i].data_matrix.clear();
        for(auto &ind: indices){
            bags[i].data_matrix.push_back(data.data_matrix[ind]);
        }
        bags[i].nexamples = bags[i].data_matrix.size();
        bags[i].nfeatures = bags[i].data_matrix[0].size();
        bags[i].dsize = data.dsize;

        vector<ldouble> weights = vector<ldouble> (bags[i].nexamples, 1.0);
        trees[i].learn(bags[i], weights, true, true, r);
        prob_mixture[i] = trees[i].log_likelihood(valid_data);

    }
    Utils::normalize1d(prob_mixture);
}

void MT::learnSEM(Data &data, int m) {
    //Initialization

    ncomponents = HyperParameters::num_components;
    prob_mixture = vector<ldouble> (ncomponents);
    trees = vector<CLT> (ncomponents);
    vector<vector<ldouble>> weights = vector<vector<ldouble>> (ncomponents);
    for(auto & weight : weights){
        weight = vector<ldouble> (m);
    }
    for (int i = 0; i < data.nfeatures; i++) {
        auto *var = new Variable(i, data.dsize[i]);
        variables.push_back(var);
    }
    for(int i = 0; i < ncomponents; i++){
        trees[i].variables = variables;
    }

    vector<int> indices;
    for(int i = 0; i < data.nexamples; i++){
        indices.push_back(i);
    }
    shuffle(begin(indices), end(indices), myRandom::m_g);
    vector<int> batch(indices.begin(), indices.begin()+m);
    Data cur_data = *(Utils::sliceOfData(batch, data));

    //EM
    //Initialization
    for(auto & weight : weights){
        for(ldouble & j : weight){
            j = myRandom::getDouble();
        }
    }
    Utils::normalizeDim2(weights);
    for (int i = 0; i < ncomponents; i++) {
        prob_mixture[i] = Utils::sum1d(weights[i]);
        trees[i].learn(cur_data, weights[i], true, true);
    }
    Utils::normalize1d(prob_mixture);

    int k = 0; //stepsize
    ldouble alpha = 0.6;

    for (int iter = 1; iter < HyperParameters::num_iterations_em ; iter++) {

        for(int i = 0; i < data.nexamples/m; i++) {
            if(i == data.nexamples/m-1 && data.nexamples%m != 0 ){
                vector<int> batch(indices.begin()+i*m, indices.end());
                copy(indices.begin(), indices.begin()+m-(data.nexamples%m), back_inserter(batch));
            }
            else{
                vector<int> batch(indices.begin()+i*m, indices.begin()+(i+1)*m);
            }

            //M-Step
            Utils::normalize1d(prob_mixture);
            for (int i = 0; i < ncomponents; i++) {
                prob_mixture[i] = Utils::sum1d(weights[i]);
                trees[i].learn(cur_data, weights[i], true, iter % HyperParameters::interval_for_structure_learning == 0);
            }
            //E-Step
            ldouble eta_k = pow(k+2, -alpha);
            for (int j = 0; j < cur_data.nexamples; j++) {
                for (int i = 0; i < ncomponents; i++) {
                    weights[i][j] = (1-eta_k)*weights[i][j] + eta_k*this->prob_mixture[i] * this->trees[i].getProbability(cur_data.data_matrix[j]);
                    //weights[i][j] = this->prob_mixture[i] * this->trees[i].getProbability(cur_data.data_matrix[j]);
                }
            }
            Utils::normalizeDim2(weights);
            Utils::normalize1d(prob_mixture);
            k++;
        }
    }
}
*/
void MT::write(const string &filename) {
    if(!filename.empty()) {
        ofstream out(filename);
        out << "MT\n";
        out << this->ncomponents << " " << this->variables.size() << "\n";
        for (auto &variable : variables) {
            out << variable->d << " ";
        }
        out << endl;
        for (int i = 0; i < ncomponents; i++) {
            out << prob_mixture[i] << " ";
        }
        out << endl;
        for (int i = 0; i < this->trees.size(); i++) {
            out << trees[i].functions.size() << "\n";
            for (auto &function : this->trees[i].functions) {
                out << function.variables.size() << " ";
                for (auto &variable : function.variables) {
                    out << variable->id << " ";
                }
                out << function.cpt_var->id << endl;
            }
            for (auto &function : this->trees[i].functions) {
                out << function.potentials.size() << "\n";
                for (long double k : function.potentials) {
                    out << k << " ";
                }
                out << endl;
            }
        }
        out.close();
    }
    else{
        cout << "MT\n";
        cout << this->ncomponents << " " << this->variables.size() << "\n";
        for (auto &variable : variables) {
            cout << variable->d << " ";
        }
        cout << endl;
        for (int i = 0; i < ncomponents; i++) {
            cout << prob_mixture[i] << " ";
        }
        cout << endl;
        for (int i = 0; i < this->trees.size(); i++) {
            cout << trees[i].functions.size() << "\n";
            for (auto &function : this->trees[i].functions) {
                cout << function.variables.size() << " ";
                for (auto &variable : function.variables) {
                    cout << variable->id << " ";
                }
                cout << function.cpt_var->id << endl;
            }
            for (auto &function : this->trees[i].functions) {
                cout << function.potentials.size() << "\n";
                for (long double k : function.potentials) {
                    cout << k << " ";
                }
                cout << endl;
            }
        }
    }
}

void MT::read(const string &filename)
{
    ifstream in(filename);
    string mystr;
    in>>mystr;
    if(mystr.find("MT")==string::npos){
        cerr<<"Something wrong with the file, cannot read\n";
        exit(-1);
    }
    int numvars;
    in>>ncomponents;
    in>>numvars;

    variables=vector<Variable*>(numvars);
    for(int i=0;i<numvars;i++) {
        int d;
        in >> d;
        variables[i] = new Variable(i, d);
    }

    prob_mixture=vector<ldouble>(ncomponents);
    for(int i=0;i<ncomponents;i++){
        ldouble tmp;
        in>>tmp;
        prob_mixture[i]=tmp;
    }
    trees=vector<CLT>(ncomponents);
    for(int i=0;i<ncomponents;i++){
        int nfuncs;
        in>>nfuncs;
        trees[i].variables=variables;
        trees[i].functions=vector<Function>(nfuncs);
        for (int j=0;j<nfuncs;j++) {
            int nvars;
            in>>nvars;
            trees[i].functions[j].variables=vector<Variable*> (nvars);
            for (int k=0;k<nvars;k++) {
                int varid;
                in >> varid;
                trees[i].functions[j].variables[k] = variables[varid];
                //if(k == nvars-1)
                //    trees[i].functions[j].cpt_var = variables[varid];
            }
            int cpt_varid;
            in >> cpt_varid;
            trees[i].functions[j].cpt_var = variables[cpt_varid];
        }

        for (int j=0;j<nfuncs;j++) {
            vector<ldouble>& table=trees[i].functions[j].potentials;
            int tab_size; 
            in>>tab_size;
            table=vector<ldouble>(tab_size);
            for (int k=0;k<tab_size;k++) {
                in>>table[k];
                //if(table[k] < 0.01) table[k] = 0.01;
                //else if(table[k] > 0.99) table[k] = 0.99;
            }
            //Utils::normalize1d(table);
        }
    }
    in.close();
}

void MT::initGrad(){
    grad_prob_mixture = vector<ldouble>(ncomponents, 0.0);
    for(int i = 0; i < ncomponents; i++){
        trees[i].initGrad();
    }
}

void MT::compGrad(vector<vector<int>> &data){
    for(int i = 0; i < data.size(); i++){
        for(auto &var: variables){
            var->t_val = data[i][var->id];
        }
        ldouble log_prob = getLogProbability(data[i]);
        for(int j = 0; j < ncomponents; j++){
            ldouble comp_log_prob = trees[j].getLogProbability(data[i]);
            grad_prob_mixture[j] += exp(comp_log_prob-log_prob);

            for(int k = 0; k < trees[j].functions.size(); k++){
                int t = Utils::getAddr(trees[j].functions[k].variables);
                trees[j].gradients[k][t] += exp(log(prob_mixture[j])+comp_log_prob-log(trees[j].functions[k].potentials[t])-log_prob);
            }
        }
    }
}

void MT::doSGDUpdate(ldouble learning_rate){
    for(int k = 0; k < ncomponents; k++){
        prob_mixture[k] += learning_rate*grad_prob_mixture[k];
        trees[k].doSGDUpdate(learning_rate);
    }
    Utils::normalize1d(prob_mixture);
}

ldouble MT::gradSqNorm(){
    ldouble res = 0.0;
    for(int j = 0; j < ncomponents; j++){
        res += grad_prob_mixture[j]*grad_prob_mixture[j];
        res += trees[j].gradSqNorm();
    }
    return res;
}