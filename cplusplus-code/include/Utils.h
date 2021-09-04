//
// Created by vasundhara on 12/23/19.
//

#ifndef PROPOSALS_UTILS_H
#define PROPOSALS_UTILS_H


#include "Variable.h"
#include "MyTypes.h"
#include "Data.h"
#include "Function.h"
#include "myRandom.h"

#include <unordered_map>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>



class Utils {
    public:
    static int getDomainSize(vector<Variable*> &elements);
    static void updateCPT(Function &func, Data &data, bool doStructLearning);
    static void getXStat(Variable* var, Data &data, vector<ldouble> &table);
    static void getXYStat(Variable* var1, Variable *cpt_var, Data &data, vector<vector<ldouble>> &table);
    static Data* sliceOfData(vector<int> &indices, Data &data);
    static void getOrder(vector<Variable*> &variables, vector<Function> &functions, vector<int> &order, const unordered_map<int, int> &varid_ind = unordered_map<int, int>());
    static void getMinDegreeOrder(vector<Variable*>& variables, vector<Function>& functions, vector<int>& order, const unordered_map<int, int> &varid_ind = unordered_map<int, int>());
    static void getTopologicalOrder(vector<Variable*>& variables, vector<Function>& functions, vector<int>& order, const unordered_map<int, int> &varid_ind = unordered_map<int, int>());
    static void getMinFillOrder(vector<Variable*>& variables, vector<Function>& functions, vector<int>& order, const unordered_map<int, int> &varid_ind = unordered_map<int, int>());
    static bool less_than_comp(Variable* a, Variable *b);
    static void doUnion(vector<Variable*> &vars1, vector<Variable*> &vars2);
    static void doIntersection(vector<Variable*> &vars1, vector<Variable*> &vars2, vector<Variable*> &out);
    static void doDifference(vector<Variable*> &vars1, vector<Variable*> &vars2, vector<Variable*> &out);
    static void multiplyBucket(vector<Function> &functions, vector<Variable*> &out_vars, vector<ldouble> &out);
    static void elimVariables( vector<Variable*> &all_vars, vector<ldouble> &joint, vector<Variable*> &mar_vars, vector<ldouble> &mar);

    static void computeEntropy(vector<vector<ldouble>> &px, ldouble &entropy);
    static void computeMI(vector<vector<vector<vector<ldouble>>>> &pxy, vector<vector<ldouble>> &px, vector<vector<ldouble>> &mi);
    static void computePartialMeasures(Data &data, vector<int> &data_indices, vector<int> &features, vector<vector<vector<vector<ldouble>>>> &pxy, vector<vector<ldouble>> &px, vector<vector<vector<vector<ldouble>>>> &cxy, vector<vector<ldouble>> &cx);
    static void updateCPT(Function &func, const vector<vector<vector<vector<ldouble>>>> &pxy, const vector<vector<ldouble>> &px, const unordered_map<int, int> &varid_ind, const vector<vector<vector<vector<ldouble>>>> &cn_cxy = vector<vector<vector<vector<ldouble>>>>(), const vector<vector<ldouble>> &cn_cx = vector<vector<ldouble>>());
    static int getSplittingVar(const vector<vector<ldouble>> &mi, bool randomize);

    static void poissonUpdateOR(Data &data, vector<int> &data_indices, Variable* var, vector<ldouble> &wts);
    static void poissonUpdateCPT(Data &dt, vector<int> &indices, Function &func);

    static void functionToCPT(Function &func){
        int ind = find(func.variables.begin(), func.variables.end(), func.cpt_var)-func.variables.begin();
        int all_size = getDomainSize(func.variables);

        vector<Variable*> other_variables(func.variables);
        other_variables.erase(other_variables.begin()+ind);
        int other_size = all_size/func.cpt_var->d;
        vector<ldouble> new_potential;
        for(int i = 0; i < other_size; i++){
            Utils::setAddr(other_variables, i);
            vector<ldouble> prob;
            for(int j = 0; j < func.cpt_var->d; j++){
                func.cpt_var->t_val = j;
                prob.emplace_back(func.potentials[Utils::getAddr(func.variables)]);
            }
            Utils::normalize1d(prob);
            copy(prob.begin(), prob.end(), back_inserter(new_potential));
        }
        func.potentials = new_potential;
    }

    static void printVarVector(vector<Variable*> &vars){
        for(auto &var: vars){
            cout << var->id << " ";
        }
        cout << endl;
    }

    template <class T>
    static ldouble sum1d(vector<T> &weights){
        T res = 0.0;
        for(auto &val: weights){
            res += val;
        }
        return res;
    }

    template <class T>
    static ldouble std1d(vector<T> &weights){
        T avg = sum1d(weights)/sizeof(weights);
        T res = 0.0;
        for(auto &val: weights){
            res += (val-avg)*(val-avg);
        }
        res/=sizeof(weights);
        res = sqrt(res);
        return res;
    }
    

    template<class T>
    static void normalizeDim2(vector<vector<T>> &v){
        vector<T> norm_const(v[0].size(), 0.0);
        for(int i = 0; i < v.size(); i++){
            for(int j = 0; j < v[i].size(); j++) {
                norm_const[j] += v[i][j];
            }
        }
        for(int i = 0; i < v.size(); i++){
            for(int j = 0; j < v[i].size(); j++) {
                v[i][j] /= norm_const[j];
            }
        }
    }
    static int getAddr(vector<Variable*> &variables){
        int ind = 0, multiplier = 1;
        if(!variables.empty()) {
            vector<Variable *>::iterator var;
            for (var = variables.end() - 1; var >= variables.begin(); var--) {
                ind += ((*var)->t_val) * multiplier;
                multiplier *= (*var)->d;
            }
        }
        return ind;
    }

    static void setAddr(vector<Variable*> &variables, int ind){
        int divider = 1;
        for(int i = variables.size()-1; i >= 0; i--){
            variables[i]->t_val = ind % variables[i]->d;
            ind /= variables[i]->d;
        }
    }

    template <class T>
    static void normalize2d(vector<vector<T>> &v){
        T norm_const = 0;
        for(auto &row: v){
            for(auto &elem: row){
                norm_const += elem;
            }
        }
        for(int i = 0; i < v.size(); i++){
            for(int j = 0; j < v[i].size(); j++){
                v[i][j] /= norm_const;
            }
        }
    }
    template <class T>
    static ldouble normalize1d(vector<T> &v){
        T norm_const = 0;
        for(auto &elem: v){
            norm_const += elem;
        }
        for(int i = 0; i < v.size(); i++){
            v[i] /= norm_const;
        }
        return norm_const;
    }
    template <class T>
    static void print1d(vector<T> &v){
        for(auto &elem: v){
            cout << elem << " ";
        }
        cout << endl;
    }
    
    template <class T>
    static void print1d(vector<T> &v, const string& outfilename){
        ofstream out(outfilename);
        for(auto &elem: v){
            out << elem << ",";
        }
    }
    
    template <class T>
    static void printMarginals(vector<vector<T> >& var_marginals, const string& outfilename)
    {
        if(!outfilename.empty()) {
            ofstream out(outfilename);
            out << "MAR\n";
            out << var_marginals.size();
            for (int i = 0; i < var_marginals.size(); i++) {
                out << " " << var_marginals[i].size();
                for (int j = 0; j < var_marginals[i].size(); j++)
                    out << " " << var_marginals[i][j];
            }
            out << "\n";
            out.close();
        }
        else{
            cout << "MAR\n";
            cout << var_marginals.size();
            for (int i = 0; i < var_marginals.size(); i++) {
                cout << " " << var_marginals[i].size();
                for (int j = 0; j < var_marginals[i].size(); j++)
                    cout << " " << var_marginals[i][j];
            }
            cout << "\n";
        }
    }

    static void printSamples(vector<vector<int> >& samples, const string& outfilename)
    {
        if(!outfilename.empty()) {
            ofstream out(outfilename);
            for (int i = 0; i < samples.size(); i++) {
                for (int j = 0; j < samples[i].size()-1; j++)
                    out << samples[i][j] << ",";
                out << samples[i][samples[i].size()-1];
                out << endl;
            }
        }
        else{
            for (int i = 0; i < samples.size(); i++) {
                for (int j = 0; j < samples[i].size()-1; j++)
                    cout << samples[i][j] << ",";
                cout << samples[i][samples[i].size()-1];
                cout << endl;
            }
        }
    }

    static void printEvid(vector<int> & evid_var, vector<int> &evid_val, const string& outfilename)
    {
        if(!outfilename.empty()) {
            ofstream out(outfilename);
            out << evid_var.size() << " ";
            for (int i = 0; i < evid_var.size(); i++) {
                out << evid_var[i] << " " << evid_val[i] << " ";
            }
            out << endl;
        }
        else{
            cout << evid_var.size() << " ";
            for (int i = 0; i < evid_var.size(); i++) {
                cout << evid_var[i] << " " << evid_val[i] << " ";
            }
            cout << endl;
        }
    }

    static void readSamples(vector<vector<int> >& data_matrix, const string& infilename)
    {
        if(!infilename.empty()) {
            ifstream in(infilename);
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
        }
        else{
            cout << "Error: File empty; nothing to read." << endl;
        }
    }
};


#endif //PROPOSALS_UTILS_H
