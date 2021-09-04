//
// Created by vasundhara on 12/23/19.
//

#include <HyperParameters.h>
#include <set>
#include <list>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <myRandom.h>
#include <boost/assert.hpp>
#include "../include/Utils.h"

int Utils::getDomainSize(vector<Variable*> &elements) {
    int res = 1;
    for(auto &it: elements){
        res *= it->d;
    }
    return res;
}

void Utils::getXStat(Variable* var, Data &data, vector<ldouble> &table) {
    table = vector<ldouble>(var->d, 1.0);
    for(int i = 0; i < data.data_matrix.size(); i++){
        table[data.data_matrix[i][var->id]] += data.weights[i];
    }
    Utils::normalize1d(table);
}


void Utils::getXYStat(Variable* var1, Variable *cpt_var, Data &data, vector<vector<ldouble>> &table){
    table = vector<vector<ldouble>> (var1->d);
    for(int i = 0; i < var1->d; i++){
        table[i] = vector<ldouble> (cpt_var->d, 1.0);
    }
    for(int i = 0; i < data.data_matrix.size(); i++){
        table[data.data_matrix[i][var1->id]][data.data_matrix[i][cpt_var->id]] += data.weights[i];
    }
    Utils::normalize2d(table);
}

void Utils::updateCPT(Function &func, Data &data, bool doStructLearning){
    if(func.variables.size() == 1){
        //Utils::getXStat(func.variables[0], data, func.potentials, partition);
        
        if(!doStructLearning) {
            Utils::getXStat(func.variables[0], data, func.potentials);
        }
        else{
            func.potentials = data.px[func.variables[0]->id];
        }
        
        for(auto &val: func.potentials){
            if(val < 0.01) val = 0.01;
            else if(val > 0.99) val = 0.99;
        }
        Utils::normalize1d(func.potentials);
    }
    else if(func.variables.size() == 2){
        Variable *t = func.cpt_var, *s;
        if (func.variables[0]->id == func.cpt_var->id) {
            s = func.variables[1];
        } else {
            s = func.variables[0];
        }
        func.potentials = vector<ldouble>(s->d*t->d, 0.0);
        vector<vector<ldouble>> table;
        //Utils::getXYStat(s, t, data, table, partition);
        
        if(!doStructLearning) {
            Utils::getXYStat(s, t, data, table);
        }
        else{
            table = data.pxy[s->id][t->id];
        }
        
        for(auto &row: table){
            for(auto &val: row){
                if(val < 0.01) val = 0.01;
                else if(val > 0.99) val = 0.99;
            }
        }
        Utils::normalize2d(table);
        for(int i = 0; i < s->d; i++){
            s->t_val = i;
            ldouble norm_const = 0.0;
            for(int j = 0; j < t->d; j++){
                norm_const += table[i][j];
            }
            for(int j = 0; j < t->d; j++) {
                t->t_val = j;
                func.potentials[Utils::getAddr(func.variables)] = table[i][j] / norm_const;
            }
        }
    }
}

Data* Utils::sliceOfData(vector<int> &indices, Data &data){
    Data* res = new Data();
    res->nexamples = indices.size();
    res->nfeatures = data.nfeatures;
    res->dsize = data.dsize;
    res->data_matrix.clear();
    for(auto &i: indices){
        res->data_matrix.push_back(data.data_matrix[i]);
    }
    return res;
}

void Utils::doIntersection(vector<Variable *> &vars1, vector<Variable *> &vars2, vector<Variable *> &out) {
    /*
    if(vars1.size() < vars2.size()){
        for(auto &var: vars1){
            if(find(vars2.begin(), vars2.end(), var) != vars2.end()){
                out.push_back(var);
            }
        }
    }
    else{
        for(auto &var: vars2){
            if(find(vars1.begin(), vars1.end(), var) != vars1.end()){
                out.push_back(var);
            }
        }
    }
    */
    if(vars1.size() < vars2.size()){
        out = vector<Variable*>(vars1.size());
    }
    else{
        out = vector<Variable*>(vars2.size());
    }
    vector<Variable*>::iterator it;
    it = set_intersection(vars1.begin(), vars1.end(), vars2.begin(), vars2.end(), out.begin(), Utils::less_than_comp);
    out.resize(it-out.begin());
}

bool Utils::less_than_comp(Variable *a, Variable *b) {
    return (a->id < b->id);
}

void Utils::doUnion(vector<Variable *> &vars1, vector<Variable *> &vars2) {
    /*
    for(auto &var: vars2){
        if(find(vars1.begin(), vars1.end(), var) == vars1.end())
            vars1.push_back(var);
    }
    */
    vector<Variable*> out(vars1.size()+vars2.size());
    vector<Variable*>::iterator it;
    it = set_union(vars1.begin(), vars1.end(), vars2.begin(), vars2.end(), out.begin(), Utils::less_than_comp);
    out.resize(it-out.begin());
    vars1.clear();
    for(auto &var: out){
        vars1.push_back(var);
    }

}

void Utils::doDifference(vector<Variable*> &vars1, vector<Variable*> &vars2, vector<Variable*> &out) {
    /*
    for (auto &var: vars1) {
        if (find(vars2.begin(), vars2.end(), var) == vars2.end()) {
            out.push_back(var);
        }
    }
    */
    out = vector<Variable*>(vars1.size());
    vector<Variable*>::iterator it;
    it = set_difference(vars1.begin(), vars1.end(), vars2.begin(), vars2.end(), out.begin(), Utils::less_than_comp);
    out.resize(it-out.begin());

}
void Utils::multiplyBucket(vector<Function> &functions, vector<Variable*> &out_vars, vector<ldouble> &out){
    int dom_size = Utils::getDomainSize(out_vars);
    out = vector<ldouble> (dom_size, 1.0);
    for(int i = 0; i < dom_size; i++){
        setAddr(out_vars, i);
        for(auto &func: functions){
            out[i] *= func.potentials[getAddr(func.variables)];
        }
    }
}

void Utils::elimVariables(vector<Variable*> &all_vars, vector<ldouble> &joint, vector<Variable*> &mar_vars, vector<ldouble> &mar) {
    int dom_size = Utils::getDomainSize(mar_vars);
    mar = vector<ldouble>(dom_size, 0);
    int full_size = Utils::getDomainSize(all_vars);
    for (int i = 0; i < full_size; i++) {
        setAddr(all_vars, i);
        mar[getAddr(mar_vars)] += joint[i];
    }
}
void Utils::getOrder(vector<Variable *> &variables, vector<Function> &functions, vector<int> &order, const unordered_map<int, int> &varid_ind) {
    if (HyperParameters::ord_heu==min_fill){
        Utils::getMinFillOrder(variables,functions,order, varid_ind);
    } else{
        if (HyperParameters::ord_heu==min_degree){
            Utils::getMinDegreeOrder(variables,functions,order, varid_ind);
        }
        else {
            if(HyperParameters::ord_heu==topological){
                Utils::getTopologicalOrder(variables,functions,order, varid_ind);
            }
        }
    }
}

void Utils::getMinDegreeOrder(vector<Variable *> &variables, vector<Function> &functions, vector<int> &order, const unordered_map<int, int> &varid_ind) {
    vector<set<int> > degree(variables.size());
    vector<set<int> > graph(variables.size());

    for (auto &function : functions) {
        for (int j = 0; j < function.variables.size(); j++) {
            if (function.variables[j]->is_evid) continue;
            for (int k = j + 1; k < function.variables.size(); k++) {
                if (function.variables[k]->is_evid) continue;
                int a, b;
                if(varid_ind.empty()){
                    a = function.variables[j]->id;
                    b = function.variables[k]->id;
                }
                else{
                    a = varid_ind.at(function.variables[j]->id);
                    b = varid_ind.at(function.variables[j]->id);
                }
                graph[a].insert(b);
                graph[b].insert(a);
            }
        }
    }
    int min_degree = variables.size();
    int max_degree = -1;
    for (int i = 0; i < variables.size(); i++) {
        degree[graph[i].size()].insert(i);
        if (graph[i].size() < min_degree) {
            min_degree = graph[i].size();
        }
        if (graph.size() > max_degree) {
            max_degree = graph.size();
        }
    }
    order = vector<int>();
    while (order.size() != variables.size()) {
        while (degree[min_degree].empty()) {
            min_degree++;
        }
        int curr = *(degree[min_degree].begin());
        degree[min_degree].erase(curr);
        for (auto i = graph[curr].begin(); i != graph[curr].end(); i++) {
            degree[graph[*i].size()].erase(*i);
            graph[*i].erase(curr);
            degree[graph[*i].size()].insert(*i);
            if (graph[*i].size() < min_degree) { min_degree = graph[*i].size(); }
        }
        order.push_back(curr);
    }
}

void Utils::getTopologicalOrder(vector<Variable *> &variables, vector<Function> &functions, vector<int> &order, const unordered_map<int, int> &varid_ind) {
    vector<set<int> > degree(variables.size());
    vector<set<int> > parents(variables.size());
    vector<set<int> > children(variables.size());

    for (auto function : functions) {
        if(function.cpt_var==nullptr){
            cerr<<"Something wrong: Topological order for a non Bayes net\n";
            exit(-1);
        }
        int a;
        if(varid_ind.empty())
            a = function.cpt_var->id;
        else
            a = varid_ind.at(function.cpt_var->id);
        for (int j = 0; j < function.variables.size(); j++) {
            if (function.variables[j]->id==function.cpt_var->id) continue;
            int b;
            if(varid_ind.empty()) 
                b = function.variables[j]->id;
            else
                b = varid_ind.at(function.variables[j]->id);
            parents[a].insert(b);
            children[b].insert(a);
        }
    }
    for (int i = 0; i < variables.size(); i++) {
        degree[parents[i].size()].insert(i);
    }
    order = vector<int>();
    while (!degree[0].empty()) {
        int curr = *(degree[0].begin());
        degree[0].erase(curr);
        for (auto i = children[curr].begin(); i != children[curr].end(); i++) {
            degree[parents[*i].size()].erase(*i);
            parents[*i].erase(curr);
            degree[parents[*i].size()].insert(*i);
        }
        order.push_back(curr);
    }
    if(order.size()!=variables.size()){
        cerr<<"Something wrong while computing topological order\n";
        cerr<<"Seems that the Bayes net is not a DAG\n";
        exit(-1);
    }
}

void Utils::getMinFillOrder(vector<Variable *> &variables, vector<Function> &functions, vector<int> &order, const unordered_map<int, int> &varid_ind) {
    double estimate = 0.0;
    int max_cluster_size = 0;
    order = vector<int>(variables.size());
    vector<set<int> >clusters(variables.size());
    vector<vector<bool> > adj_matrix(variables.size());

    // Create the interaction graph of the functions in this graphical model - i.e.
    // create a graph structure such that an edge is drawn between variables in the
    // model that appear in the same function
    for (int i = 0; i < variables.size(); i++) {
        adj_matrix[i] = vector<bool>(variables.size());
    }
    vector<set<int> > graph(variables.size());
    vector<bool> processed(variables.size());
    for (auto & function : functions) {
        for (int j = 0; j < function.variables.size(); j++) {
            if (function.variables[j]->is_evid) continue;
            for (int k = j + 1; k < function.variables.size(); k++) {
                if (function.variables[k]->is_evid) continue;
                int a, b;
                if(varid_ind.empty()){
                    a = function.variables[j]->id;
                    b = function.variables[k]->id;
                }
                else{
                    a = varid_ind.at(function.variables[j]->id);
                    b = varid_ind.at(function.variables[k]->id);
                }
                graph[a].insert(b);
                graph[b].insert(a);
                adj_matrix[a][b] = true;
                adj_matrix[b][a] = true;
            }
        }
    }
    list<int> zero_list;

    // For i = 1 to number of variables in the model
    // 1) Identify the variables that if deleted would add the fewest number of edges to the
    //    interaction graph
    // 2) Choose a variable, pi(i), from among this set
    // 3) Add an edge between every pair of non-adjacent neighbors of pi(i)
    // 4) Delete pi(i) from the interaction graph
    for (int i = 0; i < variables.size(); i++) {
        // Find variables with the minimum number of edges added
        double min = DBL_MAX;
        int min_id = -1;
        bool first = true;

        // Flag indicating whether the variable to be removed is from the
        // zero list - i.e. adds no edges to interaction graph when deleted
        bool fromZeroList = false;
        // Vector to keep track of the ID of each minimum fill variable
        vector<int> minFillIDs;

        // If there are no variables that, when deleted, add no edges...
        if (zero_list.empty()) {

            // For each unprocessed (non-deleted) variable
            for (int j = 0; j < variables.size(); j++) {
                if (processed[j])
                    continue;
                double curr_min = 0.0;
                for (auto a = graph[j].begin(); a != graph[j].end(); a++) {
                    auto b = a;
                    b++;
                    for (; b != graph[j].end(); b++) {
                        if (!adj_matrix[*a][*b]) {
                            curr_min += (variables[*a]->d
                                         * variables[*b]->d);
                            if (curr_min > min)
                                break;
                        }
                    }
                    if (curr_min > min)
                        break;
                }

                // Store the first non-deleted variable as a potential minimum
                if (first) {
                    minFillIDs.push_back(j);
                    min = curr_min;
                    first = false;
                } else {
                    // If this is a new minimum...
                    if (min > curr_min) {
                        min = curr_min;
                        minFillIDs.clear();
                        minFillIDs.push_back(j);
                    }
                        // Otherwise, if the number of edges removed is also a minimum, but
                        // the minimum is zero
                    else if (curr_min < DBL_MIN) {
                        zero_list.push_back(j);
                    }
                        // Else if this is another potential min_fill
                    else if (min == curr_min) {
                        minFillIDs.push_back(j);
                    }
                }
            }
        }
            // Else...delete variables from graph that don't add any edges
        else {
            min_id = zero_list.front();
            zero_list.pop_front();
            fromZeroList = true;
        }

        // If not from zero_list, choose one of the variables at random
        // from the set of min fill variables
        if (!fromZeroList) {
            int indexInVector;
            indexInVector = myRandom::getInt(minFillIDs.size());
            min_id = minFillIDs[indexInVector];
        }

        //cout<<"order["<<i<<"]= "<<min_id<<" "<<flush;
        assert(min_id!=-1);
        order[i] = min_id;
        // Now form the cluster
        clusters[i] = graph[min_id];
        clusters[i].insert(min_id);

        // Trinagulate min id and remove it from the graph
        for (auto a = graph[min_id].begin();
             a != graph[min_id].end(); a++) {
            auto b = a;
            b++;
            for (; b != graph[min_id].end(); b++) {
                if (!adj_matrix[*a][*b]) {
                    adj_matrix[*a][*b] = true;
                    adj_matrix[*b][*a] = true;
                    graph[*a].insert(*b);
                    graph[*b].insert(*a);
                }
            }
        }
        for (auto a = graph[min_id].begin();
             a != graph[min_id].end(); a++) {
            graph[*a].erase(min_id);
            adj_matrix[*a][min_id] = false;
            adj_matrix[min_id][*a] = false;
        }
        graph[min_id].clear();
        processed[min_id] = true;
    }
    /*
    // compute the estimate
    for (auto & cluster : clusters) {
        if ((int) cluster.size() > max_cluster_size)
            max_cluster_size = (int) cluster.size();
        double curr_estimate = 1.0;
        for (std::__1::__tree_const_iterator<int, std::__1::__tree_node<int, void *> *, long>::value_type j : cluster) {
            curr_estimate *= (double) variables[j]->d;
        }
        estimate += curr_estimate;
    }
    cout<<"Max cluster size = "<<max_cluster_size<<endl;
    */
}

void Utils::computeEntropy(vector<vector<ldouble>> &px, ldouble &entropy){
    entropy = 0.0;
    vector<vector<ldouble>> lpx = vector<vector<ldouble>>(px.size());
    for(int j = 0; j < px.size(); j++){
        lpx[j] = vector<ldouble>(px[j].size());
        for(int xj = 0; xj < px[j].size(); xj++){
            lpx[j][xj] = log(px[j][xj]);
            entropy -= px[j][xj]*lpx[j][xj];
        }
    }
    entropy /= px.size();
}

void Utils::computeMI(vector<vector<vector<vector<ldouble>>>> &pxy, vector<vector<ldouble>> &px, vector<vector<ldouble>> &mi){
    vector<vector<ldouble>> lpx = vector<vector<ldouble>>(px.size());
    for(int j = 0; j < px.size(); j++){
        lpx[j] = vector<ldouble>(px[j].size());
        for(int xj = 0; xj < px[j].size(); xj++){
            lpx[j][xj] = log(px[j][xj]);
        }
    }

    mi = vector<vector<ldouble>>(px.size(), vector<ldouble>(px.size(), 0.0));
    for(int j = 0; j < pxy.size(); j++){
        for(int k = j+1; k < pxy[j].size(); k++){
            for(int xj = 0; xj < pxy[j][k].size(); xj++){
                for(int xk = 0; xk < pxy[j][k][xj].size(); xk++){
                    mi[j][k] += pxy[j][k][xj][xk]*(log(pxy[j][k][xj][xk])-lpx[j][xj]-lpx[k][xk]);
                }
            }
        }
    }
}

void Utils::computePartialMeasures(Data &data, vector<int> &data_indices, vector<int> &features, vector<vector<vector<vector<ldouble>>>> &pxy, vector<vector<ldouble>> &px, vector<vector<vector<vector<ldouble>>>> &cxy, vector<vector<ldouble>> &cx){
    pxy = vector<vector<vector<vector<ldouble>>>>(features.size(), vector<vector<vector<ldouble>>>(features.size()));
    px = vector<vector<ldouble>>(features.size());
    cxy = vector<vector<vector<vector<ldouble>>>>(features.size(), vector<vector<vector<ldouble>>>(features.size()));
    cx = vector<vector<ldouble>>(features.size());
    for(int j = 0; j < features.size(); j++){
        for(int k = 0; k < features.size(); k++){
            pxy[j][k] = vector<vector<ldouble>>(data.dsize[features[j]], vector<ldouble>(data.dsize[features[k]], HyperParameters::laplace));
            cxy[j][k] = vector<vector<ldouble>>(data.dsize[features[j]], vector<ldouble>(data.dsize[features[k]], HyperParameters::laplace));
        }
        px[j] = vector<ldouble>(data.dsize[features[j]], HyperParameters::laplace);
        cx[j] = vector<ldouble>(data.dsize[features[j]], HyperParameters::laplace);
    }
    for(auto &i: data_indices){
        for(int j = 0; j < features.size(); j++){
            cx[j][data.data_matrix[i][features[j]]] += data.weights[i];
            for(int k = 0; k < features.size(); k++){
                cxy[j][k][data.data_matrix[i][features[j]]][data.data_matrix[i][features[k]]] += data.weights[i];
            }
        }
    }
    
    pxy = cxy;
    px = cx;
    for(int j = 0; j < features.size(); j++){
        for(int k = 0; k < features.size(); k++){
            Utils::normalize2d(pxy[j][k]);
        }
        Utils::normalize1d(px[j]);
    }
}

void Utils::poissonUpdateOR(Data &data, vector<int> &data_indices, Variable* var, vector<ldouble> &wts){
    for(auto &i: data_indices){
        wts[data.data_matrix[i][var->id]] += data.weights[i];
    }
}

void Utils::poissonUpdateCPT(Data &dt, vector<int> &indices, Function &func){
    if(func.variables.size() == 1){
        for(auto &i: indices){
            func.count_potentials[dt.data_matrix[i][func.cpt_var->id]] += dt.weights[i];
        }
        func.potentials = vector<ldouble>(func.count_potentials);
        Utils::normalize1d(func.potentials);
        for(auto &val: func.potentials){
            if(val < 0.01) val = 0.01;
            else if(val > 0.99) val = 0.99;
        }
        Utils::normalize1d(func.potentials);
    }
    else if(func.variables.size() == 2){
        Variable *t = func.cpt_var, *s;
        if (func.variables[0]->id == func.cpt_var->id) {
            s = func.variables[1];
        }
        else {
            s = func.variables[0];
        }
        
        for(auto &i: indices){
            s->t_val = dt.data_matrix[i][s->id];
            t->t_val = dt.data_matrix[i][t->id];
            int ind = Utils::getAddr(func.variables);
            func.count_potentials[ind] += dt.weights[i];
        }

        for(int i = 0; i < s->d; i++){
            s->t_val = i;
            ldouble norm_const = 0.0;
            for(int j = 0; j < t->d; j++){
                t->t_val = j;
                norm_const += func.count_potentials[Utils::getAddr(func.variables)];
            }
            for(int j = 0; j < t->d; j++) {
                t->t_val = j;
                int ind = Utils::getAddr(func.variables);
                func.potentials[ind] = func.count_potentials[ind] / norm_const;
            }
        }
        for(auto &val: func.potentials){
            if(val < 0.01) val = 0.01;
            else if(val > 0.99) val = 0.99;
        }
        for(int i = 0; i < s->d; i++){
            s->t_val = i;
            ldouble norm_const = 0.0;
            for(int j = 0; j < t->d; j++){
                t->t_val = j;
                norm_const += func.potentials[Utils::getAddr(func.variables)];
            }
            for(int j = 0; j < t->d; j++) {
                t->t_val = j;
                int ind = Utils::getAddr(func.variables);
                func.potentials[ind] /= norm_const;
            }
        }
    }
}

void Utils::updateCPT(Function &func, const vector<vector<vector<vector<ldouble>>>> &pxy, const vector<vector<ldouble>> &px, const unordered_map<int, int> &varid_ind, const vector<vector<vector<vector<ldouble>>>> &cxy, const vector<vector<ldouble>> &cx){
    if(func.variables.size() == 1){
        func.potentials = px[varid_ind.at(func.variables[0]->id)];
        if (!cx.empty()){
            func.count_potentials = cx[varid_ind.at(func.variables[0]->id)];
        }
        for(auto &val: func.potentials){
            if(val < 0.01) val = 0.01;
            else if(val > 0.99) val = 0.99;
        }
        Utils::normalize1d(func.potentials);
    }
    else if(func.variables.size() == 2){
        Variable *t = func.cpt_var, *s;
        if (func.variables[0]->id == func.cpt_var->id) {
            s = func.variables[1];
        } else {
            s = func.variables[0];
        }

        func.potentials = vector<ldouble>(s->d*t->d, 0.0);
        if(!cxy.empty())
            func.count_potentials = vector<ldouble>(s->d*t->d, 0.0);
        vector<vector<ldouble>> table, counts_table;
        table = pxy[varid_ind.at(s->id)][varid_ind.at(t->id)];
        if(!cxy.empty())
            counts_table = cxy[varid_ind.at(s->id)][varid_ind.at(t->id)];
        for(auto &row: table){
            for(auto &val: row){
                if(val < 0.01) val = 0.01;
                else if(val > 0.99) val = 0.99;
            }
        }
        Utils::normalize2d(table);
        for(int i = 0; i < s->d; i++){
            s->t_val = i;
            ldouble norm_const = 0.0;
            for(int j = 0; j < t->d; j++){
                norm_const += table[i][j];
            }
            for(int j = 0; j < t->d; j++) {
                t->t_val = j;
                func.potentials[Utils::getAddr(func.variables)] = table[i][j] / norm_const;
                if(!cxy.empty())
                    func.count_potentials[Utils::getAddr(func.variables)] = counts_table[i][j];
            }
        }
    }
}

int Utils::getSplittingVar(const vector<vector<ldouble>> &mi, bool randomize){
    int cur_max = -1;
    ldouble cur_max_mi = -100000, temp_mi;
    if(randomize){
        vector<int> rind;
        for(int j = 0; j < mi.size(); j++){
            rind.push_back(j);
        }
        shuffle(begin(rind), end(rind), myRandom::m_g);
        for(auto j = rind.begin(); j <= rind.begin()+rind.size()/2; j++){
            temp_mi = 0;
            for(auto &k: rind){
                if(k != *j){
                    if(k > *j)
                        temp_mi += mi[*j][k];
                    else
                        temp_mi += mi[k][*j];                    
                }
            }
            if(temp_mi > cur_max_mi){
                cur_max_mi = temp_mi;                   
                cur_max = *j;
            }
        }
    }
    else{
        for(int j = 0; j < mi.size(); j++){
            temp_mi = 0;
            for(int k = 0; k < mi.size(); k++){
                if(k != j){
                    if(k > j)
                        temp_mi += mi[j][k];
                    else
                        temp_mi += mi[k][j];                    
                }
            }
            if(temp_mi > cur_max_mi){
                cur_max_mi = temp_mi;
                cur_max = j;
            }
        }
    }
    return cur_max;
}