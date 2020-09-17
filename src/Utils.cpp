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
        if(!doStructLearning) {
            vector<ldouble> table;
            Utils::getXStat(func.variables[0], data, table);
            func.potentials = table;
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
        vector<vector<ldouble>> table;
        if(!doStructLearning) {
            Utils::getXYStat(s, t, data, table);
        }
        else{
            table = data.pxy[s->id][t->id];
        }
        vector<ldouble> potentials_;
        for(vector<ldouble> row: table){
            for(auto &val: row){
                if(val < 0.01) val = 0.01;
                else if(val > 0.99) val = 0.99;
            }
            Utils::normalize1d(row);
            copy(row.begin(), row.end(), back_inserter(potentials_));
        }
        func.potentials = potentials_;
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
void Utils::getOrder(vector<Variable *> &variables, vector<Function> &functions, vector<int> &order) {
    if (HyperParameters::ord_heu==min_fill){
        Utils::getMinFillOrder(variables,functions,order);
    } else{
        if (HyperParameters::ord_heu==min_degree){
            Utils::getMinDegreeOrder(variables,functions,order);
        }
        else {
            if(HyperParameters::ord_heu==topological){
                Utils::getTopologicalOrder(variables,functions,order);
            }
        }
    }
}

void Utils::getMinDegreeOrder(vector<Variable *> &variables, vector<Function> &functions, vector<int> &order) {
    vector<set<int> > degree(variables.size());
    vector<set<int> > graph(variables.size());

    for (auto &function : functions) {
        for (int j = 0; j < function.variables.size(); j++) {
            if (function.variables[j]->is_evid) continue;
            for (int k = j + 1; k < function.variables.size(); k++) {
                if (function.variables[k]->is_evid) continue;
                int a = function.variables[j]->id;
                int b = function.variables[k]->id;
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

void Utils::getTopologicalOrder(vector<Variable *> &variables, vector<Function> &functions, vector<int> &order) {
    vector<set<int> > degree(variables.size());
    vector<set<int> > parents(variables.size());
    vector<set<int> > children(variables.size());

    for (auto function : functions) {
        if(function.cpt_var==nullptr){
            cerr<<"Something wrong: Topological order for a non Bayes net\n";
            exit(-1);
        }
        int a=function.cpt_var->id;
        for (int j = 0; j < function.variables.size(); j++) {
            if (function.variables[j]->id==function.cpt_var->id) continue;
            int b = function.variables[j]->id;
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

void Utils::getMinFillOrder(vector<Variable *> &variables, vector<Function> &functions, vector<int> &order) {
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
                int a = function.variables[j]->id;
                int b = function.variables[k]->id;
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
                for (auto a = graph[j].begin();
                     a != graph[j].end(); a++) {
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






