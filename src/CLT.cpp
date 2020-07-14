//
// Created by vasundhara on 1/3/20.
//

#include <myRandom.h>
#include "../include/CLT.h"


struct Spanning_Tree {
    vector<Edge> edges;
    Graph g;
    int V;
    vector<vector<int>> adj_list;

    vector<vector<int>> directed_edges;
    vector<int> roots;
    vector<bool> visited;
    int vertices_visited;

    Spanning_Tree(vector<Edge> &edges_, Graph &g_, int nfeatures_) : edges(edges_), g(g_), V(nfeatures_) {
        adj_list = vector<vector<int>> (V);
    }

    void edgeListToAdjList(){
        for (auto &ei : edges) {
            int v1 = boost::source(ei, g);
            int v2 = boost::target(ei, g);
            adj_list[v1].push_back(v2);
            adj_list[v2].push_back(v1);
        }
    }

    void DFS_Recurse(int v){
        visited[v] = true;
        vertices_visited++;
        for(auto v2: adj_list[v]){
            if(!visited[v2]){
                directed_edges[v].push_back(v2);
                DFS_Recurse(v2);
            }
        }
    }

    void DFS(){
        visited.clear();
        visited = vector<bool> (V, false);
        vertices_visited = 0;

        directed_edges.clear();
        directed_edges = vector<vector<int>> (V);
        roots.clear();

        int i = 0;
        while(vertices_visited != V){
            while(!visited[i]){
                DFS_Recurse(i);
                roots.push_back(i);
            }
            i++;
        }
    }
};

void CLT::learn(Data &data, vector<ldouble> &weights_, bool isComp, bool doStructLearning, int r){
    data.setWeights(weights_);
    if(!isComp) {
        for (int i = 0; i < data.nfeatures; i++) {
            Variable *var = new Variable(i, data.dsize[i]);
            variables.push_back(var);
        }
    }
    if(doStructLearning) {
        data.computeMI();
        Graph g;
        for (int i = 0; i < data.nfeatures; i++) {
            for (int j = i + 1; j < data.nfeatures; j++) {
                if(r > 0){
                    double p = myRandom::getDouble();
                    if((p-(int)p) >= 0.5) {
                        r--;
                    }
                    else{
                        boost::add_edge(i, j, -data.mi[i][j], g);
                    }
                }
                else{
                    boost::add_edge(i, j, -data.mi[i][j], g);
                }
            }
        }
        vector<Edge> spanning_tree;
        boost::property_map<Graph, boost::edge_weight_t>::type weight = get(boost::edge_weight, g);
        boost::kruskal_minimum_spanning_tree(g, back_inserter(spanning_tree));
        Spanning_Tree st = Spanning_Tree(spanning_tree, g, data.nfeatures);
        st.edgeListToAdjList();
        st.DFS();
        functions.clear();// = vector<Function>(st.roots.size()+spanning_tree.size());
        for (int i = 0; i < st.roots.size(); i++) {
            Function func = Function();
            func.variables.push_back(variables[st.roots[i]]);
            func.cpt_var = variables[st.roots[i]];
            functions.push_back(func);
        }

        for (int i = 0; i < st.directed_edges.size(); i++) {
            for (int j = 0; j < st.directed_edges[i].size(); j++) {
                Function func = Function();
                func.variables.push_back(variables[i]);
                func.variables.push_back(variables[st.directed_edges[i][j]]);
                func.cpt_var = variables[st.directed_edges[i][j]];
                functions.push_back(func);
            }
        }
    }
    for (auto &func: functions) {
        Utils::updateCPT(func, data, doStructLearning);
    }
}

ldouble CLT::log_likelihood(Data &data){
    ldouble res = 0;
    for(int m = 0; m < data.data_matrix.size(); m++){
        for(auto func: functions){
            for(int i = 0; i < func.variables.size(); i++){
                func.variables[i]->t_val = data.data_matrix[m][func.variables[i]->id];
            }
            int ind = Utils::getAddr(func.variables);
            res += log(func.potentials[ind]);
        }
    }
    res /= data.data_matrix.size();
    return res;
}

ldouble CLT::getProbability(vector<int> example) {
    ldouble res = 1.0;
    for(int i = 0; i < example.size(); i++){
        variables[i]->t_val = example[i];
    }
    for(auto func: functions){
        res *= func.potentials[Utils::getAddr(func.variables)];
    }
    return res;
}

ldouble CLT::getLogProbability(vector<int> example) {
    ldouble res = 0.0;
    for(int i = 0; i < example.size(); i++){
        variables[i]->t_val = example[i];
    }
    for(auto func: functions){
        res += log(func.potentials[Utils::getAddr(func.variables)]);
    }
    return res;
}

void CLT::print() {
    for(int i = 0; i < functions.size(); i++){
        for(auto var: functions[i].variables){
            cout << var->id << " ";
        }
        cout << endl;
        Utils::print1d(functions[i].potentials);
        cout << endl;
    }
}

void CLT::write(string infile) {
    ofstream out(infile);
    out << "BAYES" << endl;

    out << this->variables.size() << endl;
    for (auto &variable : variables) {
        out << variable->d << " ";
    }
    out << endl;

    out << this -> functions.size() << "\n";
    for (auto &function : this->functions) {
        out << function.variables.size() << " ";
        for (auto &variable : function.variables) {
            out << variable->id << " ";
        }
        out << endl;
    }
    for (auto &function : this->functions) {
        out << function.potentials.size() << "\n";
        for (long double k : function.potentials) {
            out << k << " ";
        }
        out << endl;
    }
    out.close();
}

void CLT::readUAI08(string infile) {
    ifstream in(infile);
    string mystr;
    in >> mystr;
    if (mystr.find("BAYES") == string::npos) {
        cerr << "Something wrong with the file, cannot read\n";
        exit(-1);
    }
    int numvars;
    in >> numvars;
    variables = vector<Variable *>(numvars);
    for (int i = 0; i < numvars; i++) {
        int d;
        in >> d;
        variables[i] = new Variable(i, d);
    }
    int nfuncs;
    in >> nfuncs;
    functions = vector<Function>(nfuncs);
    for (int j = 0; j < nfuncs; j++) {
        int nvars;
        in >> nvars;
        functions[j].variables = vector<Variable *>(nvars);
        for (int k = 0; k < nvars; k++) {
            int varid;
            in >> varid;
            functions[j].variables[k] = variables[varid];
            if (k == nvars - 1) functions[j].cpt_var = variables[varid];
        }
    }
    for (int j = 0; j < nfuncs; j++) {
        vector<Variable*> func_variables=functions[j].variables;
        vector<ldouble> &table = functions[j].potentials;
        int tab_size;
        in >> tab_size;
        table = vector<ldouble>(tab_size);
        for (int k = 0; k < tab_size; k++) {
            Utils::setAddr(functions[j].variables,k);
            int d=Utils::getAddr(func_variables);
            in >> table[d];
        }
        functions[j].variables=func_variables;
    }
    in.close();
}
