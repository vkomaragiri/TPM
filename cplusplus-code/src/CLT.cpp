//
// Created by vasundhara on 1/3/20.
//

#include <myRandom.h>
#include <CLT.h>


struct Spanning_Tree {
    vector<Edge> edges;
    Graph g;
    int V;
    vector<vector<int>> adj_list;

    vector<vector<int>> directed_edges;
    vector<int> roots;
    vector<bool> visited;
    //vector<int> features;
    int vertices_visited;

    Spanning_Tree(vector<Edge> &edges_, Graph &g_, int nfeatures_) : edges(edges_), g(g_), V(nfeatures_){
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
        /*
        for(auto &i: features ){
            visited[i] = false;
        }
        
        vertices_visited = V-features.size();
        */
        vertices_visited = 0;
        directed_edges.clear();
        directed_edges = vector<vector<int>> (V);
        roots.clear();

        //for(auto &i: features){
        int i = 0;
        while(vertices_visited != V){
            while(!visited[i]){
                DFS_Recurse(i);
                roots.push_back(i);
            }
            i++;
            //if(vertices_visited == V) break;
        }
        //}
    }
};

void CLT::setEvidence(int var, int val)
{
    variables[var]->setValue(val);
}

void CLT::removeEvidence(int var)
{
    variables[var]->val = -1;
    variables[var]->is_evid = 0;
}

void CLT::learn(Data &data, bool isComp, bool doStructLearning, int r, ldouble laplace, bool part_cn, const vector<vector<ldouble>> &cn_mi, const vector<vector<ldouble>> &cn_px, const vector<vector<vector<vector<ldouble>>>> &cn_pxy, const unordered_map<int, int> &cn_varid_ind, const vector<vector<ldouble>> &cn_cx, const vector<vector<vector<vector<ldouble>>>> &cn_cxy){
    if(!isComp) {
        for (int i = 0; i < data.nfeatures; i++) {
            Variable *var = new Variable(i, data.dsize[i]);
            variables.push_back(var);
        }
        vector<ldouble> weights;
        for(int i  = 0; i < data.nexamples; i++){
            weights.push_back(1);
        }
        data.setWeights(weights);
    }
    if(doStructLearning) {
        if(!part_cn)
            data.computeMI(laplace);
        Graph g;
        if(r > 0){
            vector<int> rind;
            for(int j = 0; j < variables.size(); j++){
                rind.push_back(j);
            }
            shuffle(begin(rind), end(rind), myRandom::m_g);
            for (int i = 0; i < rind.size()-r*rind.size(); i++) {
                int u = rind[i];
                for (int j = i+1; j < rind.size()-r*rind.size(); j++) {
                    int v = rind[j];
                    if(part_cn)
                        boost::add_edge(u, v, -cn_mi[u][v], g);
                    else
                        boost::add_edge(u, v, -data.mi[u][v], g);
                }
            }
        }
        else{
            for (int i = 0; i < variables.size(); i++) {
                for (int j = i+1; j < variables.size(); j++) {
                    if(part_cn)
                        boost::add_edge(i, j, -cn_mi[i][j], g);
                    else
                        boost::add_edge(i, j, -data.mi[i][j], g);
                }
            }

        }
        vector<Edge> spanning_tree;
        boost::property_map<Graph, boost::edge_weight_t>::type weight = get(boost::edge_weight, g);
        boost::kruskal_minimum_spanning_tree(g, back_inserter(spanning_tree));
        Spanning_Tree st = Spanning_Tree(spanning_tree, g, variables.size());
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
                if(i < st.directed_edges[i][j]) {
                    func.variables.push_back(variables[i]);
                    func.variables.push_back(variables[st.directed_edges[i][j]]);
                }
                else{
                    func.variables.push_back(variables[st.directed_edges[i][j]]);
                    func.variables.push_back(variables[i]);
                }
                func.cpt_var = variables[st.directed_edges[i][j]];
                functions.push_back(func);
            }
        }
    }
    if(part_cn){
        for(auto &func: functions) {
            Utils::updateCPT(func, cn_pxy, cn_px, cn_varid_ind, cn_cxy, cn_cx);
        }
    }
    else{
        for(auto &func: functions){
            Utils::updateCPT(func, data, doStructLearning);
        }
    }
}

ldouble CLT::log_likelihood(Data &data){
    ldouble res = 0;
    for(int m = 0; m < data.data_matrix.size(); m++){
        for(auto &func: functions){
            for(int i = 0; i < func.variables.size(); i++){
                func.variables[i]->t_val = data.data_matrix[m][func.variables[i]->id];
            }
            int ind = Utils::getAddr(func.variables);
            if(func.potentials[ind] == 0){
                for(int i = 0; i < func.variables.size(); i++){
                    cout << func.variables[i]->id << " " << data.data_matrix[m][func.variables[i]->id] <<", ";
                }
                cout << ind << endl;
                Utils::printVarVector(func.variables);
                Utils::print1d(func.potentials);
            }
            res += log(func.potentials[ind]);
        }
    }
    res /= data.data_matrix.size();
    return res;
}

ldouble CLT::getProbability(vector<int> &example) {
    /*
    ldouble res = 1.0;
    for(int i = 0; i < example.size(); i++){
        variables[i]->t_val = example[i];
    }
    for(auto &func: functions){
        res *= func.potentials[Utils::getAddr(func.variables)];
    }
    return res;
    */
    return exp(getLogProbability(example));
}

ldouble CLT::getLogProbability(vector<int> &example) {
    ldouble res = 0.0;
    for(int i = 0; i < variables.size(); i++){
        variables[i]->t_val = example[variables[i]->id];
    }
    for(auto &func: functions){
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
        int k = 0, l = 0;
        vector<Variable*> other_variables;
        copy(func_variables.begin(), func_variables.end()-1, back_inserter(other_variables)); 
        for(int t = 0; t < tab_size/functions[j].cpt_var->d; t++){
            ldouble norm_const = 0.0;
            for(int d = 0; d < functions[j].cpt_var->d; d++){
                in >> table[k];
                if(table[k] < 0.01)
                    table[k] = 0.01;
                else if(table[k] > 0.99)
                    table[k] = 0.99;

                if(table[k] < 0 || table[k] > 1.0)
                    cout << "Invalid value in function" << endl;
                norm_const += table[k];
                k++;
            }
            for(int d = 0; d < functions[j].cpt_var->d; d++){
                table[l] /= norm_const;
                l++;
            }
        }
        functions[j].variables=func_variables;
    }
    in.close();
}


void CLT::writeCN(ofstream &out) {
    out << "BAYES" << endl;

    out << this -> functions.size() << "\n";
    for (auto &function : this->functions) {
        out << function.variables.size() << " ";
        for (auto &variable : function.variables) {
            out << variable->id << " ";
        }
        out << function.cpt_var->id << endl;
    }
    for (auto &function : this->functions) {
        out << function.potentials.size() << "\n";
        for (long double k : function.potentials) {
            out << k << " ";
        }
        out << endl;
    }
}

void CLT::writeCNCounts(ofstream &out) {
    out << "BAYES" << endl;

    out << this -> functions.size() << "\n";
    for (auto &function : this->functions) {
        out << function.variables.size() << " ";
        for (auto &variable : function.variables) {
            out << variable->id << " ";
        }
        out << function.cpt_var->id << endl;
    }
    for (auto &function : this->functions) {
        out << function.count_potentials.size() << "\n";
        for (long double k : function.count_potentials) {
            out << k << " ";
        }
        out << endl;
    }
}

void CLT::readCN(ifstream &in, const unordered_map<int, int> &cn_varid_ind) {
    string mystr;
    in >> mystr;
    if (mystr.find("BAYES") == string::npos) {
        cerr << "Something wrong with the file, cannot read\n";
        exit(-1);
    }
    int nfuncs;
    in >> nfuncs;
    functions = vector<Function>(nfuncs);
    for (int j = 0; j < nfuncs; j++) {
        int nvars;
        in >> nvars;
        functions[j].variables = vector<Variable *>(nvars);
        for (int k = 0; k < nvars; k++) {
            int id;
            in >> id;
            functions[j].variables[k] = variables[cn_varid_ind.at(id)];
            //if (k == nvars - 1) functions[j].cpt_var = variables[cn_varid_ind.at(id)];
        }
        int cpt_varid;
        in >> cpt_varid;
        functions[j].cpt_var = variables[cn_varid_ind.at(cpt_varid)];
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
}

void CLT::readCNCounts(ifstream &in, const unordered_map<int, int> &cn_varid_ind) {
    string mystr;
    in >> mystr;
    if (mystr.find("BAYES") == string::npos) {
        cerr << "Something wrong with the file, cannot read\n";
        exit(-1);
    }
    int nfuncs;
    in >> nfuncs;
    functions = vector<Function>(nfuncs);
    for (int j = 0; j < nfuncs; j++) {
        int nvars;
        in >> nvars;
        functions[j].variables = vector<Variable *>(nvars);
        for (int k = 0; k < nvars; k++) {
            int id;
            in >> id;
            functions[j].variables[k] = variables[cn_varid_ind.at(id)];
            //if (k == nvars - 1) functions[j].cpt_var = variables[cn_varid_ind.at(id)];
        }
        int cpt_varid;
        in >> cpt_varid;
        functions[j].cpt_var = variables[cn_varid_ind.at(cpt_varid)];
    }
    for (int j = 0; j < nfuncs; j++) {
        vector<Variable*> func_variables=functions[j].variables;
        vector<ldouble> &table = functions[j].count_potentials;
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
}

void CLT::normalizeParams(){
    for(auto &func: functions){
        func.potentials = vector<ldouble>(func.count_potentials);
        if(func.variables.size() == 1){
            Utils::normalize1d(func.potentials);
            for(auto &val: func.potentials){
                if(val < 0.01) val = 0.01;
                else if(val > 0.99) val = 0.99;
            }
            Utils::normalize1d(func.potentials);
        }
        else{
            Variable *t = func.cpt_var, *s;
            if (func.variables[0]->id == func.cpt_var->id) {
                s = func.variables[1];
            } else {
                s = func.variables[0];
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
}

void CLT::initGrad(){
    gradients = vector<vector<ldouble>>(functions.size());
    for(int i = 0; i < functions.size(); i++){
        gradients[i] = vector<ldouble>(functions[i].potentials.size(), 0.0);
    }
}

void CLT::compGrad(vector<vector<int>> &data, vector<int> &partition){
    for(auto &i: partition){
        for(auto &var: variables){
            var->t_val = data[i][var->id];
        }
        for(int j = 0; j < functions.size(); j++){
            int t = Utils::getAddr(functions[j].variables);
            gradients[j][t] += 1.0/functions[j].potentials[t];
        }
    }
}

void CLT::doSGDUpdate(ldouble learning_rate){
    for(int i = 0; i < functions.size(); i++){
        int ind = find(functions[i].variables.begin(), functions[i].variables.end(), functions[i].cpt_var)-functions[i].variables.begin();
        int all_size = functions[i].potentials.size();
        vector<Variable*> other_variables(functions[i].variables);
        other_variables.erase(other_variables.begin()+ind);
        int other_size = all_size/functions[i].cpt_var->d;
        vector<ldouble> new_potential;
        for(int j = 0; j < other_size; j++){
            Utils::setAddr(other_variables, j);
            vector<ldouble> prob;
            for(int k = 0; k < functions[i].cpt_var->d; k++){
                functions[i].cpt_var->t_val = k;
                int t = Utils::getAddr(functions[i].variables);
                prob.emplace_back(functions[i].potentials[t]+learning_rate*gradients[i][t]);
            }
            Utils::normalize1d(prob);
            copy(prob.begin(), prob.end(), back_inserter(new_potential));
        }
        functions[i].potentials = new_potential;
    }
}

ldouble CLT::gradSqNorm(){
    ldouble res = 0.0;
    for(auto &grad: gradients){
        for(auto &val: grad){
            res += val*val;
        }
    }
    return res;
}