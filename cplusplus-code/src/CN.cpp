#include <CN.h>

void CN::learn(Data &train_data, Data &valid_data, bool prune, bool randomize, bool isComponent, int max_depth){
    vector<int> data_indices, features;
    if(!isComponent)
        variables = vector<Variable*>(train_data.nfeatures);
    for(int j = 0; j < train_data.nfeatures; j++){
        features.push_back(j);
        if(!isComponent)
            variables[j] = new Variable(j, train_data.dsize[j]);
    }
    vector<ldouble> weights;
    for(int i = 0; i < train_data.nexamples; i++){
        data_indices.push_back(i);
        weights.push_back(1);
    }
    if(!isComponent){
        train_data.setWeights(weights);
    }
    root = learnCNode(train_data, data_indices, features, 0, randomize, max_depth);
    if(prune){
        pruneCNode(root, train_data, valid_data);
    }
}

CNode* CN::learnCNode(Data &data, vector<int> &data_indices, vector<int> &features, int depth, bool randomize, int max_depth){
    CNode* out = new CNode();
    if(features.size() <= 0){
        out = nullptr;
        return out;
    }
    Utils::computePartialMeasures(data, data_indices, features, out->pxy, out->px, out->cxy, out->cx);
    out->features = features;
    Utils::computeEntropy(out->px, out->entropy);
    Utils::computeMI(out->pxy, out->px, out->mi);
    if(termination_condition(data_indices.size(), out->entropy, depth, max_depth)){
        out->type = 1;
        out->clt = new CLT();
        out->clt->variables = vector<Variable*>(features.size());
        for(int j = 0; j < features.size(); j++){
            out->clt->variables[j] = variables[features[j]];
            out->varid_ind[features[j]] = j;
        }
        out->clt->learn(data, true, true, 0, HyperParameters::laplace, true, out->mi, out->px, out->pxy, out->varid_ind, out->cx, out->cxy);
    }
    else{
        out->type = 0;
        int var_ind = Utils::getSplittingVar(out->mi, randomize);
        out->var = variables[features[var_ind]];
        out->child_weights = out->px[var_ind];
        for(auto &val: out->child_weights){
            if(val < 0.01) val = 0.01;
            else if(val > 0.99) val = 0.99;
        }
        Utils::normalize1d(out->child_weights);
        out->count_child_weights = out->cx[var_ind];
        out->children = vector<CNode*>(out->var->d);
        vector<vector<int>> child_data_indices = vector<vector<int>>(out->var->d);
        for(auto &i: data_indices){
            child_data_indices[data.data_matrix[i][out->var->id]].push_back(i);
        }
        vector<int> temp;
        for(auto &j: features){
            if(j != out->var->id){
                temp.push_back(j);
            }
        }
        vector<vector<int>> child_features = vector<vector<int>>(out->var->d, temp);
        out->children = vector<CNode*>(out->var->d);
        for(int k = 0; k < out->var->d; k++){
            out->children[k] = learnCNode(data, child_data_indices[k], child_features[k], depth+1, randomize, max_depth);
        }
    }
    return out;
}

bool CN::termination_condition(int nexamples, ldouble entropy, int depth, int max_depth){
    if(nexamples < 10 || entropy < 0.01 || depth >= max_depth)
        return true;
    return false;
}

void CN::pruneCNode(CNode* nd, Data &train_data, Data &valid_data){
    if(nd && nd->type == 0){
        for(int j = 0; j < nd->children.size(); j++){
            pruneCNode(nd->children[j], train_data, valid_data);
        }
        ldouble cur_ll = log_likelihood(valid_data), new_ll = 0.0;
        nd->type = 1;
        nd->clt = new CLT();
        nd->clt->variables = vector<Variable*>(nd->features.size());
        for(int j = 0; j < nd->features.size(); j++){
            nd->clt->variables[j] = variables[nd->features[j]];
            nd->varid_ind[nd->features[j]] = j;
        }
        nd->clt->learn(train_data, true, true, 0, HyperParameters::laplace, true, nd->mi, nd->px, nd->pxy, nd->varid_ind, nd->cx, nd->cxy);
        new_ll = log_likelihood(valid_data);
        if(new_ll < cur_ll){
            nd->type = 0;
            nd->clt = nullptr;
            nd->varid_ind = unordered_map<int, int>();
        }
        else{
            nd->children = vector<CNode*>();
            nd->child_weights = vector<ldouble>();
            nd->count_child_weights = vector<ldouble>();
            nd->var = nullptr;
        }
    }
}

ldouble CN::log_likelihood(Data &data){
    ldouble out = 0.0;
    for(auto &i: data.data_matrix){
        out += getLogProb(i);
    }
    out /= data.nexamples;
    return out;
}

ldouble CN::getLogProb(vector<int> &example){
    ldouble out = 0.0;
    getLogProbCNode(root, example, out);
    return out;
}

void CN::getLogProbCNode(CNode* nd, vector<int> &example, ldouble &out){
    if(nd){
        if(nd->type == 0){
            out += log(nd->child_weights[example[nd->var->id]]);
            getLogProbCNode(nd->children[example[nd->var->id]], example, out);
        }
        else{
            out += nd->clt->getLogProbability(example);
        }
    }
}

ldouble CN::getProb(vector<int> &example){
    return exp(getLogProb(example));
}

void CN::print(){
    printCNNode(root);
    cout << endl;
}

void CN::printCNNode(CNode* nd){
    if(nd){
        cout << "[";
        if(nd->type == 0){
            cout << "OR," << nd->var->id << ",";
            for(int k = 0; k < nd->children.size(); k++){
                cout << "(";
                printCNNode(nd->children[k]);
                cout << ")";
            }
        }
        else{
            cout << "CLT,";
            Utils::printVarVector(nd->clt->variables);
            cout << "...";
            nd->clt->print();
        }
        cout << "]" << endl;
    }
}

void CN::write(string infile){
    ofstream out(infile);
    out << "CN" << endl;

    out << this->variables.size() << endl;
    for (auto &variable : variables) {
        out << variable->d << " ";
    }
    out << endl;

    writeCNode(root, out);
    out.close();
}

void CN::writeCounts(string infile){
    ofstream out(infile);
    out << "CN" << endl;

    out << this->variables.size() << endl;
    for (auto &variable : variables) {
        out << variable->d << " ";
    }
    out << endl;

    writeCNodeCounts(root, out);
    out.close();
}

void CN::writeCNode(CNode* nd, ofstream &out){
    out << "BEGIN" << endl;
    if(nd){
        if(nd->type == 0)
            out << "OR" << endl;
        else
            out << "CLT" << endl;
        out << nd->features.size() << endl;
        for(auto &j: nd->features){
            out << j << " ";
        }
        out << endl;
        if(nd->type == 0){
            out << nd->var->id << endl;
            for(int k = 0; k < nd->children.size(); k++){
                out << nd->child_weights[k] << " ";
            }
            out << endl;
            for(int k = 0; k < nd->children.size(); k++){
                writeCNode(nd->children[k], out);
            }
        }
        else{
            nd->clt->writeCN(out);
        }
    }
    else{
        out << "NULL" << endl;
    }
    out << "END" << endl;
}

void CN::writeCNodeCounts(CNode* nd, ofstream &out){
    out << "BEGIN" << endl;
    if(nd){
        if(nd->type == 0)
            out << "OR" << endl;
        else
            out << "CLT" << endl;
        out << nd->features.size() << endl;
        for(auto &j: nd->features){
            out << j << " ";
        }
        out << endl;
        if(nd->type == 0){
            out << nd->var->id << endl;
            for(int k = 0; k < nd->children.size(); k++){
                out << nd->count_child_weights[k] << " ";
            }
            out << endl;
            for(int k = 0; k < nd->children.size(); k++){
                writeCNodeCounts(nd->children[k], out);
            }
        }
        else{
            nd->clt->writeCNCounts(out);
        }
    }
    else{
        out << "NULL" << endl;
    }
    out << "END" << endl;
}

void CN::read(string infile){
    ifstream in(infile);
    string mystr;
    in >> mystr;
    if (mystr.find("CN") == string::npos) {
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
    root = readCNode(in);
}


void CN::readCounts(string infile){
    ifstream in(infile);
    string mystr;
    in >> mystr;
    if (mystr.find("CN") == string::npos) {
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
    root = readCNodeCounts(in);
}

CNode* CN::readCNode(ifstream &in){
    string mystr;
    in >> mystr;
    if (mystr.find("BEGIN") == string::npos) {
        cerr << "Something wrong with the file, cannot read\n";
        exit(-1);
    }
    in >> mystr;
    CNode* nd = new CNode();
    if(mystr.find("NULL") != string::npos){
        nd = nullptr;
    }
    else if(mystr.find("OR") != string::npos){
        nd->type = 0;
        int nfeatures;
        in >> nfeatures;
        nd->features = vector<int>(nfeatures);
        for(int j = 0; j < nfeatures; j++){
            int t;
            in >> t;
            nd->features[j] = t;
        }
        int label;
        in >> label;
        nd->var = variables[label];

        nd->child_weights = vector<ldouble> (nd->var->d);
        for(int k = 0; k < nd->var->d; k++){
            ldouble t;
            in >> t;
            nd->child_weights[k] = t;
        }
        nd->children = vector<CNode*>(nd->var->d);
        for(int k = 0; k < nd->var->d; k++){
            nd->children[k] = readCNode(in);
        }
    }
    else if(mystr.find("CLT") != string::npos){
        nd->type = 1;//cout << "CLT" << endl;
        int nfeatures;
        in >> nfeatures;
        nd->features = vector<int>(nfeatures);
        nd->clt = new CLT();
        nd->clt->variables = vector<Variable *>(nfeatures);
        for(int j = 0; j < nfeatures; j++){
            int t;
            in >> t;
            nd->features[j] = t;
            nd->clt->variables[j] = variables[t];
            nd->varid_ind[t] = j;
        }
        nd->clt->readCN(in, nd->varid_ind);
    }
    else{
        cerr << "Something wrong with the file, cannot read\n";
        exit(-1);
    }
    in >> mystr;
    if (mystr.find("END") == string::npos) {
        cerr << "Something wrong with the file, cannot read\n";
        exit(-1);
    }
    return nd;
}

CNode* CN::readCNodeCounts(ifstream &in){
    string mystr;
    in >> mystr;
    if (mystr.find("BEGIN") == string::npos) {
        cerr << "Something wrong with the file, cannot read\n";
        exit(-1);
    }
    in >> mystr;
    CNode* nd = new CNode();
    if(mystr.find("NULL") != string::npos){
        nd = nullptr;
    }
    else if(mystr.find("OR") != string::npos){
        nd->type = 0;
        int nfeatures;
        in >> nfeatures;
        nd->features = vector<int>(nfeatures);
        for(int j = 0; j < nfeatures; j++){
            int t;
            in >> t;
            nd->features[j] = t;
        }
        int label;
        in >> label;
        nd->var = variables[label];

        nd->count_child_weights = vector<ldouble> (nd->var->d);
        for(int k = 0; k < nd->var->d; k++){
            ldouble t;
            in >> t;
            nd->count_child_weights[k] = t;
        }
        nd->children = vector<CNode*>(nd->var->d);
        for(int k = 0; k < nd->var->d; k++){
            nd->children[k] = readCNodeCounts(in);
        }
    }
    else if(mystr.find("CLT") != string::npos){
        nd->type = 1;//cout << "CLT" << endl;
        int nfeatures;
        in >> nfeatures;
        nd->features = vector<int>(nfeatures);
        nd->clt = new CLT();
        nd->clt->variables = vector<Variable *>(nfeatures);
        for(int j = 0; j < nfeatures; j++){
            int t;
            in >> t;
            nd->features[j] = t;
            nd->clt->variables[j] = variables[t];
            nd->varid_ind[t] = j;
        }
        nd->clt->readCNCounts(in, nd->varid_ind);
    }
    else{
        cerr << "Something wrong with the file, cannot read\n";
        exit(-1);
    }
    in >> mystr;
    if (mystr.find("END") == string::npos) {
        cerr << "Something wrong with the file, cannot read\n";
        exit(-1);
    }
    return nd;
}


void CN::compGrad(vector<vector<int>> &data, vector<int> &partition, CNode* nd){
    if(nd->type == 0){
        vector<vector<int>> child_partitons = vector<vector<int>>(nd->children.size());
        for(auto &i: partition){
            nd->grad_child_weights[data[i][nd->var->id]] += 1.0/nd->child_weights[data[i][nd->var->id]];
            child_partitons[data[i][nd->var->id]].push_back(i);
        }
        for(int i = 0; i < nd->children.size(); i++){
            compGrad(data, child_partitons[i], nd->children[i]);
        }
    }
    else{
        nd->clt->compGrad(data, partition);
    }
}

void CN::doSGDUpdateCNode(CNode* nd, ldouble learning_rate){
    if(nd->type == 0){
        for(int i = 0; i < nd->children.size(); i++){
            nd->child_weights[i] += learning_rate*nd->grad_child_weights[i];
            doSGDUpdateCNode(nd->children[i], learning_rate);
        }
        Utils::normalize1d(nd->child_weights);
    }
    else{
        nd->clt->doSGDUpdate(learning_rate);
    }
}

void CN::doSGDUpdate(ldouble learning_rate){
    doSGDUpdateCNode(root, learning_rate);
}

void CN::initGradCNode(CNode* nd){
    if(nd->type == 0){
        nd->grad_child_weights = vector<ldouble>(nd->child_weights.size(), 0.0);
        for(int i = 0; i < nd->children.size(); i++){
            initGradCNode(nd->children[i]);
        }
    }
    else{
        nd->clt->initGrad();
    }
}

void CN::initGrad(){
    initGradCNode(root);
}

ldouble CN::gradSqNorm(CNode* nd){
    ldouble res = 0.0;
    if(nd->type == 0){
        for(auto &val: nd->grad_child_weights){
            res += val*val;
        }
    }
    else{
        res += nd->clt->gradSqNorm();
    }
    return res;
}

void CN::normalizeParams(){
    normalizeParamsCNode(root);
}

void CN::normalizeParamsCNode(CNode* nd){
    if(nd->type == 0){
        nd->child_weights = vector<ldouble>(nd->count_child_weights);
        Utils::normalize1d(nd->child_weights);
        for(auto &val:nd->child_weights){
            if(val < 0.01) val = 0.01;
            else if(val > 0.99) val = 0.99;
        }
        Utils::normalize1d(nd->child_weights);
        for(auto &child: nd->children){
            normalizeParamsCNode(child);
        }
    }
    else{
        nd->clt->normalizeParams();
    }
}

void CN::poissonOnlineLearnCNode(CNode* nd, Data &dt, vector<int> &indices, int depth){//}, bool paramsOnly, int depth){
        if(nd->type == 0){
            int var_ind;
            for(int j = 0; j < nd->features.size(); j++){
                if(nd->var->id == nd->features[j]){
                    var_ind = j;
                    break;
                }
            }
            Utils::poissonUpdateOR(dt, indices, nd->var, nd->count_child_weights);
            nd->child_weights = vector<ldouble>(nd->count_child_weights);
            Utils::normalize1d(nd->child_weights);
            vector<vector<int>> child_data_indices = vector<vector<int>>(nd->var->d);
            for(auto &i: indices){
                child_data_indices[dt.data_matrix[i][nd->var->id]].push_back(i);
            }
            for(int k = 0; k < nd->var->d; k++){
                poissonOnlineLearnCNode(nd->children[k], dt, child_data_indices[k], depth+1);
            }
        }
        else{
            for(auto &func: nd->clt->functions) {
                Utils::poissonUpdateCPT(dt, indices, func);
                //Code...
                //Utils::updateCPT(func, nd->pxy, nd->px, nd->varid_ind);
            }
        }

    /*
    else{
        Utils::onlinePartialMeasures(dt, indices, nd->features, nd->pxy, nd->px, nd->mi, nd->entropy, nd->cxy, nd->cx, true);
        if(nd->type == 0){
            if(termination_condition(indices.size(), nd->entropy, depth)){
                nd->type = 1;
                nd->clt = new CLT();
                nd->clt->variables = vector<Variable*>(nd->features.size());
                for(int j = 0; j < nd->features.size(); j++){
                    nd->clt->variables[j] = variables[nd->features[j]];
                    nd->varid_ind[nd->features[j]] = j;
                }
                nd->clt->learn(dt, true, true, 0, HyperParameters::laplace, true, nd->mi, nd->px, nd->pxy, nd->varid_ind);
            }
            else{
                int new_split = Utils::getSplittingVar(nd->mi, true);
                nd->child_weights = nd->px[new_split];
                vector<vector<int>> child_data_indices = vector<vector<int>>(nd->var->d);
                for(auto &i: indices){
                    child_data_indices[dt.data_matrix[i][nd->var->id]].push_back(i);
                }

                if(nd->features[new_split] == nd->var->id){
                    for(int k = 0; k < nd->var->d; k++){
                        poissonOnlineLearnCNode(nd->children[k], dt, child_data_indices[k], paramsOnly, depth+1);
                    }
                }
                else{
                    nd->var = variables[nd->features[new_split]];
                    nd->children = vector<CNode*>(nd->var->d);
                    
                    vector<int> temp;
                    for(auto &j: nd->features){
                        if(j != nd->var->id){
                            temp.push_back(j);
                        }
                    }
                    vector<vector<int>> child_features = vector<vector<int>>(nd->var->d, temp);
                    nd->children = vector<CNode*>(nd->var->d);
                    for(int k = 0; k < nd->var->d; k++){
                        nd->children[k] = learnCNode(dt, child_data_indices[k], child_features[k], depth+1, true);
                    }
                }
            }
        }
        else{
            if(termination_condition(indices.size(), nd->entropy, depth)){
                nd->clt->learn(dt, true, true, 0, HyperParameters::laplace, true, nd->mi, nd->px, nd->pxy, nd->varid_ind);
            }
            else{
                nd->type = 0;
                int var_ind = Utils::getSplittingVar(nd->mi, true);
                nd->var = variables[nd->features[var_ind]];
                nd->child_weights = nd->px[var_ind];
                nd->children = vector<CNode*>(nd->var->d);
                
                vector<vector<int>> child_data_indices = vector<vector<int>>(nd->var->d);
                for(auto &i: indices){
                    child_data_indices[dt.data_matrix[i][nd->var->id]].push_back(i);
                }
                vector<int> temp;
                for(auto &j: nd->features){
                    if(j != nd->var->id){
                        temp.push_back(j);
                    }
                }
                vector<vector<int>> child_features = vector<vector<int>>(nd->var->d, temp);
                nd->children = vector<CNode*>(nd->var->d);
                for(int k = 0; k < nd->var->d; k++){
                    nd->children[k] = learnCNode(dt, child_data_indices[k], child_features[k], depth+1, true);
                }
            }
        }
    }
    */
}

void CN::poissonOnlineLearn(Data &dt, vector<int> &indices){//}, bool paramsOnly){
    poissonOnlineLearnCNode(root, dt, indices, 0);
}