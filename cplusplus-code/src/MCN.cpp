#include <MCN.h>
#include <chrono>
#include <random>


void MCN::learn(Data &train_data, Data &valid_data, int nbags, int max_depth){
    ncomponents = nbags;
    cout <<"ncomponents: " << ncomponents << endl;
    prob_mixture = vector<ldouble>(ncomponents);
    cns = vector<CN>(ncomponents);

    for (int i = 0; i < train_data.nfeatures; i++) {
        Variable* var = new Variable(i, train_data.dsize[i]);
        variables.push_back(var);
    }

    vector<ldouble> weights = vector<ldouble>(train_data.nexamples, 1.0);
    vector<Data> bags = vector<Data>(ncomponents);
    for(int j = 0; j < ncomponents; j++){
        bags[j].nexamples = train_data.nexamples;
        bags[j].nfeatures = train_data.nfeatures;
        bags[j].dsize = train_data.dsize;
        bags[j].setWeights(weights);
        bags[j].data_matrix = vector<vector<int>>(train_data.nexamples);
        for(int i = 0; i < train_data.nexamples; i++){
            int k = myRandom::getInt(train_data.nexamples-1);
            bags[j].data_matrix[i] = train_data.data_matrix[k];
        }
        cns[j].variables = variables;

        //cns[j].learn(bags[j], valid_data, true, true, true, max_depth);
        
        if(j < ((float)0.2*ncomponents))
            cns[j].learn(bags[j], valid_data, true, true, true, 2);
        else if (j <= ((float)0.2*ncomponents) && j < ((float)0.4*ncomponents))
            cns[j].learn(bags[j], valid_data, true, true, true, 4);
        else if (j <= ((float)0.4*ncomponents) && j < ((float)0.6*ncomponents))
            cns[j].learn(bags[j], valid_data, true, true, true, 6);
        else if (j <= ((float)0.6*ncomponents) && j < ((float)0.8*ncomponents))
            cns[j].learn(bags[j], valid_data, true, true, true, 8);
        else
        
            cns[j].learn(bags[j], valid_data, true, true, true, 10);
        ldouble exp_ll = exp(cns[j].log_likelihood(valid_data));
        prob_mixture[j] = exp_ll;
    }
    count_prob_mixture = vector<ldouble>(prob_mixture);
    Utils::normalize1d(prob_mixture);
}

void MCN::addComps(vector<vector<vector<int>>> &bags, vector<int> &dsize, Data &valid_data){
    prob_mixture = vector<ldouble>(ncomponents+bags.size());
    for(int j = 0; j < ncomponents; j++){
        prob_mixture[j] = exp(cns[j].log_likelihood(valid_data));
    }
    
    ncomponents += bags.size();
    for(int j = 0; j < bags.size(); j++){
        Data dt;
        dt.nexamples = bags[j].size();
        dt.nfeatures = bags[j][0].size();
        dt.dsize = dsize;
        vector<ldouble> weights = vector<ldouble>(dt.nexamples, 1.0);
        dt.setWeights(weights);
        dt.data_matrix = bags[j];
        
        CN cn = CN();
        cn.variables = variables;
        cn.learn(dt, valid_data, true, true, true);
        cns.push_back(cn);
        prob_mixture[j] = exp(cn.log_likelihood(valid_data));
    }
    Utils::normalize1d(prob_mixture);
}

ldouble MCN::log_likelihood(Data &data){
    ldouble ll = 0.0;
    for(int i = 0; i < data.nexamples; i++){
        ll += getLogProb(data.data_matrix[i]);
    }
    ll /= data.nexamples;
    return ll;
}

ldouble MCN::getProb(vector<int> &example){
    ldouble prob = 0.0;
    for(int j = 0; j < ncomponents; j++){
        prob += prob_mixture[j]*cns[j].getProb(example);
    }
    return prob;
}    
    
ldouble MCN::getLogProb(vector<int> &example){
    return log(getProb(example));
}

void MCN::write(const string &infile){
    ofstream out(infile);
    out << "MCN" << endl;
    out << this->ncomponents << " " << this->variables.size() << "\n";
    for (auto &variable : variables) {
        out << variable->d << " ";
    }
    out << endl;
    for (int i = 0; i < ncomponents; i++) {
        out << prob_mixture[i] << " ";
    }
    out << endl;
    for(int i = 0; i < ncomponents; i++){
        cns[i].writeCNode(cns[i].getRoot(), out);
    }
    out.close();
}

void MCN::writeCounts(const string &infile){
    ofstream out(infile);
    out << "MCN" << endl;
    out << this->ncomponents << " " << this->variables.size() << "\n";
    for (auto &variable : variables) {
        out << variable->d << " ";
    }
    out << endl;
    for (int i = 0; i < ncomponents; i++) {
        out << count_prob_mixture[i] << " ";
    }
    out << endl;
    for(int i = 0; i < ncomponents; i++){
        cns[i].writeCNodeCounts(cns[i].getRoot(), out);
    }
    out.close();
}

void MCN::read(const string &filename)
{
    ifstream in(filename);
    string mystr;
    in>>mystr;
    if(mystr.find("MCN")==string::npos){
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

    cns=vector<CN>(ncomponents);
    for(int i=0;i<ncomponents;i++){
        cns[i] = CN();
        cns[i].variables = variables;
        cns[i].setRoot(cns[i].readCNode(in));
    }
    in.close();
}

void MCN::readCounts(const string &filename)
{
    ifstream in(filename);
    string mystr;
    in>>mystr;
    if(mystr.find("MCN")==string::npos){
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
    count_prob_mixture=vector<ldouble>(ncomponents);
    for(int i=0;i<ncomponents;i++){
        ldouble tmp;
        in>>tmp;
        count_prob_mixture[i]=tmp;
    }
    cns=vector<CN>(ncomponents);
    for(int i=0;i<ncomponents;i++){
        cns[i] = CN();
        cns[i].variables = variables;
        cns[i].setRoot(cns[i].readCNodeCounts(in));
    }
    in.close();
}

void MCN::setEvidence(int var, int val)
{
    variables[var]->setValue(val);
}

void MCN::compCNodeGrad(CNode* nd, vector<int> &data, ldouble &total_log_prob, ldouble &comp_log_prob){
    if(nd->type == 0){
        nd->grad_child_weights[data[nd->var->id]] += exp(comp_log_prob-total_log_prob-log(nd->child_weights[data[nd->var->id]]));
    }
    else{
        for(int j = 0; j < nd->clt->functions.size(); j++){
            int t = Utils::getAddr(nd->clt->functions[j].variables);
            nd->clt->gradients[j][t] += exp(comp_log_prob-total_log_prob-log(nd->clt->functions[j].potentials[t]));
        }
    }
}

void MCN::compGrad(vector<vector<int>> &data){
    for(int i = 0; i < data.size(); i++){
        for(auto &var: variables){
            var->t_val = data[i][var->id];
        }
        ldouble log_prob = getLogProb(data[i]);
        for(int j = 0; j < ncomponents; j++){
            ldouble comp_log_prob = cns[j].getLogProb(data[i]);
            grad_prob_mixture[j] += exp(comp_log_prob-log_prob);
            compCNodeGrad(cns[j].getRoot(), data[i], log_prob, comp_log_prob);
        }
    }
}

void MCN::doSGDUpdate(ldouble learning_rate){
    for(int k = 0; k < ncomponents; k++){
        prob_mixture[k] += learning_rate*grad_prob_mixture[k];
        cns[k].doSGDUpdate(learning_rate);
    }
    Utils::normalize1d(prob_mixture);
}

void MCN::initGrad(){
    grad_prob_mixture = vector<ldouble>(ncomponents, 0.0);
    for(int i = 0; i < ncomponents; i++){
        cns[i].initGrad();
    }
}

ldouble MCN::gradSqNorm(){
    ldouble res = 0.0;
    for(int j = 0; j < ncomponents; j++){
        res += grad_prob_mixture[j]*grad_prob_mixture[j];
        res += cns[j].gradSqNorm(cns[j].getRoot());
    }
    return res;
}

void MCN::normalizeParams(){
    prob_mixture = vector<ldouble>(count_prob_mixture);
    Utils::normalize1d(prob_mixture);
    for(int j = 0; j < ncomponents; j++){
        cns[j].normalizeParams();
    }
}

void MCN::poissonOnlineLearn(Data &dt, Data &valid_data){//}, bool paramsOnly){
    poisson_distribution<int> distribution(1.0);
    vector<vector<int>>indices(ncomponents);
    for(int i = 0; i < dt.nexamples; i++){
        for(int j = 0; j < ncomponents; j++){
            int p = distribution(myRandom::m_g);
            if(p){
                for(int k = 0; k < p; k++){
                    indices[j].push_back(i);
                }
            }
        }
    }
    for(int j = 0; j < ncomponents; j++){
        cns[j].poissonOnlineLearn(dt, indices[j]);//, paramsOnly);
        ldouble exp_ll = exp(cns[j].log_likelihood(valid_data));
        prob_mixture[j] = exp_ll;
    }
    Utils::normalize1d(prob_mixture);
}

void MCN::mergeModel(MCN &model){
    for(int i = 0; i < model.ncomponents; i++){
        count_prob_mixture.push_back(model.count_prob_mixture[i]);
        cns.push_back(model.cns[i]);
    }
    prob_mixture = vector<ldouble>(count_prob_mixture);
    Utils::normalize1d(prob_mixture);
    ncomponents += model.ncomponents;
}
