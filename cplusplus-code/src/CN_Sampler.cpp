
#include <CN_Sampler.h>

CN_Sampler::CN_Sampler(CN &cn_){
    cns = CN();
    variables = cn_.variables;
    cns.setRoot(getSamplerCNode(cn_.getRoot()));
}

CNode* CN_Sampler::getSamplerCNode(CNode* nd){
    CNode* out = new CNode();
    if(!nd){
        out = nullptr;
    }
    else{
        out->type = nd->type;
        if(nd->type == 0){
            out->var = nd->var;
            out->distribution = discrete_distribution<int>(nd->child_weights.begin(), nd->child_weights.end());
            out->children = vector<CNode*>(nd->children.size());
            for(int k = 0; k < nd->children.size(); k++){
                out->children[k] = getSamplerCNode(nd->children[k]);
            }
        }
        else{
            out->bns = new BN_Sampler(*(nd->clt), nd->varid_ind);
        }
    }
    return out;
}

void CN_Sampler::generateSamples(int n, vector<vector<int>> &samples){
    samples = vector<vector<int>>(n, vector<int>(variables.size(), -1));
    for(int i=0; i < n; i++){
        generateCNodeSample(cns.getRoot(), samples[i]);
    }
}

void CN_Sampler::generateCNodeSample(CNode* nd, vector<int> &sample){
    if(nd){
        if(nd->type == 0){
            int val = nd->distribution(myRandom::m_g);
            sample[nd->var->id] = val;
            generateCNodeSample(nd->children[val], sample);
        }
        else{
            nd->bns->generateSample(sample);
        }
    }
}

ldouble CN_Sampler::getProbability(vector<int> &example){
    return exp(getCNodeLogProbability(cns.getRoot(), example));
}

ldouble CN_Sampler::getCNodeLogProbability(CNode* nd, vector<int> &example){
    if(!nd)
        return 0.0;
    if(nd->type == 0){
        int k = example[nd->var->id];
        if(nd->var->isEvid()){ 
            if(k != nd->var->getValue()){
                cout << "Error!!" << endl;
                exit(0);
            }
            return getCNodeLogProbability(nd->children[k], example);
        }
        else{
            return log(nd->distribution.probabilities()[k])+getCNodeLogProbability(nd->children[k], example);
        }
    }
    else{
        return log(nd->bns->getProbability(example));
    }
}

