#include <CN_BTP.h>

CN_BTP::CN_BTP(CN &cn_){
    cn_btp = CN();
    cn_btp = cn_;
    pe = 1.0;
    upward = false;
    //downward = false;
    setEvid(cn_btp.getRoot());
}

void CN_BTP::setEvid(CNode* nd){
    if(nd){
        if(nd->type == 0){
            if(nd->var->isEvid()){
                for(int k = 0; k < nd->children.size(); k++){
                    if(k != nd->var->getValue()){
                        nd->child_weights[k] *= 0;
                    }
                    else{
                        setEvid(nd->children[k]);
                    }
                }
            }
            else{
                for(int k = 0; k < nd->children.size(); k++){
                    setEvid(nd->children[k]);
                }
            }
        }
        else{
            nd->btp = new BTP(*(nd->clt), nd->varid_ind);
        }
    }
}

ldouble CN_BTP::getPE(){
    if(!upward){
        doUpwardPass();
    }
    pe = cn_btp.getRoot()->val;
    return pe;
}

void CN_BTP::doUpwardPass(){
    upwardPass(cn_btp.getRoot());
    upward = true;
}

void CN_BTP::upwardPass(CNode* nd){
    if(nd){
        if(nd->type == 0){
            nd->child_posterior_weights = vector<ldouble>(nd->child_weights.size(), 0.0);
            if(nd->var->isEvid()){
                int k = nd->var->getValue();
                upwardPass(nd->children[k]);
                if(nd->children[k])
                    nd->val = nd->child_weights[k]*nd->children[k]->val;
                else 
                    nd->val = nd->child_weights[k];
                nd->child_posterior_weights[k] = nd->val;
            }
            else{
                ldouble out = 0.0, temp;
                for(int k = 0; k < nd->children.size(); k++){
                    upwardPass(nd->children[k]);
                    if(nd->children[k])
                        temp = nd->child_weights[k]*nd->children[k]->val;
                    else 
                        temp = nd->child_weights[k];
                    out += temp;
                    nd->child_posterior_weights[k] = temp;
                }
                nd->val = out;
            }
        }
        else{
            nd->val = nd->btp->getPE();
        }
    }
}



void CN_BTP::getVarMarginals(vector<vector<ldouble>> &var_marginals){
    computeVarMarginals(cn_btp.getRoot(), var_marginals);
    for(int j = 0; j < var_marginals.size(); j++){
        Utils::normalize1d(var_marginals[j]);
    }
}

void CN_BTP::computeVarMarginals(CNode* nd, vector<vector<ldouble>> &marg){
    if(nd){
        if(nd->type == 0){
            marg = vector<vector<ldouble>>(nd->features.size());

            if(nd->var->isEvid()){
                vector<vector<ldouble>> temp;
                int k = nd->var->getValue();
                computeVarMarginals(nd->children[k], temp);
                int i = 0;
                for(int j = 0; j < marg.size(); j++){
                    if(nd->var->id == nd->features[j]){
                        marg[j] = nd->child_weights;
                    }
                    else{
                        marg[j] = vector<ldouble>(temp[i].size());
                        for(int xj = 0; xj < temp[i].size(); xj++){
                            marg[j][xj] = temp[i][xj]*nd->child_weights[k];
                        }
                        i++;
                    }
                }
            }
            else{
                int ind = -1;
                for(int j = 0; j < nd->features.size(); j++){
                    if(nd->features[j] == nd->var->id){
                        marg[j] = nd->child_weights;
                        ind = j;
                    }
                    else{
                        marg[j] = vector<ldouble>(cn_btp.variables[nd->features[j]]->d, 0.0);
                    }
                }
                
                for(int k = 0; k < nd->children.size(); k++){
                    vector<vector<ldouble>> temp;
                    computeVarMarginals(nd->children[k], temp);
                    int i = 0;
                    for(int j = 0; j < marg.size(); j++){
                        if(j == ind) continue;
                        for(int xj = 0; xj < marg[j].size(); xj++){
                            marg[j][xj] += nd->child_weights[k]*temp[i][xj];
                        }
                        i++;
                    }
                }
            }
        }
        else{
            nd->btp->getVarMarginals(marg);
        }
    }
}


void CN_BTP::getPosteriorSampler(CN_Sampler &cn_sampler){
    cn_sampler = CN_Sampler();
    if(!upward){
        doUpwardPass();
    }
    /*
    if(!downward){
        doDownwardPass();
    }
    */
    cn_sampler.cns = CN();
    cn_sampler.variables = cn_btp.variables;
    cn_sampler.cns.setRoot(getPosteriorSamplerCNode(cn_btp.getRoot()));
}

/*
void CN_BTP::doDownwardPass(){
    downwardPass(cn_btp.getRoot());
    downward = true;
}

void CN_BTP::downwardPass(CNode* nd){
    if(nd){
        if(nd->type == 0){
            Utils::normalize1d(nd->child_posterior_weights);
            if(nd->var->isEvid()){
                int k = nd->var->getValue();
                downwardPass(nd->children[k]);
            }
            else{
                for(int k = 0; k < nd->children.size(); k++){
                    downwardPass(nd->children[k]);
                }
            }
        }
    }
}
*/


CNode* CN_BTP::getPosteriorSamplerCNode(CNode* nd){
    CNode* out = new CNode();
    if(!nd){
        out = nullptr;
    }
    else{
        out->type = nd->type;
        if(nd->type == 0){
            out->var = nd->var;
            Utils::normalize1d(nd->child_posterior_weights);
            out->distribution = discrete_distribution<int>(nd->child_posterior_weights.begin(), nd->child_posterior_weights.end());
            out->children = vector<CNode*>(nd->children.size());
            if(nd->var->isEvid()){
                int k = nd->var->getValue();
                out->children[k] = getPosteriorSamplerCNode(nd->children[k]);
            }
            else{
                for(int k = 0; k < nd->children.size(); k++){
                    out->children[k] = getPosteriorSamplerCNode(nd->children[k]);
                }
            }
        }
        else{
            out->bns = new BN_Sampler();
            nd->btp->getPosteriorSampler(*(out->bns));
        }
    }
    return out;
}