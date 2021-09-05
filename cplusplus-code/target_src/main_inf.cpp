//
// Created by Vasundhara Komaragiri on 11/6/20.
//

#include <iomanip>
#include <string>

#include "../include/Utils.h"
#include "../include/Data.h"
#include "../include/CLT.h"
#include "../include/CN.h"
#include "../include/MT.h"
#include "../include/MCN.h"
#include "../include/MCN_BTP.h"

using namespace std;

ldouble compute_kl(vector<vector<ldouble>> &exact_marg, vector<vector<ldouble>> &approx_marg){
    vector<vector<ldouble>> log_exact, log_approx;
    log_exact = vector<vector<ldouble>>(exact_marg);
    log_approx = vector<vector<ldouble>>(approx_marg);
    for(int i = 0; i < exact_marg.size(); i++){
        ldouble norm_const1 = 0.0, norm_const2 = 0.0;
        for(int j = 0; j < exact_marg[i].size(); j++){
            if(log_exact[i][j] < 1e-300){
                log_exact[i][j] = 1e-300;
            }
            if(log_approx[i][j] < 1e-300){
                log_approx[i][j] = 1e-300;
            }
            norm_const1 += log_exact[i][j];
            norm_const2 += log_approx[i][j];
        }
        for(int j = 0; j < exact_marg[i].size(); j++){
            log_exact[i][j] /= norm_const1;
            log_exact[i][j] = log(log_exact[i][j]);
            
            log_approx[i][j] /= norm_const2;
            log_approx[i][j] = log(log_approx[i][j]);
        }
        
    }

    ldouble res = 0.0;
    for(int i = 0; i < exact_marg.size(); i++){
        for(int j = 0; j < exact_marg[i].size(); j++){
            res += exact_marg[i][j]*(log_exact[i][j]-log_approx[i][j]);
        }
    }
    res /= exact_marg.size();
    return res;
}

void compute_posterior_marginals(vector<vector<int>> &samples, vector<ldouble>&log_weights, vector<vector<ldouble>> &marginals){
    for(int i = 0; i < samples.size(); i++){
        //cout << exp(log_weights[i]) << " ";
        for(int j = 0; j < samples[i].size(); j++){
            marginals[j][samples[i][j]] = log(exp(marginals[j][samples[i][j]])+exp(log_weights[i]));
        }
    }
    //cout << endl;
    for(int j = 0; j < marginals.size(); j++){
        ldouble norm_const = 0.0;
        for(int xj = 0; xj < marginals[j].size(); xj++){
            marginals[j][xj] = exp(marginals[j][xj]);
            norm_const += marginals[j][xj];
        }
        //cout << "norm: " << norm_const << endl;
        //Utils::print1d(marginals[j]);

        for(int xj = 0; xj < marginals[j].size(); xj++){
            marginals[j][xj] /= norm_const;
        }
        //Utils::print1d(marginals[j]);
        //Utils::normalize1d(marginals[j]);
    }
    //Utils::printMarginals(marginals, "");
}

int main(int argc, char *argv[]) {
    if(argc < 5){
        cout << "Usage format: ./inf-bn <model_directory> <(results)dataset_directory> <dataset_name> <evid_percent>" << endl;
        exit(0);
    }
    string model_dirctory(argv[1]);
    string dataset_directory(argv[2]);
    string dataset_name(argv[3]);
    float evid_percent(atof(argv[4]));
    
    vector<int> modified_evid_var, modified_evid_val;
    string evidfilename = dataset_directory+dataset_name+"_modified.evid";
    ifstream evid(evidfilename);
    int n;
    if(!evid.good()){
        n = 0;
        cout << "Original model used" << endl << endl;
    }
    else{
        cout << "Modified treewidth model used" << endl << endl;
        evid >> n;
        modified_evid_val = vector<int>(n);
        modified_evid_var = vector<int>(n);
        for(int j = 0; j < n; j++){
            evid >> modified_evid_var[j];
            evid >> modified_evid_val[j];
        }
    }
    
    CLT bn = CLT();
    bn.readUAI08(model_dirctory+dataset_name+".uai");
    int nsamples = 100000;
    vector<int> order;
    Utils::getTopologicalOrder(bn.variables, bn.functions, order);
    for(int k = 0; k < 5; k++){
        cout << "Reading models..." << endl;
        //MCN mcn = MCN();
        //mcn.read(model_dirctory+dataset_name+"-oracle-uai-50.mcn");

        //MCN mcn2 = MCN();
        //mcn2.read(model_dirctory+dataset_name+"-50.mcn");
        
        CLT bn = CLT();
        bn.readUAI08(model_dirctory+dataset_name+".uai");

        for(int j = 0; j < n; j++){
            bn.setEvidence(modified_evid_var[j], modified_evid_val[j]);
            //mcn.setEvidence(modified_evid_var[j], modified_evid_val[j]);
            //mcn2.setEvidence(modified_evid_var[j], modified_evid_val[j]);
        }

        vector<int> evid_var, evid_val;
        int j = 0;
        int i = bn.variables.size()-1;
        while(j < (evid_percent/100.0)*(bn.variables.size()-n)){
            int var = order[i];
            for(int t = 0; t < n; t++){
                if(modified_evid_var[t] == var){
                    i--;
                    continue;
                }
            }
            
            int val = myRandom::getInt(2);
            evid_var.push_back(var);
            evid_val.push_back(val);
            bn.setEvidence(var, val);
            //mcn.setEvidence(var, val);
            //mcn2.setEvidence(var, val);
            j++;
            i--;
        }


        cout << "Computing posterior distributions..." << endl;
        BTP btp = BTP(bn);
        BN_Sampler bns;
        btp.getPosteriorSampler(bns);
        
        //MCN_BTP mcn_btp = MCN_BTP(mcn);
        //MCN_Sampler mcns;
        //mcn_btp.getPosteriorSampler(mcns);
        //MCN_BTP mcn_btp2 = MCN_BTP(mcn2);
        //MCN_Sampler mcns2;
        //mcn_btp2.getPosteriorSampler(mcns2);
        
        //cout << flag << endl;
        cout << "Generating samples..." << endl;
        vector<vector<int>> samples;
        bns.generateSamples(nsamples, samples);    

        cout << "Writing samples and evidence..." << endl;
        string outfilename, evidfilename, mtprobfilename, oracleprobfilename, mtprobfilename2, lwprobfilename;
        if(n == 0){
            outfilename = dataset_directory+dataset_name+"_"+argv[4]+"_percent_"+to_string(k)+".post.data";
            evidfilename = dataset_directory+dataset_name+"_"+argv[4]+"_percent_"+to_string(k)+".evid";
            //mtprobfilename = dataset_directory+dataset_name+"_"+argv[4]+"_percent_"+to_string(k)+".tpm";
            oracleprobfilename = dataset_directory+dataset_name+"_"+argv[4]+"_percent_"+to_string(k)+".bn.wt";
            lwprobfilename = dataset_directory+dataset_name+"_"+argv[4]+"_percent_"+to_string(k)+".lw.wt";
            //mtprobfilename2 = dataset_directory+dataset_name+"_"+argv[4]+"_percent_"+to_string(k)+".tpm2";
        }
        else{
            outfilename = dataset_directory+dataset_name+"_"+argv[4]+"_percent_"+to_string(k)+"_modified.post.data";
            evidfilename = dataset_directory+dataset_name+"_"+argv[4]+"_percent_"+to_string(k)+"_modified.evid";
            //mtprobfilename = dataset_directory+dataset_name+"_"+argv[4]+"_percent_"+to_string(k)+"_modified.tpm";
            oracleprobfilename = dataset_directory+dataset_name+"_"+argv[4]+"_percent_"+to_string(k)+"_modified.bn.wt";
            //mtprobfilename2 = dataset_directory+dataset_name+"_"+argv[4]+"_percent_"+to_string(k)+"_modified.tpm2";
            lwprobfilename = dataset_directory+dataset_name+"_"+argv[4]+"_percent_"+to_string(k)+"_modified.lw.wt";
        }
        Utils::printSamples(samples, outfilename);
        Utils::printEvid(evid_var, evid_val, evidfilename);
        
        cout <<"Computing sample probabilities" << endl;
        vector<ldouble> mcn_log_prob = vector<ldouble>(samples.size()), bn_log_prob = vector<ldouble>(samples.size());
        vector<ldouble> mcn_log_prob2 = vector<ldouble>(samples.size());
        vector<ldouble> lw_log_prob = vector<ldouble>(samples.size());
        for(int i = 0; i < samples.size(); i++){
            //mcn_log_prob[i] = log(mcns.getProbability(samples[i]));
            bn_log_prob[i] = log(bns.getProbability(samples[i]));
            lw_log_prob[i] = bn.getLogLWPostProbability(samples[i]);
            //mcn_log_prob2[i] = log(mcns2.getProbability(samples[i]));
        }

        //Utils::print1d(mcn_log_prob, mtprobfilename);
        Utils::print1d(bn_log_prob, oracleprobfilename);
        Utils::print1d(lw_log_prob, lwprobfilename);
        //Utils::print1d(mcn_log_prob2, mtprobfilename2);
    }

    
    /*
    // The commented code computes KL-D between marginals obtained from the exact estimate of the Oracle BN from exct-TPM estimate, imp-TPM estimate and LW-estimate
    string model_dirctory(argv[1]);
    string dataset_name(argv[2]);
    float evid_percent(atof(argv[3]));
    int nsamples;
    vector<vector<ldouble>> marginals, mcn_marginals, mcnp_marginals, lw_marginals;
    vector<vector<int>> mcn_samples, lw_samples;
    vector<ldouble> mcnp_wts, lw_wts;

    CLT bn = CLT();
    bn.readUAI08(model_dirctory+dataset_name+".uai");
    cout << "Read Oracle BN..." << endl;
    
    vector<int> order;
    Utils::getTopologicalOrder(bn.variables, bn.functions, order);
    Utils::print1d(order);

    cout << "Setting evidence..." << endl;
    vector<int> evid_var, evid_val;
    for(int i = bn.variables.size()-1; i >= evid_percent/100.0*bn.variables.size(); i--){
        //int val = myRandom::getInt(2);
        int val = 0;
        int var = order[i];
        evid_var.push_back(var);
        evid_val.push_back(val);
    }

    cout << "Computing exact estimate on TPM..." << endl;
    MCN mcn = MCN();
    mcn.read(model_dirctory+dataset_name+"-oracle-uai.mcn");
    for(int i = 0; i < evid_var.size(); i++){
        mcn.setEvidence(evid_var[i], evid_val[i]);
    }
    MCN_BTP mcn_btp;
    mcn_btp  = MCN_BTP(mcn);
    mcn_btp.getVarMarginals(mcn_marginals);

    cout << "Computing imp sampling estimate on TPM..." << endl;
    MCN_Sampler mcns;
    mcn_btp.getPosteriorSampler(mcns);

    cout << "Computing exact estimate on Bayesian network..." << endl;
    for(int i = 0; i < evid_var.size(); i++){
        bn.setEvidence(evid_var[i], evid_val[i]);
    }
    BTP btp;
    btp = BTP(bn);
    btp.getVarMarginals(marginals);
    cout << "KL(exact-TPM): " << compute_kl(marginals, mcn_marginals) << endl; 
    BN_Sampler bns;
    btp.getPosteriorSampler(bns);
        

    for(int j = 1000; j <= 1000; j*=10){
        nsamples = j;
        mcns.generateSamples(nsamples, mcn_samples);
        mcnp_wts = vector<ldouble>(nsamples, 0.0);
        for(int i = 0; i < nsamples; i++){
            ldouble den = mcns.getProbability(mcn_samples[i]);
            ldouble num = bn.getProbability(mcn_samples[i]);
            mcnp_wts[i] = log(num/den);
        }
        //Utils::print1d(mcnp_wts);
        cout << "Computing LW estimate on Bayesian network..." << endl;
        bns.generateSamples(nsamples, lw_samples);
        lw_wts = vector<ldouble>(nsamples, 0.0);
        for(int i = 0; i < nsamples; i++){
            lw_wts[i] = bns.getLogWeight(lw_samples[i]);
        }
        
        mcnp_marginals = vector<vector<ldouble>>(mcn.variables.size());
        lw_marginals = vector<vector<ldouble>>(bn.variables.size());
        for(int i = 0; i < mcnp_marginals.size(); i++){
            mcnp_marginals[i] = vector<ldouble> (mcn.variables[i]->d, -1e100);
            lw_marginals[i] = vector<ldouble> (bn.variables[i]->d, -1e100);
        }
        cout << "num of samples: " << nsamples << endl;
        compute_posterior_marginals(mcn_samples, mcnp_wts, mcnp_marginals);
        compute_posterior_marginals(lw_samples, lw_wts, lw_marginals);

        cout << "KL(imp-TPM): " << compute_kl(marginals, mcnp_marginals) << endl;
        cout << "KL(imp-LW): " << compute_kl(marginals, lw_marginals) << endl;;
    }
    */
    return 0;
}