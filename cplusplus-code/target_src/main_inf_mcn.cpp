//
// Created by Vasundhara Komaragiri on 11/6/20.
//

#include <iomanip>
#include <string>
#include <sstream>
#include <iostream>

#include "../include/Utils.h"
#include "../include/Data.h"
#include "../include/CLT.h"
#include "../include/CN.h"
#include "../include/MT.h"
#include "../include/MCN.h"
#include "../include/MCN_BTP.h"

using namespace std;

int main(int argc, char *argv[]) {
    if(argc < 6){
        cout << "Usage format: ./inf-MCN <model_directory> <(results)dataset_directory> <dataset_name> <evid_percent> <data-mcn-available>" << endl;
        exit(0);
    }
    string model_dirctory(argv[1]);
    string dataset_directory(argv[2]);
    string dataset_name(argv[3]);
    float evid_percent(atoi(argv[4]));
    bool flag_data_mcn(atoi(argv[5]));
    

    for(int k = 0; k < 5; k++){
        if(!flag_data_mcn){
            cout << "Reading models..." << endl;
            MCN mcn = MCN();
            mcn.read(model_dirctory+dataset_name+"-oracle-uai-50.mcn");
            
            cout << "Readong oracle samples and evidence..." << endl;
            string outfilename, evidfilename, mcnprobfilename;    
            outfilename = dataset_directory+dataset_name+"_"+argv[4]+"_percent_"+to_string(k)+".post.data";
            evidfilename = dataset_directory+dataset_name+"_"+argv[4]+"_percent_"+to_string(k)+".evid";
            mcnprobfilename = dataset_directory+dataset_name+"_"+argv[4]+"_percent_"+to_string(k)+".mcn.wt";

            vector<int> evid_var, evid_val;
            ifstream evid_stream(evidfilename);
            int nevid;
            evid_stream >> nevid;
            evid_val = vector<int>(nevid);
            evid_var = vector<int>(nevid);
            for(int j = 0; j < nevid; j++){
                evid_stream >> evid_var[j];
                evid_stream >> evid_val[j];
                mcn.setEvidence(evid_var[j], evid_val[j]);
            }
            evid_stream.close();
            MCN_BTP mcn_btp = MCN_BTP(mcn);
            MCN_Sampler mcns;
            mcn_btp.getPosteriorSampler(mcns);
            
            cout <<"Computing sample probabilities" << endl;
            vector<ldouble> mcn_log_prob;
            ifstream bn_data(outfilename);
            string line;
            int i = 0;
            while(getline(bn_data, line)){
                stringstream inss(line);
                vector<int> row;
                int m;
                while(inss >> m){
                    row.push_back(m);
                    if (inss.peek() == ',')
                        inss.ignore();
                }
                ldouble p = log(mcns.getProbability(row));
                mcn_log_prob.push_back(p);
                i++;
            }
            bn_data.close();
            Utils::print1d(mcn_log_prob, mcnprobfilename);
        }
        else{
            cout << "Reading models..." << endl;
            MCN oracle_mcn = MCN();
            oracle_mcn.read(model_dirctory+dataset_name+"-oracle-uai-50.mcn");
            MCN data_mcn = MCN();
            data_mcn.read(model_dirctory+dataset_name+"-50.mcn");

            cout << "Readong oracle samples and evidence..." << endl;
            string outfilename, evidfilename, oracle_mcnprobfilename, data_mcnprobfilename;    
            outfilename = dataset_directory+dataset_name+"_"+argv[4]+"_percent_"+to_string(k)+".post.data";
            evidfilename = dataset_directory+dataset_name+"_"+argv[4]+"_percent_"+to_string(k)+".evid";
            oracle_mcnprobfilename = dataset_directory+dataset_name+"_"+argv[4]+"_percent_"+to_string(k)+".mcn.wt";
            data_mcnprobfilename = dataset_directory+dataset_name+"_"+argv[4]+"_percent_"+to_string(k)+".data_mcn.wt";

            vector<int> evid_var, evid_val;
            ifstream evid_stream(evidfilename);
            int nevid;
            evid_stream >> nevid;
            evid_val = vector<int>(nevid);
            evid_var = vector<int>(nevid);
            for(int j = 0; j < nevid; j++){
                evid_stream >> evid_var[j];
                evid_stream >> evid_val[j];
                oracle_mcn.setEvidence(evid_var[j], evid_val[j]);
                data_mcn.setEvidence(evid_var[j], evid_val[j]);
            }
            evid_stream.close();
            MCN_BTP oracle_mcn_btp = MCN_BTP(oracle_mcn);
            MCN_Sampler oracle_mcns;
            oracle_mcn_btp.getPosteriorSampler(oracle_mcns);
            MCN_BTP data_mcn_btp = MCN_BTP(data_mcn);
            MCN_Sampler data_mcns;
            data_mcn_btp.getPosteriorSampler(data_mcns);

            cout <<"Computing sample probabilities" << endl;
            vector<ldouble> oracle_mcn_log_prob, data_mcn_log_prob;
            ifstream bn_data(outfilename);
            string line;
            int i = 0;
            while(getline(bn_data, line)){
                stringstream inss(line);
                vector<int> row;
                int m;
                while(inss >> m){
                    row.push_back(m);
                    if (inss.peek() == ',')
                        inss.ignore();
                }
                oracle_mcn_log_prob.push_back(log(oracle_mcns.getProbability(row)));
                data_mcn_log_prob.push_back(log(data_mcns.getProbability(row)));
                i++;
            }
            cout << k << "," << i << endl;
            bn_data.close();
            Utils::print1d(oracle_mcn_log_prob, oracle_mcnprobfilename);
            Utils::print1d(data_mcn_log_prob, data_mcnprobfilename);
        }
    }

    return 0;
}