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
    if(argc < 5){
        cout << "Usage format: ./inf-MCN <model_directory> <(results)dataset_directory> <dataset_name> <evid_percent>" << endl;
        exit(0);
    }
    string model_dirctory(argv[1]);
    string dataset_directory(argv[2]);
    string dataset_name(argv[3]);
    float evid_percent(atoi(argv[4]));
    

    for(int k = 0; k < 5; k++){
        cout << "Reading models..." << endl;
        MCN mcn = MCN();
        mcn.read(model_dirctory+dataset_name+"-oracle-uai-50.mcn");

        //MCN mcn2 = MCN();
        //mcn2.read(model_dirctory+dataset_name+"-50.mcn");

        
        cout << "Readong oracle samples and evidence..." << endl;
        string outfilename, evidfilename, mcnprobfilename, bnprobfilename, mtprobfilename2;    
        outfilename = dataset_directory+dataset_name+"_"+argv[4]+"_percent_"+to_string(k)+".post.data";
        evidfilename = dataset_directory+dataset_name+"_"+argv[4]+"_percent_"+to_string(k)+".evid";
        mcnprobfilename = dataset_directory+dataset_name+"_"+argv[4]+"_percent_"+to_string(k)+".mcn.wt";
        //mtprobfilename2 = dataset_directory+dataset_name+"_"+argv[4]+"_percent_"+to_string(k)+".tpm2";

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
            //mcn2.setEvidence(evid_var[j], evid_val[j]);
        }
        evid_stream.close();
        MCN_BTP mcn_btp = MCN_BTP(mcn);
        MCN_Sampler mcns;
        mcn_btp.getPosteriorSampler(mcns);
        //MCN_BTP mcn_btp2 = MCN_BTP(mcn2);
        //MCN_Sampler mcns2;
        //mcn_btp2.getPosteriorSampler(mcns2);

        //cout << sizeof(mcn2) << " " << sizeof(mcn_btp2) << endl;

        cout <<"Computing sample probabilities" << endl;
        vector<ldouble> mcn_log_prob;// = vector<ldouble>(bn_data.nexamples);
        //vector<ldouble> mcn_log_prob2;
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
            //ldouble p2 = log(mcns2.getProbability(row));
            //mcn_log_prob2.push_back(p2);
            i++;
        }
        bn_data.close();
        Utils::print1d(mcn_log_prob, mcnprobfilename);
        //Utils::print1d(mcn_log_prob2, mtprobfilename2);
    }

    return 0;
}