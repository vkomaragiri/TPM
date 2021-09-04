//
// Created by Vasundhara Komaragiri on 11/6/20.
//

#include <iomanip>

#include <Utils.h>
#include <Data.h>
#include <MCN.h>
#include <MCN_BTP.h>
#include <MCN_Sampler.h>

using namespace std;
Data train_data, valid_data, test_data;

int main(int argc, char *argv[]) {
    if(argc < 4){
        cout << "Usage format: ./MCN <model_directory> <dataset_directory> <dataset_name> <nbags> <outfilename>" << endl;
        exit(0);
    }
    
    string model_dirctory(argv[1]);
    string dataset_directory(argv[2]);
    string dataset_name(argv[3]);
    string dataset=dataset_directory+dataset_name;
    cout << dataset << endl;
    string train = dataset+".ts.data";
    train_data.readCSVData(train);
    string valid = dataset+".valid.data";
    valid_data.readCSVData(valid);
    string test = dataset+".test.data";
    test_data.readCSVData(test);
    cout << "Read data..." << endl;
    
    int nbags = atoi(argv[4]);

    //string outfilename = model_dirctory+dataset_name+"-"+to_string(nbags)+".mcn";
    string outfilename = model_dirctory+dataset_name+"-oracle-uai-"+to_string(nbags)+".mcn";
    
    if(argc == 6){
        outfilename += argv[5];
    }
    MCN mcn = MCN();
    cout << "Learning mixture of cutset networks..." << endl; 
    mcn.learn(train_data, valid_data, nbags);
    
    
    //cout << "Printing Cutset Network..." << endl;
    //cn.print();
    cout << "Avg test log-likelihood: " << mcn.log_likelihood(test_data) << endl;
    
    
    cout << "Writing to file..." << endl;
    mcn.writeCounts(outfilename+"Counts");
    mcn.write(outfilename);
    cout << "Done" << endl;
    
    /*
    MCN mcn2 = MCN();
    string infilename = dataset+".mcn";
    mcn2.read(infilename);
    mcn2.write(dataset+"2.mcn");
    */
    /*
    MCN_Sampler mcns2 = MCN_Sampler(mcn);
    vector<vector<int>> samples, samples2;
    mcns2.generateSamples(10, samples2);
    Utils::printSamples(samples2, "");
    vector<ldouble> wt2;
    for(auto &s: samples2){
        wt2.push_back(mcns2.getProbability(s));
    }
    Utils::print1d(wt2);
    
    mcn.setEvidence(1, 1);
    mcn.setEvidence(4, 0);
    MCN_BTP mcn_btp = MCN_BTP(mcn);
    cout << "pe: " << mcn_btp.getPE() << endl;
    vector<vector<ldouble>> marginals;
    mcn_btp.getVarMarginals(marginals);
    Utils::printMarginals(marginals, "");

    MCN_Sampler mcns;
    mcn_btp.getPosteriorSampler(mcns);
    mcns.generateSamples(10, samples);
    Utils::printSamples(samples, "");
    vector<ldouble> wt;
    for(auto &s: samples){
        wt.push_back(mcns.getProbability(s));
    }
    Utils::print1d(wt);
    */
   return 0;
}