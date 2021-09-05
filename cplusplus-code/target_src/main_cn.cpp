//
// Created by Vasundhara Komaragiri on 11/6/20.
//

#include <iomanip>

#include <Utils.h>
#include <Data.h>
#include <CN.h>
#include <CN_BTP.h>
#include <CN_Sampler.h>
//#include <MT.h>
//#include <myRandom.h>

using namespace std;
Data train_data, valid_data, test_data;

int main(int argc, char *argv[]) {
    if(argc < 2){
        cout << "Usage format: ./CN <dataset_path>" << endl;
        exit(0);
    }
    
    string dataset(argv[1]);
    cout << dataset << endl;
    string train = dataset+".ts.data";
    train_data.readCSVData(train);
    string valid = dataset+".valid.data";
    valid_data.readCSVData(valid);
    string test = dataset+".test.data";
    test_data.readCSVData(test);
    cout << "Read data..." << endl;
    
    CN cn = CN();

    
    cout << "Learning cutset network..." << endl; 
    cn.learn(train_data, valid_data, true, false, false);
    
    
    //cout << "Printing Cutset Network..." << endl;
    //cn.print();
    cout << "Avg test log-likelihood: " << cn.log_likelihood(test_data) << endl;
    
    cout << "Writing to file..." << endl;
    string outfilename = dataset+".cn";
    cn.write(outfilename);
    cout << "Done" << endl;
    
    /*
    CN cn = CN();
    string infilename = dataset+".cn";
    cn.read(infilename);
    cn.write(dataset+"2.cn");
    ////////////
    */
    cn.setEvidence(1, 1);
    
    CN_BTP cn_btp = CN_BTP(cn);

    cout << "pe: " << cn_btp.getPE() << endl;
    
    vector<vector<ldouble>> marginals;
    cn_btp.getVarMarginals(marginals);
    Utils::printMarginals(marginals, "");

    CN_Sampler cns2;
    CN_Sampler cns = CN_Sampler(cn);

    cn_btp.getPosteriorSampler(cns2);
    vector<vector<int>> samples;
    cns.generateSamples(10, samples);
    Utils::printSamples(samples, "");
    vector<ldouble> wt, wt2;
    for(auto &s: samples){
        wt.push_back(cns.getProbability(s));
    }
    Utils::print1d(wt);
    cns2.generateSamples(10, samples);
    Utils::printSamples(samples, "");
    for(auto &s: samples){
        wt2.push_back(cns2.getProbability(s));
    }
    Utils::print1d(wt2);
}