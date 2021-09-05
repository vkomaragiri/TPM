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


int main(int argc, char *argv[]) {
    if(argc < 5){
        cout << "Usage format: ./gen_samples <model_directory> <dataset_directory> <dataset_name> <k-value>" << endl;
        exit(0);
    }
    string model_dirctory(argv[1]);
    string dataset_directory(argv[2]);
    string dataset_name(argv[3]);
    int k = atoi(argv[4]); //Number of variables instatiating
    
    CLT bn = CLT();
    bn.readUAI08(model_dirctory+dataset_name+".uai");

    

    vector<int> min_order;
    Utils::getMinDegreeOrder(bn.variables, bn.functions, min_order);
    Utils::print1d(min_order);

    if(k > 0){
        string evidfilename = dataset_directory+dataset_name+"_modified.evid";
        ofstream evid(evidfilename);
        evid << k;
        for(int i = bn.variables.size()-1; i > bn.variables.size()-k-1; i--){
            int val = myRandom::getInt(2);
            int var = min_order[i];
            bn.setEvidence(var, val);
            evid << " " << var << " " << val;
        }
        evid << endl;
    }

    BTP btp = BTP(bn);
    int treewidth = btp.getTreeWidth();
    cout << "treewidth: " << treewidth << endl;
    
    string twfilename;
    if(k == 0)
        twfilename = dataset_directory+dataset_name+".treewidth";
    else
        twfilename = dataset_directory+dataset_name+"_modified.treewidth";
    ofstream out(twfilename);
    out << treewidth << endl;

    /*
    CLT modified_bn = CLT();
    btp.getPosteriorDist(modified_bn);
    modified_bn.write(model_dirctory+dataset_name+"_modified.uai");
    */


    return 0;
}