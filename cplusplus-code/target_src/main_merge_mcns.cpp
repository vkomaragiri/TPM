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
#include "../include/MT_BTP.h"
#include "../include/MCN.h"
#include "../include/MCN_BTP.h"

using namespace std;

int main(int argc, char *argv[]) {
    if(argc < 3){
        cout << "Usage format: ./merge_models <model_directory> <dataset_name> <nbags>" << endl;
        exit(0);
    }
    string model_dirctory(argv[1]);
    string dataset_name(argv[2]);

    int nbags = atoi(argv[3]);

    MCN mcn = MCN();
    mcn.readCounts(model_dirctory+dataset_name+"-oracle-uai-"+to_string(nbags)+".mcn1Counts");
    mcn.normalizeParams();
    if(mcn.ncomponents %10 == 0)
        mcn.write(model_dirctory+dataset_name+"-oracle-uai-"+to_string(mcn.ncomponents)+".mcn");
    for(int k = 2; k < (50/nbags+1); k++){
        MCN mcn2 = MCN();
        mcn2.readCounts(model_dirctory+dataset_name+"-oracle-uai-"+to_string(nbags)+".mcn"+to_string(k)+"Counts");
        //mcn2.readCounts(model_dirctory+dataset_name+"-"+to_string(nbags)+".mcn"+to_string(k)+"Counts");
        mcn2.normalizeParams();
        mcn.mergeModel(mcn2);
        if(mcn.ncomponents %10 == 0)
            mcn.write(model_dirctory+dataset_name+"-oracle-uai-"+to_string(mcn.ncomponents)+".mcn");
    }
    return 0;
}