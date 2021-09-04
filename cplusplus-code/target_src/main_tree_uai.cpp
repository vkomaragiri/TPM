//
// Created by Vasundhara Komaragiri on 11/6/20.
//

#include <iomanip>

#include <Utils.h>
#include <Data.h>
#include <CLT.h>
#include <MCN.h>

using namespace std;
Data train_data, valid_data, test_data;

int main(int argc, char *argv[]) {
    if(argc < 2){
        cout << "Usage format: ./tree_uai <dataset_name> <mcn?>" << endl;
        exit(0);
    }
    
    string directory = "/Users/vasundharakomaragiri/research/TMap.12.2020/TMap/data/";


    string model(directory+argv[1]+".uai");

    CLT clt = CLT();
    clt.readUAI08(model);

    Data test_data;
    string test(directory+argv[1]+".test.data");
    test_data.readCSVData(test);

    cout << "(UAI) Log-likelihood: " << clt.log_likelihood(test_data) << endl;
    if(argc > 2){
        MCN mcn = MCN();
        string mcnfilename(directory+argv[1]+"-50.mcn");
        mcn.read(mcnfilename);
        cout << "(MCN) Log-likelihood: " << mcn.log_likelihood(test_data) << endl;
    }


    return 0;
}