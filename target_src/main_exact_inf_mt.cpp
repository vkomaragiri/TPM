//
// Created by vasundhara on 1/15/20.
//
//
// Created by Vibhav Gogate on 9/22/19.
//


#include <iomanip>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <MT_BTP.h>


#include "../include/Utils.h"
#include "../include/Data.h"
#include "../include/CLT.h"
#include "../include/BTP.h"
//#include <MT.h>
//#include <myRandom.h>

using namespace std;

boost::program_options::options_description desc(
        "MTProposal");
string infile, outfilename_mt = "tempfile.mar", fname;


int parseOptions(int argc, char *argv[]) {

    try {

        //used (files .ts.data, .test.data, and .valid.data should be present in the directory)
        desc.add_options()
                ("help,?", "produce help message")
                ("infile,i", boost::program_options::value<std::string>(&infile), "input probabilistic model file")
                ("outfile,o", boost::program_options::value<std::string>(&outfilename_mt), "Store marginals")
                /*
                ("ivl",
                 boost::program_options::value<int>(&HyperParameters::interval_for_structure_learning)->default_value(
                         10), "Interval for structure learning in EM")
                ("nem", boost::program_options::value<int>(&HyperParameters::num_iterations_em)->default_value(100),
                 "Max Number of iterations for EM")
                ("tol", boost::program_options::value<ldouble>(&HyperParameters::tol)->default_value(1.0e-5),
                 "Tolerance for EM LL scores")
                ("nc", boost::program_options::value<int>(&HyperParameters::num_components)->default_value(10),
                 "Number of Mixture Components")
                */
                ("fname,f", boost::program_options::value<std::string>(&fname), "File Name");


        boost::program_options::variables_map vm;
        boost::program_options::store(
                boost::program_options::parse_command_line(argc, argv, desc),
                vm);
        boost::program_options::notify(vm);
        if(!fname.empty()) {
            infile = infile + fname;
            outfilename_mt = outfilename_mt + fname;
        }
        if (vm.count("help")) {
            cout << desc << endl;
            vm.clear();
            return 0;
        }
        /*
        if (!infile.empty()) {
            bool ret_value;
            ret_value = train_data.readCSVData(infile + ".ts.data");
            if (!ret_value)
                return 0;
            ret_value = valid_data.readCSVData(infile + ".valid.data");
            if (!ret_value)
                return 0;
            ret_value = test_data.readCSVData(infile + ".test.data");
            if (!ret_value)
                return 0;
            return 1;
        }
         */
        if(infile.empty()){
            cout << desc << endl;
            return 0;
        }

        return 1;
    } catch (exception &e) {
        cerr << "error: " << e.what() << "\n";
        return 0;
    } catch (...) {
        cerr << "Exception of unknown type!\n";
        return 0;
    }
}

int main(int argc, char *argv[]) {
    if (parseOptions(argc, argv) == 0) {
        return 0;
    }
    MT mt;
    mt.read(infile);
    /*
    for(auto it = mt.variables.begin(); it < mt.variables.end()-5; it++){
        auto &var = *(it);
        var->setEvid();
        var->val = 0;
    }
    */
    MT_BTP btp(mt);
    cout << btp.getPE() << endl;
    vector<vector<ldouble>> marginals;
    btp.getVarMarginals(marginals);
    Utils::printMarginals(marginals, outfilename_mt);
}