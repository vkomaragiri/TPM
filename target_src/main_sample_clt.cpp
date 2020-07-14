//
// Created by vasundhara on 1/30/20.
//

//
// Created by Vibhav Gogate on 9/22/19.
//


#include <iomanip>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <BN_Sampler.h>


#include "../include/Utils.h"
#include "../include/CLT.h"
//#include <MT.h>
//#include <myRandom.h>

using namespace std;

boost::program_options::options_description desc(
        "MTProposal");
string infile, outfile= "tempfile.data", fname;


int parseOptions(int argc, char *argv[]) {

    try {

        //used (files .ts.data, .test.data, and .valid.data should be present in the directory)
        desc.add_options()
                ("help,?", "produce help message")
                ("infile,i", boost::program_options::value<std::string>(&infile), "Input Probabilistic Model File")
                ("outfile,o", boost::program_options::value<std::string>(&outfile), "Store Samples")
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
            //dataset = dataset + fname;
            //outfilename_mt = outfilename_mt + fname;
        }
        if(infile.empty()){
            cout << desc << endl;
            return 0;
        }
        if (vm.count("help")) {
            cout << desc << endl;
            vm.clear();
            return 0;
        }
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
    CLT clt;
    clt.readUAI08(infile);
    /*
    for(auto it = clt.variables.begin(); it < clt.variables.end()-5; it++){
        auto &var = *(it);
        var->setValue(0);
    }
    */
    //BTP btp(clt);

    BN_Sampler bns = BN_Sampler(clt);
    bns.setEvidence();
    vector<vector<int>> samples;
    bns.generateSamples(300, samples);
    Utils::printSamples(samples, outfile);
}