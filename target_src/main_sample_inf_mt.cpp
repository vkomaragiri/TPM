//
// Created by vasundhara on 1/30/20.
//

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
#include <MT_Sampler.h>


#include "../include/Utils.h"
#include "../include/MT.h"
#include "../include/MT_Sampler.h"
//#include <MT.h>
//#include <myRandom.h>

using namespace std;

boost::program_options::options_description desc(
        "MTProposal");
string bnfile, mtfile, fname;
ldouble e;


int parseOptions(int argc, char *argv[]) {

    try {

        //used (files .ts.data, .test.data, and .valid.data should be present in the directory)
        desc.add_options()
                ("help,?", "produce help message")
                ("bnfile,b", boost::program_options::value<std::string>(&bnfile), "Input BN Model File")
                ("mtfile,m", boost::program_options::value<std::string>(&mtfile), "Input MT Model File")
                ("e,e", boost::program_options::value<ldouble>(&e), "Percentage of variables that are evidence")
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
        if(bnfile.empty()){
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
    clt.readUAI08(bnfile);
    MT mt;
    mt.read(mtfile);

    vector<vector<int>> samples;
    vector<int> order;
    Utils::getTopologicalOrder(clt.variables, clt.functions, order);
    int ev = e*order.size()/100;

    //Importance Sampling
    for(int i = order.size()-1; i > order.size()-1-ev; i-- ){
        auto &var2 = mt.variables[order[i]];
        var2 -> setValue(0);
    }

    MT_Sampler mts = MT_Sampler(mt);
    mts.generateSamples(100, samples);
    ldouble imp = 0.0;
    for (auto &j: samples){
        imp += clt.getProbability(j)/mts.getProbability(j);
    }
    imp /= samples.size();

    //Likelihood Weighting
    for(int i = order.size()-1; i > order.size()-1-ev; i-- ){
        auto &var = clt.variables[order[i]];
        var->setValue(0);
    }
    BN_Sampler bns = BN_Sampler(clt);
    bns.setEvidence();
    bns.generateSamples(100, samples);
    ldouble lw = 0.0;
    for (auto &j: samples){
        lw += bns.getWeight(j);
    }
    lw /= samples.size();
    cout << imp << " " << lw << endl;
}