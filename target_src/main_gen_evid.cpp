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

#include "../include/CLT.h"
#include "../include/Utils.h"
#include "../include/myRandom.h"

using namespace std;

boost::program_options::options_description desc(
        "GenEvidBN");
string infile, outfile, fname, ev;


int parseOptions(int argc, char *argv[]) {

    try {

        //used (files .ts.data, .test.data, and .valid.data should be present in the directory)
        desc.add_options()
                ("help,?", "produce help message")
                ("infile,i", boost::program_options::value<std::string>(&infile), "Input BN File")
                ("outfile,o", boost::program_options::value<std::string>(&outfile), "Evidence File")
                ("evidence,ev", boost::program_options::value<std::string>(&ev), "Evidence Percent")
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
    CLT bn;
    bn.readUAI08(infile);
    vector<int> order;
    Utils::getTopologicalOrder(bn.variables, bn.functions, order);
    int num_evid = (int)(((float)stoi(ev))/100*bn.variables.size());
    
    if(!outfile.empty()) {
        ofstream out(outfile);
        out << num_evid;
        for (int i = order.size() - 1; i > order.size() - 1 - num_evid; i--) {
            int val = myRandom::getInt(bn.variables[order[i]]->d);
            out << " " << order[i] << " " << val;
        }
        out << endl;
    }
    else{
        cout << num_evid;
        for (int i = order.size() - 1; i > order.size() - 1 - num_evid; i--) {
            int val = myRandom::getInt(bn.variables[order[i]]->d);
            cout << " " << order[i] << " " << val;
        }
        cout << endl;
    }

}