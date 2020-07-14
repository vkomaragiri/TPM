//
// Created by vasundhara on 12/28/19.
//
//
// Created by Vibhav Gogate on 9/22/19.
//


#include <iomanip>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <chrono>

#include <../include/Data.h>
#include <../include/CLT.h>
#include <../include/MT.h>
#include <../include/HyperParameters.h>

using namespace std;
using namespace std::chrono;

boost::program_options::options_description desc(
        "MTProposal");
string dataset, outfilename_mt = "tempfile.mt", fname;
Data train_data, valid_data, test_data;


int parseOptions(int argc, char *argv[]) {

    try {

        //used (files .ts.data, .test.data, and .valid.data should be present in the directory)
        desc.add_options()
                ("help,?", "produce help message")
                ("dataset,d", boost::program_options::value<std::string>(&dataset), "Dataset (without extensions)")
                ("outfile,o", boost::program_options::value<std::string>(&outfilename_mt), "Store Mixture of Trees")

                ("ivl",
                 boost::program_options::value<int>(&HyperParameters::interval_for_structure_learning)->default_value(
                         10), "Interval for structure learning in EM")
                ("nem", boost::program_options::value<int>(&HyperParameters::num_iterations_em)->default_value(100),
                 "Max Number of iterations for EM")
                ("tol", boost::program_options::value<ldouble>(&HyperParameters::tol)->default_value(1.0e-5),
                 "Tolerance for EM LL scores")
                ("nc", boost::program_options::value<int>(&HyperParameters::num_components)->default_value(10),
                 "Number of Mixture Components")

                ("fname,f", boost::program_options::value<std::string>(&fname), "File Name");


        boost::program_options::variables_map vm;
        boost::program_options::store(
                boost::program_options::parse_command_line(argc, argv, desc),
                vm);
        boost::program_options::notify(vm);
        if(!fname.empty()) {
            dataset = dataset + fname;
            outfilename_mt = outfilename_mt + fname;
        }
        if (vm.count("help")) {
            cout << desc << endl;
            vm.clear();
            return 0;
        }
        if (!dataset.empty()) {
            bool ret_value;
            ret_value = train_data.readCSVData(dataset + ".ts.data");
            if (!ret_value)
                return 0;
            ret_value = valid_data.readCSVData(dataset + ".valid.data");
            if (!ret_value)
                return 0;
            ret_value = test_data.readCSVData(dataset + ".test.data");
            if (!ret_value)
                return 0;
            return 1;
        }
        else {
            cout << desc << endl;
            return 0;
        }
        return 0;
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
    //mt.learnRF(train_data, 50, valid_data);
    train_data.append(valid_data);
    cout << "Starting GD" << endl;
    auto start = high_resolution_clock::now();
    mt.learnEM(train_data);
    //mt.learnGD(train_data, test_data);
    //for(auto func: mt.trees[0].functions){
    //    Utils::print1d(func.potentials);
    //}
    //mt.learnSEM(train_data, 1000);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);

    cout << "Time taken by function: "
         << duration.count() << " seconds" << endl;
    mt.write(outfilename_mt);
    cout << "(test) log-likelihood: " << mt.log_likelihood(test_data) << endl;
    //cout << "(test) log-likelihood: " << mt.log_likelihood(test_data) << endl;
    //cout << "(test) log-likelihood: " << mt.log_likelihood(test_data) << endl;
    /*
    cout << "Test set likelihood = " << Utils::getLLScore(test_data, mt) << endl;
    mt.write(outfilename_mt);
    */

}