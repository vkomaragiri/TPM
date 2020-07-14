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


#include <Utils.h>
#include <Data.h>
#include <CLT.h>
//#include <MT.h>
//#include <myRandom.h>

using namespace std;

boost::program_options::options_description desc(
        "MTProposal");
string dataset, outfilename_mt = "tempfile.mt", fname, model_file;
Data train_data, valid_data, test_data;


int parseOptions(int argc, char *argv[]) {

    try {

        //used (files .ts.data, .test.data, and .valid.data should be present in the directory)
        desc.add_options()
                ("help,?", "produce help message")
                ("dataset,d", boost::program_options::value<std::string>(&dataset), "Dataset (without extensions)")
                ("outfile,o", boost::program_options::value<std::string>(&outfilename_mt), "Store Mixture of Trees")
                ("model_file,m", boost::program_options::value<std::string>(&model_file), "Output Model File")
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
    train_data.append(valid_data);
    vector<ldouble> weights = vector<ldouble> (train_data.nexamples, 1.0);
    CLT clt;
    clt.learn(train_data, weights);
    clt.write(model_file);
    cout << "(test) log-likelihood: " << clt.log_likelihood(test_data) << endl;
}