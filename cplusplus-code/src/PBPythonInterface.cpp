//
// Created by vasundhara on 7/8/20.
//

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <MT.h>
#include <MT_BTP.h>
#include <MT_Sampler.h>
#include <CN.h>
#include <CN_BTP.h>
#include <CN_Sampler.h>
#include <MCN.h>
#include <MCN_BTP.h>
#include <MCN_Sampler.h>

namespace py = pybind11;


class MT_PY_Interface{
    MT mt;
    MT_BTP mtBtp;
    MT_Sampler mtSampler;

    bool posterior = false;

public:
    vector<vector<int>> samples;
    vector<ldouble> sampleWeights;
    MT_PY_Interface() = default;

    void learn(string train_filename, string valid_filename){
        mt = MT();
        Data dt = Data();
        dt.readCSVData(train_filename);
        Data dt_valid = Data();
        dt_valid.readCSVData(valid_filename);
        //dt.append(dt_valid);
        mt.learnEM(dt, dt_valid);
    }

    ldouble getLogLikelihood(string filename){
        Data dt = Data();
        dt.readCSVData(filename);
        return mt.log_likelihood(dt);
    }

    void read(string filename){
        mt = MT();
        mt.read(filename);
    }

    void write(string filename){
        mt.write(filename);
    }

    void setEvidence(int var, int val){
        mt.setEvidence(var, val);
    }

    void initPosterior(){
        posterior = true;
        mtBtp = MT_BTP(mt);
        mtSampler = MT_Sampler();
        mtBtp.getPosteriorSampler(mtSampler);
    }

    ldouble getPE(){
        return mtBtp.getPE();
    }

    py::array getVarMarginals(){
        vector<vector<ldouble>> marginals;
        mtBtp.getVarMarginals(marginals);

        size_t N = marginals.size();
        size_t M = marginals[0].size();
        py::array_t<ldouble, py::array::c_style> ret({N, M});
        auto ra = ret.mutable_unchecked();

        for(size_t i = 0; i < N; i++){
            for(size_t j = 0; j < M; j++){
                ra(i, j) = marginals[i][j];
            }
        }
        return ret;
    }

    py::array generateSamples(int n){
        if(!posterior){
            mtSampler = MT_Sampler(mt);
        }
        samples.clear();
        mtSampler.generateSamples(n, samples);
        size_t N = samples.size();
        size_t M = samples[0].size();
        py::array_t<int, py::array::c_style> ret({N, M});
        auto ra = ret.mutable_unchecked();

        for(size_t i = 0; i < N; i++){
            for(size_t j = 0; j < M; j++){
                ra(i, j) = samples[i][j];
            }
        }
        return ret;
    }

    void writeSamples(string outfile) {
        Utils::printSamples(samples, outfile);
    }

    ldouble getPriorProbability(vector<int> &s){
        return mt.getProbability(s);
    }

    ldouble getPosteriorProbability(vector<int> &s){
        return mtSampler.getProbability(s);
    }

    ldouble getProbability(vector<int> &s){
        if(posterior){
           return getPosteriorProbability(s);
        }
        return getPriorProbability(s);
    }

    py::array getProbability(py::array_t<ldouble> a){
        auto r = a.unchecked<2>();
        vector<vector<int>> samples2 = vector<vector<int>>(r.shape(0));
        for (ssize_t i = 0; i < r.shape(0); i++) {
            samples2[i] = vector<int>(r.shape(1));
            for (ssize_t j = 0; j < r.shape(1); j++) {
                samples2[i][j] = r(i, j);
            }
        }
        vector<ldouble> samples2Weights = vector<ldouble> (samples2.size());
        for(int i = 0; i < samples2.size(); i++){
                samples2Weights[i] = getProbability(samples2[i]);
        }
        return py::array(samples2Weights.size(), samples2Weights.data());
    }

    py::array generateISDenWeights(){
        sampleWeights = vector<ldouble> (samples.size());
        for(int i = 0; i < samples.size(); i++){
            sampleWeights[i] = getPosteriorProbability(samples[i]);
        }
        return py::array(sampleWeights.size(), sampleWeights.data());
    }

    py::array generateISDenWeights(int n){
        sampleWeights = vector<ldouble> (n);
        for(int i = 0; i < n; i++){
            sampleWeights[i] = getPosteriorProbability(samples[i]);
        }
        return py::array(n, sampleWeights.data());
    }

    void updateParamsOnline(py::array_t<ldouble> a, ldouble learning_rate){
        auto r = a.unchecked<2>();
        vector<vector<int>> data = vector<vector<int>>(r.shape(0));
        for (ssize_t i = 0; i < r.shape(0); i++) {
            data[i] = vector<int>(r.shape(1));
            for (ssize_t j = 0; j < r.shape(1); j++) {
                data[i][j] = r(i, j);
            }
        }
        mt.initGrad();
        mt.compGrad(data);
        mt.doSGDUpdate(learning_rate/data.size());
    }

    ldouble computeGradSqNorm(py::array_t<ldouble> a, ldouble learning_rate){
        auto r = a.unchecked<2>();
        vector<vector<int>> data = vector<vector<int>>(r.shape(0));
        for (ssize_t i = 0; i < r.shape(0); i++) {
            data[i] = vector<int>(r.shape(1));
            for (ssize_t j = 0; j < r.shape(1); j++) {
                data[i][j] = r(i, j);
            }
        }
        mt.initGrad();
        mt.compGrad(data);
        mt.doSGDUpdate(learning_rate);
        return mt.gradSqNorm();
    }
};

class CN_PY_Interface{
    CN cn;
    CN_BTP cnBtp;
    CN_Sampler cnSampler;

    bool posterior;

    public:
    vector<vector<int>> samples;
    vector<ldouble> sampleWeights;

    CN_PY_Interface() = default;

    void learn(string train_filename, string valid_filename){
        cn = CN();
        Data dt = Data();
        dt.readCSVData(train_filename);
        Data dt_valid = Data();
        dt_valid.readCSVData(valid_filename);
        cn.learn(dt, dt_valid, true, false, false);
    }

    ldouble getLogLikelihood(string filename){
        Data dt = Data();
        dt.readCSVData(filename);
        return cn.log_likelihood(dt);
    }

    void setEvidence(int var, int val){
        cn.setEvidence(var, val);
    }

    void initPosterior(){
        posterior = true;
        cnBtp = CN_BTP(cn);
        cnBtp.getPosteriorSampler(cnSampler);
    }

    ldouble getPE(){
        return cnBtp.getPE();
    }

    py::array getVarMarginals(){
        vector<vector<ldouble>> marginals;
        cnBtp.getVarMarginals(marginals);

        size_t N = marginals.size();
        size_t M = marginals[0].size();
        py::array_t<ldouble, py::array::c_style> ret({N, M});
        auto ra = ret.mutable_unchecked();

        for(size_t i = 0; i < N; i++){
            for(size_t j = 0; j < M; j++){
                ra(i, j) = marginals[i][j];
            }
        }
        return ret;
    }

    py::array generateSamples(int n){
        if(!posterior){
            cnSampler = CN_Sampler(cn);
        }
        samples.clear();
        cnSampler.generateSamples(n, samples);
        size_t N = samples.size();
        size_t M = samples[0].size();
        py::array_t<int, py::array::c_style> ret({N, M});
        auto ra = ret.mutable_unchecked();

        for(size_t i = 0; i < N; i++){
            for(size_t j = 0; j < M; j++){
                ra(i, j) = samples[i][j];
            }
        }
        return ret;
    }

    ldouble getPosteriorProbability(vector<int> &s){
        return cnSampler.getProbability(s);
    }

    py::array generateISDenWeights(){
        sampleWeights = vector<ldouble> (samples.size());
        for(int i = 0; i < samples.size(); i++){
            sampleWeights[i] = getPosteriorProbability(samples[i]);
        }
        return py::array(sampleWeights.size(), sampleWeights.data());
    }

    void read(string filename){
        cn = CN();
        cn.read(filename);
    }

    void write(string filename){
        cn.write(filename);
    }

    void updateParamsOnline(py::array_t<ldouble> a, ldouble learning_rate){
        auto r = a.unchecked<2>();
        vector<vector<int>> data = vector<vector<int>>(r.shape(0));
        vector<int> partition = vector<int>(r.shape(0));
        for (ssize_t i = 0; i < r.shape(0); i++) {
            data[i] = vector<int>(r.shape(1));
            for (ssize_t j = 0; j < r.shape(1); j++) {
                data[i][j] = r(i, j);
            }
            partition[i] = i;
        }
        cn.initGrad();
        cn.compGrad(data, partition, cn.getRoot());
        cn.doSGDUpdate(learning_rate/data.size());
    }

    ldouble computeGradSqNorm(py::array_t<ldouble> a, ldouble learning_rate){
        auto r = a.unchecked<2>();
        vector<vector<int>> data = vector<vector<int>>(r.shape(0));
        vector<int> partition = vector<int>(r.shape(0));
        for (ssize_t i = 0; i < r.shape(0); i++) {
            data[i] = vector<int>(r.shape(1));
            for (ssize_t j = 0; j < r.shape(1); j++) {
                data[i][j] = r(i, j);
            }
            partition[i] = i;
        }
        cn.initGrad();
        cn.compGrad(data, partition, cn.getRoot());
        cn.doSGDUpdate(learning_rate);
        return cn.gradSqNorm(cn.getRoot());
    }
};

class MCN_PY_Interface{
    MCN mcn;
    MCN_BTP mcnBtp;
    MCN_Sampler mcnSampler;

    bool posterior;

    public:
    vector<vector<int>> samples;
    vector<ldouble> sampleWeights;

    MCN_PY_Interface() = default;

    void learn(string train_filename, string valid_filename){
        mcn = MCN();
        Data dt = Data();
        dt.readCSVData(train_filename);
        Data dt_valid = Data();
        dt_valid.readCSVData(valid_filename);
        mcn.learn(dt, dt_valid);
    }

    ldouble getLogLikelihood(string filename){
        Data dt = Data();
        dt.readCSVData(filename);
        return mcn.log_likelihood(dt);
    }

    void setEvidence(int var, int val){
        mcn.setEvidence(var, val);
    }

    void initPosterior(){
        posterior = true;
        mcnBtp = MCN_BTP(mcn);
        mcnBtp.getPosteriorSampler(mcnSampler);
    }

    ldouble getPE(){
        return mcnBtp.getPE();
    }

    py::array getVarMarginals(){
        vector<vector<ldouble>> marginals;
        mcnBtp.getVarMarginals(marginals);

        size_t N = marginals.size();
        size_t M = marginals[0].size();
        py::array_t<ldouble, py::array::c_style> ret({N, M});
        auto ra = ret.mutable_unchecked();

        for(size_t i = 0; i < N; i++){
            for(size_t j = 0; j < M; j++){
                ra(i, j) = marginals[i][j];
            }
        }
        return ret;
    }

    py::array generateSamples(int n){
        if(!posterior){
            mcnSampler = MCN_Sampler(mcn);
        }
        samples.clear();
        mcnSampler.generateSamples(n, samples);
        size_t N = samples.size();
        size_t M = samples[0].size();
        py::array_t<int, py::array::c_style> ret({N, M});
        auto ra = ret.mutable_unchecked();

        for(size_t i = 0; i < N; i++){
            for(size_t j = 0; j < M; j++){
                ra(i, j) = samples[i][j];
            }
        }
        return ret;
    }

    ldouble getPosteriorProbability(vector<int> &s){
        return mcnSampler.getProbability(s);
    }

    py::array generateISDenWeights(){
        sampleWeights = vector<ldouble> (samples.size());
        for(int i = 0; i < samples.size(); i++){
            sampleWeights[i] = getPosteriorProbability(samples[i]);
        }
        return py::array(sampleWeights.size(), sampleWeights.data());
    }

    void read(string filename){
        mcn = MCN();
        mcn.read(filename);
    }

    void write(string filename){
        mcn.write(filename);
    }

    void updateParamsOnline(py::array_t<ldouble> a, ldouble learning_rate){
        auto r = a.unchecked<2>();
        vector<vector<int>> data = vector<vector<int>>(r.shape(0));
        for (ssize_t i = 0; i < r.shape(0); i++) {
            data[i] = vector<int>(r.shape(1));
            for (ssize_t j = 0; j < r.shape(1); j++) {
                data[i][j] = r(i, j);
            }
        }
        mcn.initGrad();
        mcn.compGrad(data);
        mcn.doSGDUpdate(learning_rate/data.size());
    }

    ldouble computeGradSqNorm(py::array_t<ldouble> a, ldouble learning_rate){
        auto r = a.unchecked<2>();
        vector<vector<int>> data = vector<vector<int>>(r.shape(0));
        for (ssize_t i = 0; i < r.shape(0); i++) {
            data[i] = vector<int>(r.shape(1));
            for (ssize_t j = 0; j < r.shape(1); j++) {
                data[i][j] = r(i, j);
            }
        }
        mcn.initGrad();
        mcn.compGrad(data);
        mcn.doSGDUpdate(learning_rate);
        return mcn.gradSqNorm();
    }

    void addComps(string train_filename, string valid_filename, py::array_t<ldouble> a){
        auto r = a.unchecked<3>();
        vector<vector<vector<int>>> bags = vector<vector<vector<int>>>(r.shape(0));
        for (ssize_t i = 0; i < r.shape(0); i++) {
            bags[i] = vector<vector<int>>(r.shape(1));
            for (ssize_t j = 0; j < r.shape(1); j++) {
                bags[i][j] = vector<int>(r.shape(2));
                for(ssize_t k = 0; k < r.shape(2); k++){
                    bags[i][j][k] = r(i, j, k);
                }
            }
        }
        Data td = Data();
        td.readCSVData(train_filename);
        Data valid_data = Data();
        valid_data.readCSVData(valid_filename);
        mcn.addComps(bags, td.dsize, valid_data);
    }
};

class BN_UAI_PY_Interface{
    CLT clt;
    BN_Sampler bns;

public:
    vector<vector<int>> samples;
    vector<ldouble> sampleWeights;
    BN_UAI_PY_Interface() = default;

    void read(string infile){
        cout << "In read" << endl;
        clt = CLT();
        clt.readUAI08(infile);
    }

    void learn(string train_filename){
        clt = CLT();
        Data dt = Data();
        dt.readCSVData(train_filename);
        clt.learn(dt);
    }

    void write(string filename){
        clt.write(filename);
    }

    ldouble getLogLikelihood(string filename){
        Data dt = Data();
        dt.readCSVData(filename);
        return clt.log_likelihood(dt);
    }

    void setEvidence(int var, int val){
        clt.setEvidence(var, val);
    }

    ldouble getPriorProbability(vector<int> &s){
        return clt.getProbability(s);
    }

    py::array generatePriorSamples(int n){
        samples.clear();
        bns = BN_Sampler(clt);
        bns.generateSamples(n, samples);
        size_t N = samples.size();
        size_t M = samples[0].size();
        py::array_t<int, py::array::c_style> ret({N, M});
        auto ra = ret.mutable_unchecked();

        for(size_t i = 0; i < N; i++){
            for(size_t j = 0; j < M; j++){
                ra(i, j) = samples[i][j];
            }
        }
        return ret;
    }


    ldouble generateLWSamples(int n){
        ldouble pe = 0.0;
        sampleWeights = vector<ldouble>(n, 1.0);
        bns = BN_Sampler(clt);
        bns.setEvidence();
        bns.generateSamples(n, samples);
        for(int i = 0; i < n; i++){
            ldouble temp = exp(bns.getLogWeight(samples[i]));
            sampleWeights[i] = temp;
            pe += temp;
            //Utils::print1d(samples[i]);
            //cout << clt.getProbability(samples[i]) << " " << bns.getProbability(samples[i]) << " " << sampleWeights[i] << endl;
        }
        pe /= n;
        return log(pe);
    }

    py::array computePosteriorMarginals(){
        size_t N = samples[0].size();
        size_t M = 2; //Assuming binary variables
        py::array_t<ldouble, py::array::c_style> ret({N, M});
        auto ra = ret.mutable_unchecked();
        for(int j = 0; j < samples[0].size(); j++){
            ra(j, 0) = 0;
            ra(j, 1) = 0;
            for(int i = 0; i < samples.size(); i++){
                ra(j, samples[i][j]) += sampleWeights[i];
            }
            ldouble norm_const = ra(j, 0)+ra(j, 1);
            ra(j, 0) /= norm_const;
            ra(j, 1) /= norm_const;
        }
        return ret;
    }

    py::array generateISNumWeights(){
        sampleWeights = vector<ldouble> (samples.size());
        for(int i = 0; i < samples.size(); i++){
            sampleWeights[i] = getPriorProbability(samples[i]);
        }
        return py::array(sampleWeights.size(), sampleWeights.data());
    }

    py::array generateISNumWeights(int n){
        sampleWeights = vector<ldouble> (n);
        for(int i = 0; i < n; i++){
            sampleWeights[i] = getPriorProbability(samples[i]);
        }
        return py::array(n, sampleWeights.data());
    }

    py::array generateISNumWeights(py::array_t<ldouble> a){
        auto r = a.unchecked<2>();
        samples = vector<vector<int>>(r.shape(0));
        for (ssize_t i = 0; i < r.shape(0); i++) {
            samples[i] = vector<int>(r.shape(1));
            for (ssize_t j = 0; j < r.shape(1); j++) {
                samples[i][j] = r(i, j);
                //cout << r(i, j) << " ";
            }
            //cout << endl;
        }
        sampleWeights = vector<ldouble> (samples.size());
        for(int i = 0; i < samples.size(); i++){
            sampleWeights[i] = clt.getProbability(samples[i]);
        }
        return py::array(samples.size(), sampleWeights.data());
    }

    void updateParamsOnline(py::array_t<ldouble> a, ldouble learning_rate){
        auto r = a.unchecked<2>();
        vector<vector<int>> data = vector<vector<int>>(r.shape(0));
        vector<int> partition = vector<int>(r.shape(0));
        for (ssize_t i = 0; i < r.shape(0); i++) {
            data[i] = vector<int>(r.shape(1));
            for (ssize_t j = 0; j < r.shape(1); j++) {
                data[i][j] = r(i, j);
            }
            partition[i] = i;
        }
        clt.initGrad();
        clt.compGrad(data, partition);
        clt.doSGDUpdate(learning_rate/data.size());
    }

};

PYBIND11_MODULE(pygmlib, m) {
    m.doc() = "PGM library";

    py::class_<MT_PY_Interface>(m, "MT")
            .def(py::init<>(), "Constructor")
            .def("learn", &MT_PY_Interface::learn, "Learn the MT model from data")
            .def("log_likelihood", &MT_PY_Interface::getLogLikelihood, "Returns the log_likelihood of the model")
            .def("read", &MT_PY_Interface::read, "Reads the MT model stored from a file")
            .def("write", &MT_PY_Interface::write, "Stores the MT model in a file")
            .def("setEvidence", &MT_PY_Interface::setEvidence, "Sets evidence")
            .def("initPosterior", &MT_PY_Interface::initPosterior, "Initializes the posterior inference engine and sampler")
            .def("getPE", &MT_PY_Interface::getPE, "Returns answer to the probability of evidence query.")
            .def("getVarMarginals", &MT_PY_Interface::getVarMarginals, "Returns the posterior marginal probabilities of variables")
            .def("generateSamples", &MT_PY_Interface::generateSamples, "Returns the generated samples in numpy array format.")
            .def("writeSamples", &MT_PY_Interface::writeSamples, "Writes samples generated in generateSamples to outfile")
            .def("getPriorProbability", &MT_PY_Interface::getPriorProbability, "Returns the prior probability of an assignment")
            .def("getPosteriorProbability", &MT_PY_Interface::getPosteriorProbability, "Returns the posterior probability of an assignment")
            .def("getProbability", (ldouble (MT_PY_Interface::*)(vector<int> &)) &MT_PY_Interface::getProbability, "Returns probability of an assignment")
            .def("getProbability", (py::array (MT_PY_Interface::*)(py::array_t<ldouble> )) &MT_PY_Interface::getProbability, "Returns probability of an assignment")
            .def("generateISDenWeights", (py::array (MT_PY_Interface::*)()) &MT_PY_Interface::generateISDenWeights, "Returns the importance sampling denominator for the generated samples")
            .def("generateISDenWeights", (py::array (MT_PY_Interface::*)(int)) &MT_PY_Interface::generateISDenWeights, "Returns the importance sampling denominator for the generated samples")
            .def("updateParams", &MT_PY_Interface::updateParamsOnline, "Performs one iteration of SGD on model parameters")
            .def("gradSqNorm", &MT_PY_Interface::computeGradSqNorm, "Returns square of the norm of the gradient")
            ;

    py::class_<CN_PY_Interface>(m, "CN")
            .def(py::init<>(), "Constructor")
            .def("learn", &CN_PY_Interface::learn, "Learn the MT model from data")
            .def("log_likelihood", &CN_PY_Interface::getLogLikelihood, "Returns the log_likelihood of the model")
            .def("read", &CN_PY_Interface::read, "Reads the CN model stored from a file")
            .def("write", &CN_PY_Interface::write, "Stores the CN model in a file")
            .def("setEvidence", &CN_PY_Interface::setEvidence, "Sets evidence")
            .def("initPosterior", &CN_PY_Interface::initPosterior, "Initializes the posterior inference engine and sampler")
            .def("getPE", &CN_PY_Interface::getPE, "Returns answer to the probability of evidence query.")
            .def("getVarMarginals", &CN_PY_Interface::getVarMarginals, "Returns the posterior marginal probabilities of variables")
            .def("generateSamples", &CN_PY_Interface::generateSamples, "Returns the generated samples in numpy array format.")
            .def("getPosteriorProbability", &CN_PY_Interface::getPosteriorProbability, "Returns the posterior probability of an assignment")
            .def("generateISDenWeights", (py::array (CN_PY_Interface::*)()) &CN_PY_Interface::generateISDenWeights, "Returns the importance sampling denominator for the generated samples")
            .def("updateParams", &CN_PY_Interface::updateParamsOnline, "Performs one iteration of SGD on model parameters")
            .def("gradSqNorm", &CN_PY_Interface::computeGradSqNorm, "Returns square of the norm of the gradient")
            ;
    py::class_<MCN_PY_Interface>(m, "MCN")
            .def(py::init<>(), "Constructor")
            .def("learn", &MCN_PY_Interface::learn, "Learn the MT model from data")
            .def("log_likelihood", &MCN_PY_Interface::getLogLikelihood, "Returns the log_likelihood of the model")
            .def("read", &MCN_PY_Interface::read, "Reads the MCN model stored from a file")
            .def("write", &MCN_PY_Interface::write, "Stores the MCN model in a file")
            .def("setEvidence", &MCN_PY_Interface::setEvidence, "Sets evidence")
            .def("initPosterior", &MCN_PY_Interface::initPosterior, "Initializes the posterior inference engine and sampler")
            .def("getPE", &MCN_PY_Interface::getPE, "Returns answer to the probability of evidence query.")
            .def("getVarMarginals", &MCN_PY_Interface::getVarMarginals, "Returns the posterior marginal probabilities of variables")
            .def("generateSamples", &MCN_PY_Interface::generateSamples, "Returns the generated samples in numpy array format.")
            .def("getPosteriorProbability", &MCN_PY_Interface::getPosteriorProbability, "Returns the posterior probability of an assignment")
            .def("generateISDenWeights", (py::array (MCN_PY_Interface::*)()) &MCN_PY_Interface::generateISDenWeights, "Returns the importance sampling denominator for the generated samples")
            .def("updateParams", &MCN_PY_Interface::updateParamsOnline, "Performs one iteration of SGD on model parameters")
            .def("gradSqNorm", &MCN_PY_Interface::computeGradSqNorm, "Returns square of the norm of the gradient")
            .def("addComps", &MCN_PY_Interface::addComps, "Add components to Bags of cutset networks")
            ;
    py::class_<BN_UAI_PY_Interface>(m, "BN_UAI")
            .def(py::init<>(), "Constructor")
            .def("read", &BN_UAI_PY_Interface::read, "Reads the BN from a UAI format file")
            .def("learn", &BN_UAI_PY_Interface::learn, "Learn the CLT model from data")
            .def("write", &BN_UAI_PY_Interface::write, "Stores the CLT model in a file")
            .def("log_likelihood", &BN_UAI_PY_Interface::getLogLikelihood, "Returns the log likelihood of the model")
            .def("setEvidence", &BN_UAI_PY_Interface::setEvidence, "Sets evidence")
            //.def("initPosterior", &BN_UAI_PY_Interface::initPosterior, "Initializes the posterior inference engine and sampler")
            //.def("getPE", &BN_UAI_PY_Interface::getPE, "Returns answer to the probability of evidence query.")
            .def("generatePriorSamples", &BN_UAI_PY_Interface::generatePriorSamples, "Generates samples using the BN in the .uai format")
            .def("generateLWSamples", &BN_UAI_PY_Interface::generateLWSamples, "Generates LW samples and returns the weights")
            .def("getPriorProbability", (ldouble (BN_UAI_PY_Interface::*)(vector<int> &)) &BN_UAI_PY_Interface::getPriorProbability, "Returns probability of an assignment")
            .def("computePosteriorMarginals", &BN_UAI_PY_Interface::computePosteriorMarginals, "Computes marginals based on LW Samples")
            //.def("getProbability", (py::array (BN_UAI_PY_Interface::*)(py::array_t<ldouble> )) &BN_UAI_PY_Interface::getProbability, "Returns probability of an assignment")
            .def("generateISNumWeights", (py::array (BN_UAI_PY_Interface::*)()) &BN_UAI_PY_Interface::generateISNumWeights, "Returns the importance sampling numerator for the generated samples")
            .def("generateISNumWeights", (py::array (BN_UAI_PY_Interface::*)(int)) &BN_UAI_PY_Interface::generateISNumWeights, "Returns the importance sampling numerator for the generated samples")
            .def("generateISNumWeights", (py::array (BN_UAI_PY_Interface::*)(py::array_t<ldouble>)) &BN_UAI_PY_Interface::generateISNumWeights, "Returns the importance sampling numerator for the generated samples")
            //.def("generateLWWeights", (py::array (BN_UAI_PY_Interface::*)()) &BN_UAI_PY_Interface::generateLWWeights, "Returns the likelihood weighting weights for the generated samples")
            //.def("generateLWWeights", (py::array (BN_UAI_PY_Interface::*)(int)) &BN_UAI_PY_Interface::generateLWWeights, "Returns the likelihood weighting weights for the generated samples")
            .def("updateParams", &BN_UAI_PY_Interface::updateParamsOnline, "Performs one iteration of SGD on model parameters")
            ;
}
