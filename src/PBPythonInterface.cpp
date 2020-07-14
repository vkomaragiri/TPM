//
// Created by vasundhara on 7/8/20.
//

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <MT.h>
#include <MT_BTP.h>

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
        dt.append(dt_valid);
        mt.learnEM(dt);
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

    ldouble getProbability(vector<int> &s){
        if(!posterior){
           return mt.getProbability(s);
        }
        return mtSampler.getProbability(s);
    }

    py::array generateSamplesDenWeight(){
        sampleWeights = vector<ldouble> (samples.size());
        for(int i = 0; i < samples.size(); i++){
            sampleWeights[i] = getProbability(samples[i]);
        }
        return py::array(sampleWeights.size(), sampleWeights.data());
    }
};

PYBIND11_MODULE(pygmlib, m) {
    m.doc() = "PGM library";

    py::class_<MT_PY_Interface>(m, "MT")
            .def(py::init<>(), "Constructor")
            .def("learn", &MT_PY_Interface::learn, "Learn the MT model from data")
            .def("log_likelihood", &MT_PY_Interface::getLogLikelihood, "Returns the log_likelihood of the model")
            .def("read", &MT_PY_Interface::read, "Reads the MT model stored in a file")
            .def("write", &MT_PY_Interface::write, "Stores the MT model in a file")
            .def("setEvidence", &MT_PY_Interface::setEvidence, "Sets evidence")
            .def("initPosterior", &MT_PY_Interface::initPosterior, "Initializes the posterior inference engine and sampler")
            .def("getPE", &MT_PY_Interface::getPE, "Returns answer to the probability of evidence query.")
            .def("generateSamples", &MT_PY_Interface::generateSamples, "Returns the generated samples in numpy array format.")
            .def("getProbability", &MT_PY_Interface::getProbability, "Returns probability of an assignment")
            .def("generateSampleWeights", &MT_PY_Interface::generateSamplesDenWeight, "Returns the importance sampling denominator for the generated samples");

}
