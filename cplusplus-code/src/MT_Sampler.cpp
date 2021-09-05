//
// Created by vasundhara on 1/30/20.
//

#include "../include/MT_Sampler.h"

MT_Sampler::MT_Sampler(MT &mt) {
    mixture_distribution = discrete_distribution<int> (mt.prob_mixture.begin(), mt.prob_mixture.end());
    variables = mt.variables;
    samplers = vector<BN_Sampler> (mt.ncomponents);
    for(int i = 0; i < mt.ncomponents; i++){
        samplers[i] = BN_Sampler(mt.trees[i]);
    }
}

void MT_Sampler::generateSamples(int n, vector<vector<int>> &samples) {
    vector<int> mix_counts = vector<int> (samplers.size(), 0);
    for(int i = 0; i < n; i++){
        int m = mixture_distribution(myRandom::m_g);
        mix_counts[m]++;
    }
    for(int m = 0; m < samplers.size(); m++){
        vector<vector<int>> samples_m;
        samplers[m].generateSamples(mix_counts[m], samples_m);
        samples.insert(samples.end(), samples_m.begin(), samples_m.end());
    }
}

ldouble MT_Sampler::getProbability(vector<int> &sample) {
    for(auto &var: variables){
        var->t_val = sample[var->id];
    }
    ldouble p = 0.0;
    for(int i = 0; i < samplers.size(); i++){
        p += mixture_distribution.probabilities()[i]*samplers[i].getProbability(sample);
    }
    return p;
}
