#include <MCN_Sampler.h>

MCN_Sampler::MCN_Sampler(MCN &mcn) {
    mixture_distribution = discrete_distribution<int> (mcn.prob_mixture.begin(), mcn.prob_mixture.end());
    variables = mcn.variables;
    samplers = vector<CN_Sampler> (mcn.ncomponents);
    for(int i = 0; i < mcn.ncomponents; i++){
        samplers[i] = CN_Sampler(mcn.cns[i]);
    }
}

void MCN_Sampler::generateSamples(int n, vector<vector<int>> &samples) {
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

ldouble MCN_Sampler::getProbability(vector<int> &sample) {
    for(auto &var: variables){
        var->t_val = sample[var->id];
    }
    ldouble p = 0.0;
    for(int i = 0; i < samplers.size(); i++){
        p += mixture_distribution.probabilities()[i]*samplers[i].getProbability(sample);
    }
    return p;
}
