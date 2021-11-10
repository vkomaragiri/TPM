# TPM

To compile the code into a directory build/, enter cplusplus-code and run cmake:
```
cd cplusplus-code
cmake -B./build CMakeLists.txt
make
```


# To generate samples from the posterior distribution of the oracle BN, run:
# ```
# ./inf-bn <bn-model-directory> <samples-directory> <datafilename> <evidence_percent>
# ```
# This generates the posterior samples file, evidence file, and, oracle posterior probabilities and likelihood weighting posterior probabilities. 
 
# To estimate the BCN weights of the oracle posterior distribution samples, run:
# ```
# ./inf-mcn <mcn-model-directory> <samples-directory> <datafilename> <evidence_percent> <is_data_mcn_model_available>
# ```
# This generates the weights of the BCN model as well as the BCN model learnt from original data.
# 
# 
# 
# 
# 
# 
# 
# To compute the distance measures between distributiond, use:
# ```
# python main_oracle_sample_inf_lw_tmp.py <distance measure>
# ```
# To plot the measures for different evidences and sample values, run:
# ``` 
# python plot_oracle_sample_inf.py <distance measure>
# ```
# Note: The distance measure can take the values {'kld', 'cd'} for {KL-divergence and Chan-Darwiche distance}.

