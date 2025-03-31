# README

This repository contains everything needed to optimize a parameter configuration of the EASE framework using the NSGAII genetic algorithm.  
Additionally, it is possible to validate the results.

## Proper Usage  
- Ensure that the regressor has been correctly trained and that the corresponding `.joblib` file is placed inside the `regressors` folder.  
- Use the `dataset_generator` file to generate the dataset to be analyzed and the corresponding random configurations.  
  - Currently, this file takes as input a dataset in the format in which the regressor was trained and samples 100 data points.  
- Run the `NSGAII` script, specifying the values for `n_gen` and `pop_size` to be used.  
  - This script saves the optimized configuration.  

## Result Validation  
If you wish to proceed with result validation:  
- Run the script `results_validation/plots_and_tables_NSGAII/results_validation.py`.  
  - You need to have the four optimized configurations generated with different parameter combinations in order to compare them with the random configurations.  
- The script `results_validation/comparison_with_MC/comparison_with_model_checker.py` contains all subsequent validations.  
- The script `results_validation/comparison_with_uppaal/comparison_with_log.py` allows verification of how many configurations would "adapt in time."  

Additionally, a conversion script is provided to adapt the format of the generated configurations to the one required for executing the model checker.
