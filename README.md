Simulation code for the paper 
# "A neural network model for the evolution of learning in changing environments"
by Magdalena Kozielska and Franz J.Weissing

Correspondence to: m.a.kozielska@rug.nl or f.j.weissing@rug.nl 


# Overview

We built individual-based simulations in C++ to study the evolution of learning. 
This main "folder" includes the simulation code in c++, an executable file (for Windows), and an example of the parameter file.

The executable file (evolution_of_learning.exe) can be used to run the model on a Windows computer. The parameter json file should be provided as a command line argument to run the simulation.
The parameter file's name has to be a json file and end with _Parameters.json

The folder "FixedLEperiod_LS_500_initLE_20_envSD_0.25_envChange_0.25_envChangeRate_1" contains example output files and the corresponding parameter file for simulation with evolving number of learning episodes (after a period when the number of learning episodes was fixed to 20) for a lifespan of 500, environmental variability σ = 0.25, environmental change m = 0.25 every generation.

The folder "prediction profiles" contains an R script for creating prediction profiles of a network based on its weights.

The folder "summary data" contains:
- Fixed_LE_N_1000_G_50000_mean_last_2000_generations_random_normalDist_025.csv – summary data used for plotting evolutionary outcomes for simulations with a fixed number of learning episodes 
- FixedLEperiod_N_1000_G_50000_initLE_0_mean_last_2000_generations_all_changeRates_random.csv – summary data used for plotting evolutionary outcomes for simulations with no learning.
- FixedLEperiod_N_1000_G_50000_initLE_20_mean_last_2000_generations_all_changeRates_random.csv – summary data used for plotting evolutionary outcomes for simulation where the number of learning episodes could evolve.






