#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "json.hpp"

struct parameters
{   // values are taken from the parameter file 
	int seed;

	// for initialising population
	double init_w_range;  // weights - uniform distribution from -init_wrange to + init_w_range
	double init_lr_range;  // learning rate - uniform distribution from 0 to init_rl_range
	int init_le_mean;    // learning eposodes fixed or taken from Poisson distribution with this mean
	int length_fixed_LE;

	// parameters for mutation
	double mut_rate_w;
	double mut_step_w;
	double mut_rate_lr;
	double mut_step_lr;
	double mut_rate_le;
	int mut_step_le;

	int N;  // population size
	int G;  // number of generations
	int lifespan;
	int nr_replicates;

	int env_sample_size;  // from how many environments to sample when exploiting
	double env_range;  // range of env input value between -env_range to + env_range (for the paper always between -1 and 1
	double init_mean_env;   // initial location of the env. quality peak - should be within environmental range
	double sd_env;
	double env_change;    // should be lower than env_range and higher than 0 - gives mean of distribution of env change
	double env_change_sd;  //actually coefficient of variation of the environmental change distribution
	std::string env_change_type;   // for now supports "cyclic" (goes around the torus!) and "random" (can be to the left or to the right, randomly)
	int env_change_rate;    // environmental change every n generations
};

void from_json(const nlohmann::json& j, parameters& p);


#endif // SIM_PARAMETERS_H