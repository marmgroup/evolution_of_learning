// Simulation code for the paper "A neural network model for the evolution of learning in changing environments"					
// by Magdalena Kozielskaand Franz J.Weissing
// (c) Magdalena Kozielska 2023


#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <sys/stat.h>
#include <array>
#include <string>
#include <random>
#include <numeric>
#include "agent.h"
#include "rndutils.hpp"
#include "parameters.h"
#include <algorithm>

// json parameter file is needed to determined the setting of the simulation. See also parameters.cpp and parameters.h
// first generations (number dependents on parameter file) learning episodes fixed and then they can evolve

std::mt19937_64 reng;

auto env_change_dir_dist = std::bernoulli_distribution(0.5); // used if environmental change - increase or decrease

auto fitness_dist = rndutils::mutable_discrete_distribution<int, rndutils::all_zero_policy_uni>{};   //if all ind have fitness of 0, they all have the same chance to be picked for reproduction


	Agent mutate(const Agent &ind, const parameters& p, int gen) {
		std::array<double, 33> new_weights = ind.get_weights();
		double new_lr = ind.get_learning_rate();
		int new_le = ind.get_learning_episodes();
		int lifespan = ind.get_lifespan();

		// distribtuions for mutation of weights
		auto w_mu_chance_dist = std::bernoulli_distribution(p.mut_rate_w);
		
		// distribtuions for mutation of learning rate
		auto lr_mu_chance_dist = std::bernoulli_distribution(p.mut_rate_lr);
		
		// distribtuions for mutation of learning episodes
		auto le_mu_chance_dist = std::bernoulli_distribution(p.mut_rate_le);


		
		// mutation weights
		
		if (p.mut_rate_w != 0.0 && p.mut_step_w !=0.0) {
			
			for (int i = 0; i < 33; ++i) {
				if (w_mu_chance_dist(reng)) {
					auto w_mu_step_dist = std::normal_distribution<double>(0.0, p.mut_step_w);
					new_weights[i] += w_mu_step_dist(reng);
					
				}
			}
		}
		
		// mutation learning rate
		if (p.mut_rate_lr != 0.0 && p.mut_step_lr != 0.0) {
			
			if (lr_mu_chance_dist(reng)) {
				auto lr_mu_step_dist = std::normal_distribution<double>(0.0, p.mut_step_lr);
				new_lr += lr_mu_step_dist(reng);
				
				new_lr = std::max(new_lr, 0.0);  // LR can't be lower than 0
			}
		}
		
		// mutation of the learning epoiseodes
		if (p.mut_rate_le != 0.0 && p.mut_step_le != 0.0 && gen > p.length_fixed_LE) {     //mutation in LE starts only after generation p.length_fixed_LE
			
			if (le_mu_chance_dist(reng)) {
				int le_change = 0;
				auto le_mu_step_dist = std::uniform_int_distribution<int>(-p.mut_step_le, p.mut_step_le);
				do {
					le_change = le_mu_step_dist(reng);  
				} while (le_change == 0);   // do not accept change of zero
				new_le += le_change;
				new_le = std::min(std::max(new_le, 0), lifespan); // constratint learning rate between 0 and lifespan
				
			}
		}
		

		return Agent(new_weights, new_lr, new_le, lifespan);
		
	}

	
	void prepare_output_files(const std::string &file_prefix, int rep) {
		
		const std::string fname_details = file_prefix + "_" + std::to_string(rep) + "_Details.csv";
		const std::string fname_weights = file_prefix + "_" + std::to_string(rep) + "_Weights.csv";
		const std::string fname_averages = file_prefix + "_" + std::to_string(rep) + "_Averages.csv";

		///!!!!!!!!!!!!!!!!!!!!! remove any old file with filename before adding to it  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		std::remove(fname_details.c_str());
		std::remove(fname_weights.c_str());
		std::remove(fname_averages.c_str());

		// for details about individuals
		std::ofstream output_details;
		output_details.open(fname_details, std::ios_base::app);
		assert(output_details.is_open());
		output_details << "Replicate,Generation,Individual,Fitness,Learning_rate,Learning_episodes,Best_fitness,Env_mean_location,Mean_abs_weights,Mean_last_abs_weights,Bias_output\n";
		output_details.close();


		// for weights
		std::ofstream output_weights;
		output_weights.open(fname_weights, std::ios_base::app);
		assert(output_weights.is_open());
		output_weights << "Replicate,Generation,Individual,"
			<< "b2,b3,b4,b5,b6,b7,b8,b9,b10,"
			<< "w12,w13,w14,w15,w26,w27,w28,w29,w36,w37,w38,w39,w46,w47,w48,w49,w56,w57,w58,w59,w610,w710,w810,w910,new_w610,new_w710,new_w810,new_w910\n";
		
		output_weights.close();

		// for averages
		std::ofstream output_averages;
		output_averages.open(fname_averages, std::ios_base::app);
		assert(output_averages.is_open());

		output_averages << "Replicate,Generation,mean fitness,mean learning rate,mean learning episodes,mean best fitness,environmental mean location\n";

		output_averages.close();

		return;
	}
	

	// saving population mean data
	void save_mean_data(const std::string &file_prefix, std::vector<Agent>& population, int replicate, std::vector<double>& fit, std::vector<double>& lr, std::vector<double>& le, std::vector<double>& chosen_e, std::vector<double>& e_mean) {
		
		const std::string fname_averages = file_prefix + "_" + std::to_string(replicate) + "_Averages.csv";

		std::ofstream output_file;
		output_file.open(fname_averages, std::ios_base::app);

		int G = static_cast<int>(fit.size());
		
		for (int gen = 0; gen < G; ++gen) {
			output_file << std::setprecision(10) << replicate << "," << gen << "," << fit[gen] << "," << lr[gen] << "," << le[gen] << "," << chosen_e[gen] << "," << e_mean[gen] << "\n";

		}
		output_file.close();

		return;
	}

	// saving individual data (except individual weights - see below)
	void save_details(const std::string &file_prefix, const std::vector<Agent> &population, int replicate, int generation, double env_mean_loc) {
		const std::string fname_details = file_prefix + "_" + std::to_string(replicate) + "_Details.csv";

		std::ofstream output_file;
		output_file.open(fname_details, std::ios_base::app);
		assert(output_file.is_open());
		
		double mean_w = 0.0; 
		double mean_last_w = 0.0;
		std::array<double, 33> ind_weights;
		
		int N = static_cast<int>(population.size());

		for (int i = 0; i < N; ++i) {

			mean_w = 0.0;
			mean_last_w = 0.0;
			ind_weights = population[i].get_weights();

			for (int w = 0; w < 33; ++w) {
				mean_w += std::abs(ind_weights[w]);
			}
			mean_w = mean_w / 33.0;

			//getting the average value of the final weights, that change in learning

			for (int w = 29; w < 33; ++w) {
				mean_last_w += std::abs(ind_weights[w]);
			}
			mean_last_w = mean_last_w / 4.0;

			output_file << replicate << "," << generation << ',' << i << "," << population[i].get_fitness() << "," << population[i].get_learning_rate() << ',' << population[i].get_learning_episodes() << ','
				<< population[i].get_best_fitness() << ',' << env_mean_loc << ',' << mean_w << "," << mean_last_w << "," << ind_weights[8] << '\n';
		}
		output_file.close();

	}


	// saving individual weights
	void save_weights(const std::string &file_prefix, const std::vector<Agent>& population, int replicate, int generation) {
		const std::string fname_details = file_prefix + "_" + std::to_string(replicate) + "_Weights.csv";

		std::ofstream output_file;
		output_file.open(fname_details, std::ios_base::app);
		assert(output_file.is_open());
		
		std::array<double, 33> ind_weights;

		int N = static_cast<int>(population.size());

		for (int i = 0; i < N; ++i) {

			output_file << replicate << "," << generation << ',' << i;

			ind_weights = population[i].get_weights();
			
			for(int w = 0; w < 33; ++w) {
				output_file << ',' << ind_weights[w];
			}	
			
			ind_weights = population[i].get_new_weights();
			for (int w = 29; w < 33; ++w) {
				output_file << ',' << ind_weights[w];
			}
				
			output_file << '\n';
		}
		output_file.close();
	}

	inline double env_function(double a, double b, double input) {   // baseline distribution of quality of a cue/input - wrapping is done later
		
		return std::exp(-0.5 * ((input - a) * (input - a)) / (b * b));      
	}

	inline double sign(double t) {   // a small function to return a sign of a double as double, positive if t = 0
		return t < 0.0 ? -1.0 : 1.0;
	}

	

int main(int argc, char* argv[])
{
	// getting PARAMETERS from the parameter file
	// !!!!!!!!!!! IMPORTANT !!!!!!!!!!!!!!!!!1
	// the name of the parameter file needs to end with _Parameters.json

	std::cout << argv[1] << std::endl;

	nlohmann::json json_in;
	std::ifstream is(argv[1]);   // the parameter file name must be given as a parameter in the command line
	is >> json_in;
	parameters sim_pars = json_in.get<parameters>();



	std::uniform_real_distribution<double> init_weights_dist(-sim_pars.init_w_range, sim_pars.init_w_range);
	std::uniform_real_distribution<double> init_lr_dist(0.0, sim_pars.init_lr_range);
	std::poisson_distribution<int> init_le_dist(sim_pars.init_le_mean);

	// some check for debuggind
	assert(sim_pars.env_change <= sim_pars.env_range && sim_pars.env_change >= 0.0);
	assert(sim_pars.env_range > 0.0);
	assert(sim_pars.init_mean_env > -sim_pars.env_range && sim_pars.init_mean_env < sim_pars.env_range);

	reng.seed(sim_pars.seed);

	std::string file_name_prefix = argv[1];
	const std::string toRemove = "_Parameters.json";  // the suffix to be removed
	file_name_prefix.erase(file_name_prefix.find(toRemove), toRemove.length());

	std::vector<std::vector<double>> env_cues(sim_pars.lifespan, std::vector<double>(sim_pars.env_sample_size));  //vecor that will contain all the cues encounter by all individuals
	std::vector<std::vector<double>> env_quality(sim_pars.lifespan, std::vector<double>(sim_pars.env_sample_size)); //vector that will contain env. quality for the cues in env_cues

	std::uniform_real_distribution<double> cues_dist(-sim_pars.env_range, sim_pars.env_range);  // distribution from which to draw cues
	std::normal_distribution<double> env_change_dist(sim_pars.env_change, sim_pars.env_change_sd* sim_pars.env_range); // distribution from which actually environmental change is drown

	double actual_env_change = 0.0;

	for (int rep = 0; rep < sim_pars.nr_replicates; ++rep) {  // for each replicate
	
		prepare_output_files(file_name_prefix, rep);
		
		std::vector<Agent> pop;  //population vector

		std::vector<double> fitnesses(sim_pars.N); //vector of fitnesses in the whole population
	
		double mean_env = sim_pars.init_mean_env;

		// vectors to save average population values
		std::vector<double> mean_learning_rates(sim_pars.G + 1, 0.0);
		std::vector<double> mean_learning_episodes(sim_pars.G + 1, 0.0);
		std::vector<double> mean_fitness(sim_pars.G + 1, 0.0);
		std::vector<double> environmental_mean(sim_pars.G + 1, 0.0);
		std::vector<double> mean_chosen_env(sim_pars.G + 1, 0.0);

	
		// INITIALISING POPULATION
		for (int i = 0; i < sim_pars.N; ++i) {  // for each individual

			std::array<double, 33> weights;
			double lr = 0.0;
			double le = 0.0;

			if(sim_pars.init_w_range>0.0){
			
				for (int w = 0; w < 33; ++w) {  //for each weight
					weights[w] = init_weights_dist(reng);
				}
			}
			else {
				weights.fill(0.0);
			}

			if (sim_pars.init_lr_range > 0.0) {
			
				lr = init_lr_dist(reng);
			}

			
			if (sim_pars.length_fixed_LE > 0) {
				le = sim_pars.init_le_mean;  // starting with homogenous population with fixed LE
			}
			else if (std::abs(sim_pars.init_le_mean) > 0) {
			
				le = init_le_dist(reng);
			} // otherwise LE = 0

			pop.push_back(Agent(weights, lr, le, sim_pars.lifespan));
		}
		assert(pop.size() == sim_pars.N);

		//RUN THE CORE OF THE SIMULATION
		for (int gen = 0; gen <= sim_pars.G; ++gen) {  // for each generation

			//creating an new environmental mean if needed
			if (sim_pars.env_change > 0 && (gen > 0) && (gen% sim_pars.env_change_rate == 0)) {
				
				

				actual_env_change = env_change_dist(reng);

				if (sim_pars.env_change_type == "random") { // RANDOM SHIFT + or - env_change 

					if (env_change_dir_dist(reng)) {
						mean_env += actual_env_change;  					
					}
					else {
						mean_env -= actual_env_change;  					
					}
					if (mean_env > sim_pars.env_range) {				// if mean is higher than range it is moved to the lower side of the range
						mean_env = -sim_pars.env_range + (mean_env - sim_pars.env_range);
					}
					else if (mean_env < -sim_pars.env_range) {				// if mean is lower than -range it is moved to the upper side of the range
						mean_env = sim_pars.env_range + (mean_env + sim_pars.env_range);
					}
					
					if (std::abs(mean_env) < 0.00001) { mean_env = 0.0; }    // set to 0 if very small
				}
				else if (sim_pars.env_change_type == "cyclic") { // this settnig was no used for the paper
					mean_env += std::max(0.0,actual_env_change);  // env value always moves in the same direction !!!!!!!!!!!!!!

					if (mean_env > sim_pars.env_range) {				// if mean is higher than range itis moved to the lower side of the range
						mean_env = -sim_pars.env_range + (mean_env - sim_pars.env_range);
					}
					if (std::abs(mean_env) < 0.00001) { mean_env = 0.0; }
				}
				else { return 1; }

			}

			//creating cues matrix
			for (int ls = 0; ls < sim_pars.lifespan; ++ls)
			{
				for (int ess = 0; ess < sim_pars.env_sample_size; ++ess)
				{
					env_cues[ls][ess] = cues_dist(reng);
				}
			}

			// creating coresponding matrix of environmental quality distribution
			for (int ls = 0; ls < sim_pars.lifespan; ++ls)
			{
				for (int ess = 0; ess < sim_pars.env_sample_size; ++ess)
				{	// wrapping quality distribution if needed
					if (env_cues[ls][ess] < mean_env - sim_pars.env_range) {
						env_quality[ls][ess] = env_function(mean_env, sim_pars.sd_env, 2* sim_pars.env_range + env_cues[ls][ess]);
					}
					else if (env_cues[ls][ess] > mean_env + sim_pars.env_range) {
						env_quality[ls][ess] = env_function(mean_env, sim_pars.sd_env, env_cues[ls][ess] - 2 * sim_pars.env_range);
					}
					else {
						env_quality[ls][ess] = env_function(mean_env, sim_pars.sd_env, env_cues[ls][ess]);
					}	
				}
			}

			for (int ind = 0; ind < sim_pars.N; ++ind) {   //for each individual
				pop[ind].learn_and_exploit(env_cues, env_quality);
				mean_learning_rates[gen] += pop[ind].get_learning_rate();
				mean_learning_episodes[gen] += static_cast<double>(pop[ind].get_learning_episodes());
				mean_chosen_env[gen] += pop[ind].get_best_fitness();
				fitnesses[ind] = pop[ind].get_fitness();
			}

			mean_learning_rates[gen] /= static_cast<double>(sim_pars.N);
			mean_learning_episodes[gen] /= static_cast<double>(sim_pars.N);
			mean_chosen_env[gen] /= static_cast<double>(sim_pars.N);
			mean_fitness[gen] = std::accumulate(fitnesses.begin(), fitnesses.end(), 0.0) / static_cast<double>(sim_pars.N);
			environmental_mean[gen] = mean_env;

			// save some output once a while
			if (gen % (sim_pars.G/50) == 0) {
				save_details(file_name_prefix, pop, rep, gen, mean_env);
			
				std::cout << "Replication " << rep << " Generation " << gen << " finished\n";
			}

			// save weights once a while
			if ((gen == 0) || (gen==sim_pars.length_fixed_LE) || (gen > (sim_pars.G-1))) {  
			
				save_weights(file_name_prefix, pop, rep, gen);
			
			}

		
			// REPRODUCTION
		
			std::vector<Agent> new_population;
			
			fitness_dist.mutate(fitnesses.cbegin(), fitnesses.cend());  // probability of becoming a parent depends on fitness value (i.e. resources gained in the foraging period)

			for (int i = 0; i < sim_pars.N; ++i) {
				int next_parent = fitness_dist(reng);
				new_population.push_back(mutate(pop[next_parent], sim_pars, gen));
			}
			assert(static_cast<int>(new_population.size()) == sim_pars.N);

			new_population.swap(pop);


		}  // end generation

	// saving averages
		
			
		save_mean_data(file_name_prefix, pop, rep, mean_fitness, mean_learning_rates, mean_learning_episodes, mean_chosen_env, environmental_mean);

			
	}  // end replicates

	

	return 0;
}


