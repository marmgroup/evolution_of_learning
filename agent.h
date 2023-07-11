#pragma once
#include <array>
#include<vector>

class Agent {
public:
	Agent(const std::array<double, 33> &weights, double learningRate, int learningEpisodes, int lifespan);
	std::array<double, 33> get_weights() const;
	std::array<double, 33> get_new_weights() const;
	double get_learning_rate() const;
	int get_learning_episodes() const;
	double get_fitness() const;
	double get_best_fitness() const;
	double get_lifespan() const;
	double network_calculation(double input);
	void learning(double input, double outout);
	void learn_and_exploit(const std::vector<std::vector<double>> &cues, const std::vector<std::vector<double>> &quality);
	

private:
	const std::array<double, 33> m_weights;  // inherited weights + biases
	std::array<double, 33> m_w;  // weights for learning that can change during the lifetime
	const double m_learning_rate;
	const int m_learning_episodes;
	const int m_lifespan;
	double m_fitness = 0.0; // amount of resources/energy gathered during the foraging period
	double m_best_fitness = 0.0;  // maximum possible amount of resources/energy that could have been gathered during the foraging period if always the best item was chosen

	// Description of the network:
	// 1 input (node 1) - 2 hiden layers each wtih 4 nodes (nodes 2-9) - 1 output (node 10)
	// biases and weights  ( e.g. w 26 - weight connectiong node 2 with 6)
	//	b2		b3		b4		b5		b6		b7		b8		b9		b10	
	//  w[0]	w[1]	w[2]	w[3]	w[4]	w[5]	w[6]	w[7]	w[8]	
	//	w12		w13		w14		w15		w26		w27		w28		w29		w36		w37		w38		w39		w46		w47		w48		w49		w56		w57		w58		w59		w610	w710	w810	w910
	//	w[9]	w[10]	w[11]	w[12]	w[13]	w[14]	w[15]	w[16]	w[17]	w[18]	w[19]	w[20]	w[21]	w[22]	w[23]	w[24]	w[25]	w[26]	w[27]	w[28]	w[29]	w[30]	w[31]	w[32]	
};

double clamped_reLU(double input);

