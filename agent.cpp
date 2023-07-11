#include "agent.h"
#include <random>
#include <cassert>

constexpr double PI = 3.14159265;

extern std::mt19937_64 reng;

Agent::Agent(const std::array<double, 33> &weights, double learningRate, int learningEpisodes, int lifespan) :
	m_weights(weights), m_w(weights), m_learning_rate(learningRate), m_learning_episodes(learningEpisodes), m_lifespan(lifespan)
{	
}

std::array<double, 33> Agent::get_weights() const {
	return m_weights;
}

std::array<double, 33> Agent::get_new_weights() const {
	return m_w;
}

double Agent::get_learning_rate() const {
	return m_learning_rate;
}

int Agent::get_learning_episodes() const {
	return m_learning_episodes;
}

double Agent::get_fitness() const {
	return m_fitness;
}

double Agent::get_best_fitness() const {
	return m_best_fitness;
}

double Agent::get_lifespan() const {
	return m_lifespan;
}

double Agent::network_calculation(double input) {  // calculates the ouput of the network given the input
	// Description of the network:
	// 1 input (node 1) - 2 hiden layers each wtih 4 nodes (nodes 2-9) - 1 output (node 10)
	// biases and weights  ( e.g. w 26 - weight connectiong node 2 with 6)
	//	b2		b3		b4		b5		b6		b7		b8		b9		b10	
	//  w[0]	w[1]	w[2]	w[3]	w[4]	w[5]	w[6]	w[7]	w[8]	
	//	w12		w13		w14		w15		w26		w27		w28		w29		w36		w37		w38		w39		w46		w47		w48		w49		w56		w57		w58		w59		w610	w710	w810	w910
	//	w[9]	w[10]	w[11]	w[12]	w[13]	w[14]	w[15]	w[16]	w[17]	w[18]	w[19]	w[20]	w[21]	w[22]	w[23]	w[24]	w[25]	w[26]	w[27]	w[28]	w[29]	w[30]	w[31]	w[32]

	double N2 = 0.0, N3 = 0.0, N4 = 0.0, N5 = 0.0, N6 = 0.0, N7 = 0.0, N8 = 0.0, N9 = 0.0;
	double output = 0.0;

	N2 = clamped_reLU(m_w[9] * input + m_w[0]);
	N3 = clamped_reLU(m_w[10] * input + m_w[1]);
	N4 = clamped_reLU(m_w[11] * input + m_w[2]);
	N5 = clamped_reLU(m_w[12] * input + m_w[3]);

	N6 = clamped_reLU(m_w[13] * N2 + m_w[17] * N3 + m_w[21] * N4 + m_w[25] * N5 + m_w[4]);
	N7 = clamped_reLU(m_w[14] * N2 + m_w[18] * N3 + m_w[22] * N4 + m_w[26] * N5 + m_w[5]);
	N8 = clamped_reLU(m_w[15] * N2 + m_w[19] * N3 + m_w[23] * N4 + m_w[27] * N5 + m_w[6]);
	N9 = clamped_reLU(m_w[16] * N2 + m_w[20] * N3 + m_w[24] * N4 + m_w[28] * N5 + m_w[7]);

	output = m_w[29] * N6 + m_w[30] * N7 + m_w[31] * N8 + m_w[32] * N9 + m_w[8];

	return output;

}

void Agent::learning(double input, double expect) {    // one round of calculation + updating weights
	// Description of the network:
	// 1 input (node 1) - 2 hiden layers each wtih 4 nodes (nodes 2-9) - 1 output (node 10)
	// biases and weights  ( e.g. w 26 - weight connectiong node 2 with 6)
	//	b2		b3		b4		b5		b6		b7		b8		b9		b10	
	//  w[0]	w[1]	w[2]	w[3]	w[4]	w[5]	w[6]	w[7]	w[8]	
	//	w12		w13		w14		w15		w26		w27		w28		w29		w36		w37		w38		w39		w46		w47		w48		w49		w56		w57		w58		w59		w610	w710	w810	w910
	//	w[9]	w[10]	w[11]	w[12]	w[13]	w[14]	w[15]	w[16]	w[17]	w[18]	w[19]	w[20]	w[21]	w[22]	w[23]	w[24]	w[25]	w[26]	w[27]	w[28]	w[29]	w[30]	w[31]	w[32]

	double N2 = 0.0, N3 = 0.0, N4 = 0.0, N5 = 0.0, N6 = 0.0, N7 = 0.0, N8 = 0.0, N9 = 0.0;
	double output = 0.0;
	double error; 

	N2 = clamped_reLU(m_w[9] * input + m_w[0]);
	N3 = clamped_reLU(m_w[10] * input + m_w[1]);
	N4 = clamped_reLU(m_w[11] * input + m_w[2]);
	N5 = clamped_reLU(m_w[12] * input + m_w[3]);

	N6 = clamped_reLU(m_w[13] * N2 + m_w[17] * N3 + m_w[21] * N4 + m_w[25] * N5 + m_w[4]);
	N7 = clamped_reLU(m_w[14] * N2 + m_w[18] * N3 + m_w[22] * N4 + m_w[26] * N5 + m_w[5]);
	N8 = clamped_reLU(m_w[15] * N2 + m_w[19] * N3 + m_w[23] * N4 + m_w[27] * N5 + m_w[6]);
	N9 = clamped_reLU(m_w[16] * N2 + m_w[20] * N3 + m_w[24] * N4 + m_w[28] * N5 + m_w[7]);

	output = m_w[29] * N6 + m_w[30] * N7 + m_w[31] * N8 + m_w[32] * N9 + m_w[8];

	error = expect - output;

	 m_w[29] = m_w[29] + m_learning_rate * error * N6;
	 m_w[30] = m_w[30] + m_learning_rate * error * N7;
	 m_w[31] = m_w[31] + m_learning_rate * error * N8;
	 m_w[32] = m_w[32] + m_learning_rate * error * N9;

	 return;

}

void Agent::learn_and_exploit(const std::vector<std::vector<double>> &cues, const std::vector<std::vector<double>> &quality) {
	
	assert(cues.size() == quality.size());

	int sample_size = static_cast<int> (cues[0].size());  // number of options to choose from during foraging
	std::vector<double> assessment(sample_size);
	int best_env = -100.0;   //which cue leads to highest estimation
	
	assert(m_learning_episodes >= 0);
	assert(m_learning_rate >= 0.0);

	if (m_learning_episodes > 0 && m_learning_rate > 0.0) {   // if there is learning at all
		
		for (int le = 0; le < m_learning_episodes; ++le) {
			
			this->learning(cues[le][0], quality[le][0]);
		}
		
	}

	// choosing the best environment out of the options each time step for the rest of the life
	if (m_learning_episodes < m_lifespan) {

		for (int t = m_learning_episodes; t < m_lifespan; ++t) {
			for (int ss = 0; ss < sample_size; ++ss) {
				assessment[ss] = this->network_calculation(cues[t][ss]);				
			}
			best_env = static_cast<int>(std::distance(assessment.cbegin(), std::max_element(assessment.cbegin(), assessment.cend())));
			m_fitness += quality[t][best_env];
			m_best_fitness += *std::max_element(quality[t].begin(), quality[t].end());   //what is the quality of the best item available
		}
	}


}

double clamped_reLU(double input) {
	return std::min(std::max(0.0, input), 1.0);
}
