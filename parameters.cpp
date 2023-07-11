#include "parameters.h"

// loading parameters from json file
void from_json(const nlohmann::json& j, parameters& t)
{
	NLOHMANN_JSON_FROM(seed);
	NLOHMANN_JSON_FROM(init_w_range);
	NLOHMANN_JSON_FROM(init_lr_range);
	NLOHMANN_JSON_FROM(init_le_mean);
	NLOHMANN_JSON_FROM(length_fixed_LE);
	NLOHMANN_JSON_FROM(mut_rate_w);
	NLOHMANN_JSON_FROM(mut_step_w);
	NLOHMANN_JSON_FROM(mut_rate_lr);
	NLOHMANN_JSON_FROM(mut_step_lr);
	NLOHMANN_JSON_FROM(mut_rate_le);
	NLOHMANN_JSON_FROM(mut_step_le);
	NLOHMANN_JSON_FROM(N);
	NLOHMANN_JSON_FROM(G);
	NLOHMANN_JSON_FROM(lifespan);
	NLOHMANN_JSON_FROM(nr_replicates);
	NLOHMANN_JSON_FROM(env_sample_size);
	NLOHMANN_JSON_FROM(env_range);
	NLOHMANN_JSON_FROM(init_mean_env);
	NLOHMANN_JSON_FROM(sd_env);
	NLOHMANN_JSON_FROM(env_change);
	NLOHMANN_JSON_FROM(env_change_sd);
	NLOHMANN_JSON_FROM(env_change_type);
	NLOHMANN_JSON_FROM(env_change_rate);
}