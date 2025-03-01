#ifndef PROMPT_H
#define PROMPT_H

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <map>
#include "thorium.h"

using namespace std;

struct IdentificationConditions {
	float weight;
	int minimum_value = -1;
	int discard_condition_op = -1;
	int discard_condition_value = -1;
	// 0: >
	// 1: >=
	// 2: <
	// 3: <=
	// 4: ==
	// 5: !=
};

struct DevicePrompt {
	char* type;
	char** ingredient_whitelist;
	int pool_size;
	int max_reqs[5];
	int material_tiers[2];
	std::pair<char*, IdentificationConditions>* id_conditions;
};

struct HostPrompt {
	string type;
	std::vector<int> max_reqs;
	std::vector<int> material_tiers;
	std::vector<string> ingredient_whitelist;
	std::map<string, float> stat_weights;
	std::map<string, int> minimum_stat_values;
	std::map<string, string> discard_conditions;
	int pool_size;
	int minimum_durability;
	bool include_all_effectiveness_items;


	NLOHMANN_DEFINE_TYPE_INTRUSIVE(
		HostPrompt,
		type,
		max_reqs,
		material_tiers,
		ingredient_whitelist,
		stat_weights,
		minimum_stat_values,
		discard_conditions,
		pool_size,
		minimum_durability,
		include_all_effectiveness_items
	)
};

DevicePrompt* to_device_prompt(HostPrompt* source) {

	DevicePrompt* prompt;

	prompt->type = to_device_string(source->type);
	prompt->ingredient_whitelist = to_device_vector(source->ingredient_whitelist);

	prompt->pool_size = source->pool_size;
	prompt->max_reqs[0] = source->max_reqs[0];
	prompt->max_reqs[1] = source->max_reqs[1];
	prompt->max_reqs[2] = source->max_reqs[2];
	prompt->max_reqs[3] = source->max_reqs[3];
	prompt->max_reqs[4] = source->max_reqs[4];
	prompt->material_tiers[0] = source->material_tiers[0];
	prompt->material_tiers[1] = source->material_tiers[1];

	prompt->id_conditions = new pair<char*, IdentificationConditions>[source->stat_weights.size()];

	int i = 0;
	for (const auto& pair : source->stat_weights) {
		IdentificationConditions cond = {} ;
		cond.weight = pair.second;
		if (source->minimum_stat_values.find(pair.first) != source->minimum_stat_values.end()) {
			cond.minimum_value = source->minimum_stat_values[pair.first];
		}
		if (source->discard_conditions.find(pair.first) != source->discard_conditions.end()) {
			auto c = source->discard_conditions[pair.first];
			vector<string> splited = split(c, " ");
			string operat = splited[0];
			string operand = splited[1];
			int opCode = 0;
			if (operat == ">=") opCode = 1;
			if (operat == "<") opCode = 2;
			if (operat == "<=") opCode = 3;
			if (operat == "==") opCode = 4;
			if (operat == "!=") opCode = 5;
			int opV = stoi(operand);
			cond.discard_condition_op = opCode;
			cond.discard_condition_value = opV;
		}
		prompt->id_conditions[i] = std::pair<char*, IdentificationConditions>{ to_device_string(pair.first), cond };
	}

	return prompt;
	
}


#endif