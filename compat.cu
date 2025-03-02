#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "device.h"
#include "host.h"
#include "compat.h"

#include <vector>
#include <string>
#include <regex>
#include <map>

namespace compat {
	__host__ char* to_device_string(const std::string& str) {
		char* deviceStr = new char[str.size() + 1];
		std::copy(str.begin(), str.end(), deviceStr);
		deviceStr[str.size()] = '\0';
		return deviceStr;
	}

	__host__  char** to_device_vector(std::vector<std::string> vec) {
		char** charArray = new char* [vec.size()];
		for (size_t i = 0; i < vec.size(); ++i) {
			charArray[i] = to_device_string(vec[i]);
		}
		return charArray;
	}

	__host__  std::vector<std::string> split(const std::string& str, const std::string& delimiter) {
		std::vector<std::string> result;
		std::regex rgx(delimiter);
		std::sregex_token_iterator iter(str.begin(), str.end(), rgx, -1);
		std::sregex_token_iterator end;

		for (; iter != end; ++iter) {
			result.push_back(*iter);
		}

		return result;
	}

	__host__ dvc::DeviceIdentification* to_device_identification(host::HostIdentification& id) {
		dvc::DeviceIdentification* dev_id = new dvc::DeviceIdentification();
		dev_id->id = id.id.c_str();
		dev_id->maximum = id.maximum;
		dev_id->minimum = id.minimum;
		return dev_id;
	}

	__device__ void sum_effectiveness_array(dvc::DeviceEffectivenessArray& arr1, dvc::DeviceEffectivenessArray& arr2) {
		arr1[0] += arr2[0];
		arr1[1] += arr2[1];
		arr1[2] += arr2[2];
		arr1[3] += arr2[3];
		arr1[4] += arr2[4];
		arr1[5] += arr2[5];
	}

	__host__ dvc::DeviceIngredient* to_device_ingredient(host::HostIngredient& ing) {
		dvc::DeviceIngredient* dev_ing = new dvc::DeviceIngredient();
		dev_ing->name = compat::to_device_string(ing.name);
		dev_ing->skills = compat::to_device_vector(ing.skills);
		dev_ing->charges_modifier = ing.charges_modifier;
		dev_ing->durability_modifier = ing.durability_modifier;
		dev_ing->duration_modifier = ing.duration_modifier;

		dev_ing->position_modifiers = dvc::DevicePositionModifiers();
		dev_ing->position_modifiers.left = ing.position_modifiers.left;
		dev_ing->position_modifiers.right = ing.position_modifiers.right;
		dev_ing->position_modifiers.above = ing.position_modifiers.above;
		dev_ing->position_modifiers.under = ing.position_modifiers.under;
		dev_ing->position_modifiers.touching = ing.position_modifiers.touching;
		dev_ing->position_modifiers.not_touching = ing.position_modifiers.not_touching;

		dev_ing->requirement_modifiers[0] = ing.requirement_modifiers[0];
		dev_ing->requirement_modifiers[1] = ing.requirement_modifiers[1];
		dev_ing->requirement_modifiers[2] = ing.requirement_modifiers[2];
		dev_ing->requirement_modifiers[3] = ing.requirement_modifiers[3];
		dev_ing->requirement_modifiers[4] = ing.requirement_modifiers[4];
		dev_ing->identifications = new dvc::DeviceIdentification[ing.identifications.size()];
		for (int i = 0; i < ing.identifications.size(); i++) {
			dev_ing->identifications[i] = *to_device_identification(ing.identifications[i]);
		}
		return dev_ing;
	}


	__device__ dvc::DeviceEffectivenessArray* get_effectiveness_array(dvc::DeviceIngredient& ing, int index) {
		dvc::DeviceEffectivenessArray array = { 100, 100, 100, 100, 100, 100 };
		for (int i = 0; i < 3; i++) {
			if (compat::AFFECTED_SLOTS[index][0][i] != 0) {
				array[compat::AFFECTED_SLOTS[index][0][i] - 1] += ing.position_modifiers.left;
			}
		}
		for (int i = 0; i < 3; i++) {
			if (compat::AFFECTED_SLOTS[index][1][i] != 0) {
				array[compat::AFFECTED_SLOTS[index][1][i] - 1] += ing.position_modifiers.right;
			}
		}
		for (int i = 0; i < 3; i++) {
			if (compat::AFFECTED_SLOTS[index][2][i] != 0) {
				array[compat::AFFECTED_SLOTS[index][2][i] - 1] += ing.position_modifiers.under;
			}
		}
		for (int i = 0; i < 3; i++) {
			if (compat::AFFECTED_SLOTS[index][3][i] != 0) {
				array[compat::AFFECTED_SLOTS[index][3][i] - 1] += ing.position_modifiers.above;
			}
		}
		for (int i = 0; i < 3; i++) {
			if (compat::AFFECTED_SLOTS[index][4][i] != 0) {
				array[compat::AFFECTED_SLOTS[index][4][i] - 1] += ing.position_modifiers.touching;
			}
		}
		for (int i = 0; i < 3; i++) {
			if (compat::AFFECTED_SLOTS[index][6][i] != 0) {
				array[compat::AFFECTED_SLOTS[index][6][i] - 1] += ing.position_modifiers.not_touching;
			}
		}
		return &array;
	}

	__device__ dvc::DeviceEffectivenessArray* get_recipe_effectiveness_array(dvc::DeviceRecipe& recipe) {
		dvc::DeviceEffectivenessArray eff = {100,100,100,100,100,100};
		for (int i = 0; i < 6; i++) {
			auto ing = recipe.ingredients[i];
			sum_effectiveness_array(eff, *get_effectiveness_array(*ing, i));
		}
		return &eff;
	}

	__host__ dvc::DevicePrompt* to_device_prompt(host::HostPrompt& source) {

		dvc::DevicePrompt* prompt = new dvc::DevicePrompt();

		prompt->type = to_device_string(source.type);
		prompt->ingredient_whitelist = to_device_vector(source.ingredient_whitelist);

		prompt->pool_size = source.pool_size;
		prompt->max_reqs[0] = source.max_reqs[0];
		prompt->max_reqs[1] = source.max_reqs[1];
		prompt->max_reqs[2] = source.max_reqs[2];
		prompt->max_reqs[3] = source.max_reqs[3];
		prompt->max_reqs[4] = source.max_reqs[4];
		prompt->material_tiers[0] = source.material_tiers[0];
		prompt->material_tiers[1] = source.material_tiers[1];

		prompt->id_conditions = new dvc::IdentificationConditions[source.stat_weights.size()]();

		int i = 0;
		for (const auto& pair : source.stat_weights) {
			dvc::IdentificationConditions cond = {};
			cond.weight = pair.second;
			if (source.minimum_stat_values.find(pair.first) != source.minimum_stat_values.end()) {
				cond.minimum_value = source.minimum_stat_values[pair.first];
			}
			if (source.discard_conditions.find(pair.first) != source.discard_conditions.end()) {
				auto c = source.discard_conditions[pair.first];
				std::vector<string> splited = split(c, " ");
				string operat = splited[0];
				string operand = splited[1];
				int opCode = 0;
				if (operat == ">=") opCode = 1;
				if (operat == "<") opCode = 2;
				if (operat == "<=") opCode = 3;
				if (operat == "==") opCode = 4;
				if (operat == "!=") opCode = 5;
				int opV = stoi(operand);
				cond.id = pair.first.c_str();
				cond.discard_condition_op = opCode;
				cond.discard_condition_value = opV;
			}
			prompt->id_conditions[i] = cond;
		}

		return prompt;

	}

	__host__ dvc::DeviceRecipeModel* to_device_recipe_model(host::HostRecipeModel& hM) {
		dvc::DeviceRecipeModel* m = new dvc::DeviceRecipeModel();
		m->base_durability = hM.durability_range[1];
		m->material1_amount = hM.material1_amount;
		m->material2_amount = hM.material2_amount;
		return m;
	}

	__device__ int compare_str(const char* str1, const char* str2) {
		int i = 0;
		while (str1[i] != '\0' && str2[i] != '\0') {
			if (str1[i] != str2[i]) {
				return str1[i] - str2[i];
			}
			i++;
		}
		return str1[i] - str2[i];
	}
}