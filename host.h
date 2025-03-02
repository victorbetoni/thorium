#ifndef HOST_H
#define HOST_H

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <map>

using string = std::string;

namespace host {

	struct PositionModifiers {
		int left;
		int right;
		int under;
		int above;
		int touching;
		int not_touching;
		NLOHMANN_DEFINE_TYPE_INTRUSIVE(PositionModifiers, left, right, under, above, touching, not_touching)
	};

	struct HostIdentification {
		std::string id;
		int maximum;
		int minimum;
		NLOHMANN_DEFINE_TYPE_INTRUSIVE(HostIdentification, id, maximum, minimum)
	};

	struct HostIngredient {
		string name;
		int duration_modifier;
		int durability_modifier;
		int charges_modifier;
		std::vector<string> skills;
		std::vector<int> requirement_modifiers;
		std::vector<HostIdentification> identifications;
		PositionModifiers position_modifiers;

		NLOHMANN_DEFINE_TYPE_INTRUSIVE(
			HostIngredient,
			name,
			duration_modifier,
			durability_modifier,
			charges_modifier,
			skills,
			requirement_modifiers,
			identifications,
			position_modifiers
		)
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
		NLOHMANN_DEFINE_TYPE_INTRUSIVE(HostPrompt, type, max_reqs, material_tiers, ingredient_whitelist, stat_weights, minimum_stat_values, discard_conditions, pool_size, minimum_durability, include_all_effectiveness_items)
	};

	struct HostRecipeModel {
		string id;
		int material1_amount;
		int material2_amount;
		std::vector<int> durability_range;
		std::vector<int> hp_range;
	};

	struct Range {
		int minimum;
		int maximum;
	};

	struct BaseRecipeLevel {
		string id;
		Range levelRange;
		Range durabilityRange;
		Range durationRange;
		Range hpRange;
	};

	struct HostBaseRecipe {
		string name;
		string material1;
		string material2;
		int material1Amount;
		int material2Amount;
		Range levelRange;
		std::vector<BaseRecipeLevel> levels;
	};

}


#endif