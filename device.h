#ifndef DEVICE_H
#define DEVICE_H

namespace dvc {

	typedef float DeviceEffectivenessArray[6];

	struct IdentificationConditions {
		const char* id;
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

	struct DevicePositionModifiers {
		int right;
		int left;
		int above;
		int under;
		int touching;
		int not_touching;
	};

	struct DeviceIdentification {
		const char* id;
		int maximum;
		int minimum;
	};

	struct DeviceRecipeModel {
		int base_durability;
		int material1_amount;
		int material2_amount;
	};

	struct DeviceIngredient {
		char* name;
		int duration_modifier;
		int durability_modifier;
		int charges_modifier;
		int requirement_modifiers[5];
		char** skills;
		DeviceIdentification* identifications;
		DevicePositionModifiers position_modifiers;
	};

	struct DevicePrompt {
		char* type;
		char** ingredient_whitelist;
		int pool_size;
		int max_reqs[5];
		int material_tiers[2];
		IdentificationConditions* id_conditions;
	};

	struct DeviceRecipe {
		DeviceRecipeModel model;
		int base_duration;
		int base_durability;
		int material1_tier;
		int material2_tier;
		DeviceEffectivenessArray effectiveness_array;
		DeviceIngredient* ingredients[6];
	};

	struct DeviceCraft {
		int minimum_dura;
		int min_requirements[5];
		DeviceIdentification* identifications;
	};


}

#endif // !DEVICE_H
