#ifndef INGREDIENT_H
#define INGREDIENT_H

#include <string>
#include <vector>
#include <map>
#include "identification.h"
#include "thorium.h"

using string = std::string;

struct PositionModifiers {
	int left;
	int right;
	int under;
	int above;
	int touching;
	int not_touching;
};

struct DeviceIngredient {
	char* name;
	int duration_modifier;
	int durability_modifier;
	int charges_modifier;
	int* requirement_modifiers;
	char** skills;
	DeviceIdentification* identifications;
	PositionModifiers* position_modifiers;
};


struct HostIngredient {
	string name;
	int duration_modifier;
	int durability_modifier;
	int charges_modifier;
	std::vector<string> skills;
	std::vector<int> requirement_modifiers;
	std::vector<HostIdentification*> identifications;
	PositionModifiers* position_modifiers;
};

DeviceIngredient* to_device_ingredient(HostIngredient* ing) {
	DeviceIngredient* dev_ing;
	dev_ing->name = to_device_string(ing->name);
	dev_ing->skills = to_device_vector(ing->skills);
	dev_ing->charges_modifier = ing->charges_modifier;
	dev_ing->durability_modifier = ing->durability_modifier;
	dev_ing->duration_modifier = ing->duration_modifier;
	dev_ing->position_modifiers = ing->position_modifiers;
	dev_ing->requirement_modifiers = new int[5]{ 
		ing->requirement_modifiers[0], 
		ing->requirement_modifiers[1],
		ing->requirement_modifiers[2],
		ing->requirement_modifiers[3],
		ing->requirement_modifiers[4],
	};
	dev_ing->identifications = new DeviceIdentification[ing->identifications.size()];
	for (int i = 0; i < ing->identifications.size(); i++) {
		dev_ing->identifications[i] = *to_device_identification(ing->identifications[i]);
	}
	return dev_ing;
}

EffectivenessArray* get_effectiveness_array(DeviceIngredient* ing, int index) {
	EffectivenessArray array = { 100, 100, 100, 100, 100, 100 };
	for (int i = 0; i < 3; i++) {
		if (AFFECTED_SLOTS[index][0][i] != 0) {
			array[AFFECTED_SLOTS[index][0][i] - 1] += ing->position_modifiers->left;
		}
	}
	for (int i = 0; i < 3; i++) {
		if (AFFECTED_SLOTS[index][1][i] != 0) {
			array[AFFECTED_SLOTS[index][1][i] - 1] += ing->position_modifiers->right;
		}
	}
	for (int i = 0; i < 3; i++) {
		if (AFFECTED_SLOTS[index][2][i] != 0) {
			array[AFFECTED_SLOTS[index][2][i] - 1] += ing->position_modifiers->under;
		}
	}
	for (int i = 0; i < 3; i++) {
		if (AFFECTED_SLOTS[index][3][i] != 0) {
			array[AFFECTED_SLOTS[index][3][i] - 1] += ing->position_modifiers->above;
		}
	}
	for (int i = 0; i < 3; i++) {
		if (AFFECTED_SLOTS[index][4][i] != 0) {
			array[AFFECTED_SLOTS[index][4][i] - 1] += ing->position_modifiers->touching;
		}
	}
	for (int i = 0; i < 3; i++) {
		if (AFFECTED_SLOTS[index][6][i] != 0) {
			array[AFFECTED_SLOTS[index][6][i] - 1] += ing->position_modifiers->not_touching;
		}
	}
	return &array;
}

#endif // !INGREDIENT_H