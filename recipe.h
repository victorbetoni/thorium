#ifndef RECIPE_H
#define RECIPE_H

typedef float EffectivenessArray[6];

#include <string>
#include <vector>
#include "recipe_model.h"
#include "ingredient.h"

struct DeviceRecipe {
	RecipeModel model;
	int base_duration;
	int base_durability;
	int material1_tier;
	int material2_tier;
	EffectivenessArray effectiveness_array;
	DeviceIngredient* ingredients[6];

	bool is_consumable() const {
		return model.id.start("Potion") || model.id.starts_with("Scroll") || model.id.starts_with("Food");
	}

	EffectivenessArray* calc_effectiveness_array() const {
		EffectivenessArray arr = { 100, 100, 100, 100, 100, 100 };
	}
};

#endif