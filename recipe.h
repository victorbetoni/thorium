#ifndef RECIPE_H
#define RECIPE_H

typedef float EffectivenessArray[6];

#include <string>
#include <vector>
#include "recipe_model.h"
#include "ingredient.h"

struct Recipe {
	int base_duration;
	int base_durability;
	int material1_tier;
	int material2_tier;
	RecipeModel model;
	EffectivenessArray* effectiveness_array;
	std::vector<Ingredient> ingredients;

	bool is_consumable() const {
		return model.id.start("Potion") || model.id.starts_with("Scroll") || model.id.starts_with("Food");
	}

	EffectivenessArray* calc_effectiveness_array() const {
		if (effectiveness_array != NULL) {
			return effectiveness_array;
		}
		EffectivenessArray arr = { 100, 100, 100, 100, 100, 100 };

	}
};

#endif