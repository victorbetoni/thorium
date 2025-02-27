#ifndef RECIPE_MODEL
#define RECIPE_MODEL

#include <string>
#include <vector>

using string = std::string;

struct RecipeModel {
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

struct BaseRecipe {
	string name;
	string material1;
	string material2;
	int material1Amount;
	int material2Amount;
	Range levelRange;
	std::vector<BaseRecipeLevel> levels;
};

#endif // !RECIPE_MODEL