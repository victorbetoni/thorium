#include "device.h"
#include "compat.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ const static float tierToMult[4] = { 0, 1, 1.25, 1.4 };

__device__ const dvc::DeviceCraft* craft(dvc::DeviceRecipe& recipe) {
	
	auto effArr = compat::get_recipe_effectiveness_array(recipe);
	auto matMod = get_mat_mod(recipe.model.material1_amount, recipe.model.material2_amount, recipe.material1_tier, recipe.material2_tier);

	int capacity = 2;
	dvc::DeviceIdentification *ids = new dvc::DeviceIdentification[capacity];

	for (int i = 0; i < 6; i++) {
		auto ing = recipe.ingredients[i];
		size_t size = sizeof(ing->identifications) / sizeof(dvc::DeviceIdentification);
		for (int i = 0; i < size; i++) {

		}
	}

}

__device__ float get_mat_mod(int mat1_qty, int mat2_qty, int mat1_tier, int mat2_tier) {
	return (tierToMult[mat1_tier] * mat1_qty + tierToMult[mat2_tier] * mat2_qty) / (mat1_qty + mat2_qty);
}

__device__ void append_id(dvc::DeviceIdentification* ids, const char* id, int minV, int maxV, int& capacity) {
	size_t size = sizeof(ids) / sizeof(dvc::DeviceIdentification);

	for (int i = 0; i < size; i++) {
		dvc::DeviceIdentification iden = ids[i];
		if (compat::compare_str(iden.id, id) == 0) {
			iden.maximum += maxV;
			iden.minimum += minV;
			ids[i] = iden;
			return;
		}
	}

	if (size == capacity) {
		capacity = capacity * 2;
		ids = (dvc::DeviceIdentification*)realloc(ids, capacity * sizeof(dvc::DeviceIdentification));
	}

	dvc::DeviceIdentification* iden = new dvc::DeviceIdentification();
	iden->id = id;
	iden->maximum = maxV;
	iden->minimum = minV;
	ids[size] = *iden;
}

