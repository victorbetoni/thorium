#include "device.h"
#include "compat.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ const dvc::DeviceCraft* craft(dvc::DeviceRecipe& recipe) {
	
	auto effArr = compat::get_recipe_effectiveness_array(recipe);
	auto matMod = get_mat_mod(recipe.model.material1_amount, recipe.model.material2_amount, recipe.material1_tier, recipe.material2_tier);

	for (int i = 0; i < 6; i++) {
		// do calcs
	}

}

__device__ float get_mat_mod(int mat1_qty, int mat2_qty, int mat1_tier, int mat2_tier) {
	float tierToMult[4] = { 0, 1, 1.25, 1.4 };
	return (tierToMult[mat1_tier] * mat1_qty + tierToMult[mat2_tier] * mat2_qty) / (mat1_qty + mat2_qty);
}