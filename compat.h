#ifndef COMPAT_H
#define COMPAT_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "host.h"
#include "device.h"

#include <vector>
#include <string>
#include <regex>
#include <map>

namespace compat {

	using DeviceIdentification = dvc::DeviceIdentification;
	using DevicePrompt = dvc::DevicePrompt;
	using DeviceIngredient = dvc::DeviceIngredient;
	using DeviceEffectivenessArray = dvc::DeviceEffectivenessArray;
	using DeviceRecipeModel = dvc::DeviceRecipeModel;
	using DeviceRecipe = dvc::DeviceRecipe;

	using HostIdentification = host::HostIdentification;
	using HostPrompt = host::HostPrompt;
	using HostIngredient = host::HostIngredient;
	using HostRecipeModel = host::HostRecipeModel;


	__device__ const short AFFECTED_SLOTS[7][7][3] = {
		{ {}, {2}, {}, {3,5}, {2,3}, {4,5,6} }, // index 0
		{ {1},{}, {}, {4,6}, {1,4}, {3,5,6} },  // index 1
		{ {}, {4}, {5}, {1}, {1,4,5}, {2,6} },  // index 2
		{ {3}, {}, {6}, {2}, {2,3,6}, {1,5} },  // index 3
		{ {}, {6}, {}, {3,1}, {3,6}, {} },      // index 4
		{ {5}, {}, {}, {2,4}, {4,5}, {1,2,3} }  // index 5
	};

	__host__ char* to_device_string(const std::string& str);

	__host__  char** to_device_vector(std::vector<std::string> vec);

	__host__  std::vector<std::string> split(const std::string& str, const std::string& delimiter);

	__device__ int compare_str(const char* str1, const char* str2);

	__device__ void sum_effectiveness_array(DeviceEffectivenessArray& arr1, DeviceEffectivenessArray& arr2);

	__device__ DeviceEffectivenessArray* get_effectiveness_array(DeviceIngredient& ing, int index);

	__device__ DeviceEffectivenessArray* get_recipe_effectiveness_array(DeviceRecipe& recipe);

	__host__ DeviceIdentification* to_device_identification(HostIdentification& id);

	__host__ DevicePrompt* to_device_prompt(HostPrompt& source);

	__host__ DeviceRecipeModel* to_device_recipe_model(HostRecipeModel& hM);

}

#endif // !COMPAT_H
