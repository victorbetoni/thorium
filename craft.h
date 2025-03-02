#ifndef CRAFT_H
#define CRAFT_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device.h"

__device__ const dvc::DeviceCraft* craft(const dvc::DeviceRecipe& recipe);

#endif