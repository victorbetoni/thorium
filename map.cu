#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ingredient.h"

__global__ void findIngredient(CudaLikeIngredient* d_keys, int* d_values, int num_elements, int search_key, int* result) {
    int idx = threadIdx.x;
    if (idx < num_elements) {
        if (d_keys[idx] == search_key) {
            *result = d_values[idx];  // Encontrou o valor correspondente à chave
        }
    }
}
