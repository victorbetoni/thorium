
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "prompt.h"
#include "ingredient.h"
#include "recipe_model.h"

#include <sstream>
#include <fstream>

#include <stdio.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace std;

__device__ Prompt* PROMPT;
__device__ map<string, Ingredient>* INGREDIENTS;
__device__ map<string, RecipeModel>* RECIPE_MODELS;

Prompt* load_prompt(char* file);
map<string, Ingredient>* load_ingredients();
map<string, RecipeModel>* load_recipe_models();

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main(int argc, char *argv[]) {

    if (argc < 2) {
        fprintf(stderr, "Specify prompt file\n");
        exit(1);
    }

    Prompt* host_prompt = load_prompt(argv[1]);
    auto host_ings = load_ingredients(true);
    auto host_models = load_recipe_models();


    Prompt* device_prompt_ptr;
    cudaMalloc((void**)&device_prompt_ptr, sizeof(Prompt));
    cudaMemcpy(device_prompt_ptr, host_prompt, sizeof(Prompt), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(PROMPT, &device_prompt_ptr, sizeof(Prompt*));



    fprintf(stdout, "---fasfsa %d %d", PROMPT->material_tiers[0], PROMPT->material_tiers[1]);

    /*const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    */

    return 0;
}

Prompt* load_prompt(char* file) {
    std::ifstream prompt_file("dummy.json");
    if (!prompt_file) {
        fprintf(stderr, "Couldnt open prompt file. Is it in the same directory as this executable?\n");
        exit(1);
    }
    std::stringstream buffer;
    buffer << prompt_file.rdbuf();
    json j = json::parse(buffer.str());
    Prompt prompt = j.get<Prompt>();

    return &prompt;
}

map<string, Ingredient>* load_ingredients (bool update) {
    
    if (update) {
        if (system("data/sanitize.exe") == 0) {
            fprintf(stderr, "Couldn't run sanitize.exe. Is is in the same directory as this executable?\n");
            exit(1);
        }
    }
    
    std::ifstream sanitized("sanitized.json");

    if (!sanitized) {
        fprintf(stderr, "Couldnt open ingredients file. Is it in the same directory as this executable?\n");
        exit(1);
    }

    std::stringstream buffer;
    buffer << sanitized.rdbuf();
    json j = json::parse(buffer.str());
    map<string, Ingredient> ings = j.get<map<string, Ingredient>>();
    return &ings;
}

map<string, RecipeModel>* load_recipe_models() {
    return NULL;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    //addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
