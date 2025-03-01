
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

__device__ DevicePrompt* PROMPT;
__device__ DeviceRecipeModel* SELECT_MODEL;
__device__ thrust::device_vector<DeviceIngredient>* INGREDIENTS;

HostPrompt* load_prompt(char* file);
map<string, HostIngredient>* load_ingredients(bool reload);
RecipeModel* load_recipe_model(HostPrompt* prompt);

int main(int argc, char *argv[]) {

    if (argc < 2) {
        fprintf(stderr, "Specify prompt file\n");
        exit(1);
    }

    HostPrompt* prompt = load_prompt("");
    map<string, HostIngredient>* ings = load_ingredients(true);
    fprintf(stdout, "%s", prompt->discard_conditions["spellDamage"].c_str());


    /*const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    

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

HostPrompt* load_prompt(char* file) {
    std::ifstream prompt_file("dummy.json");
    if (!prompt_file) {
        fprintf(stderr, "Couldnt open prompt file. Is it in the same directory as this executable?\n");
        exit(1);
    }
    std::stringstream buffer;
    buffer << prompt_file.rdbuf();
    json j = json::parse(buffer.str());
    HostPrompt* prompt = new HostPrompt();
    *prompt = j.get<HostPrompt>();
    return prompt;
}

map<string, HostIngredient>* load_ingredients (bool update) {
    map<string, HostIngredient>* ingredients = new map<string, HostIngredient>{};

    if (update) {
        cout << "Loading ingredients table...\n";
        if (system("sanitize.exe") != 0) {
            cerr << "Couldn't run sanitize.exe. Is is in the same directory as this executable?\n";
            exit(1);
        }
    }
    
    std::ifstream sanitized("sanitized.json");

    if (!sanitized) {
        cerr << "Couldnt open ingredients file. Is it inside data directory?\n";
        exit(1);
    }

    std::stringstream buffer;
    buffer << sanitized.rdbuf();
    json j = json::parse(buffer.str());

    for (auto& element : j.items()) {
        json& v = element.value();
        const string key = element.key();
        HostIngredient hIng = v.get<HostIngredient>();
        ingredients->insert(std::pair<string, HostIngredient>(key, hIng));
    }
    return ingredients;
}

RecipeModel* load_recipe_model(HostPrompt* prompt) {

    auto splited = split(prompt->type, "-");
    std::transform(splited[0].begin(), splited[0].end(), splited[0].begin(),
        [](unsigned char c) { return std::tolower(c); });
    fprintf(stdout, splited[0].c_str());
    return NULL;
}

