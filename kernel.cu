﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/device_vector.h>
#include "compat.h"

#include <sstream>
#include <fstream>

#include <stdio.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace std;

using DeviceIdentification = dvc::DeviceIdentification;
using DevicePrompt = dvc::DevicePrompt;
using DeviceIngredient = dvc::DeviceIngredient;
using DeviceEffectivenessArray = dvc::DeviceEffectivenessArray;
using DeviceRecipeModel = dvc::DeviceRecipeModel;

using HostIdentification = host::HostIdentification;
using HostPrompt = host::HostPrompt;
using HostIngredient = host::HostIngredient;
using HostRecipeModel = host::HostRecipeModel;

__device__ DevicePrompt* PROMPT;
__device__ DeviceRecipeModel* SELECTED_MODEL;
__device__ thrust::device_vector<DeviceIngredient>* INGREDIENTS;

HostPrompt* load_prompt(char* file);
map<string, HostIngredient>* load_ingredients(bool reload);
HostRecipeModel* load_recipe_model(HostPrompt* prompt);

__global__ void combineCrafts(int* A, int* B, int* C, int N) {
    
}

int main(int argc, char *argv[]) {

    if (argc < 2) {
        fprintf(stderr, "Specify prompt file\n");
        exit(1);
    }

    HostPrompt* prompt = load_prompt("");
    map<string, HostIngredient>* ings = load_ingredients(true);
    HostRecipeModel* model = load_recipe_model(prompt);

    DevicePrompt* dPrompt = compat::to_device_prompt(*prompt);
    DeviceRecipeModel* dRecipeModel = compat::to_device_recipe_model(*model);

    cudaMemcpyToSymbol(PROMPT, dPrompt, sizeof(dPrompt), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(SELECTED_MODEL, dRecipeModel, sizeof(dRecipeModel), 0, cudaMemcpyHostToDevice);



    cudaDeviceSynchronize();
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
        if (system("sanitize.exe dummy.json") != 0) {
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

HostRecipeModel* load_recipe_model(HostPrompt* prompt) {

    auto splited = compat::split(prompt->type, "-");
    std::transform(splited[0].begin(), splited[0].end(), splited[0].begin(),
        [](unsigned char c) { return std::tolower(c); });
    auto part = splited[0];

    std::ifstream models("data/base_recipes/" + part + ".json");
    if (!models) {
        cerr << "Couldnt open ingredients file. Is it inside /data/base_recipes directory?\n";
        exit(1);
    }

    std::stringstream buffer;
    buffer << models.rdbuf();
    json j = json::parse(buffer.str());
    for (auto& element : j) {
        auto levels = element["levels"];
        for (auto& lvl : levels) {
            auto id = lvl["id"];
            if (id.dump().compare(splited[1] + "-" + splited[2])) {
                HostRecipeModel* m = new HostRecipeModel();
                m->id = id.dump();
                m->material1_amount = element["material1Amount"].get<int>();
                m->material1_amount = element["material2Amount"].get<int>();
                if (lvl.contains("durationRange")) {
                    m->hp_range.push_back(lvl["hprRange"]["minimum"].get<int>());
                    m->hp_range.push_back(lvl["hprRange"]["maximum"].get<int>());
                    m->durability_range.push_back(lvl["durationRange"]["minimum"].get<int>());
                    m->durability_range.push_back(lvl["durationRange"]["maximum"].get<int>());
                } else {
                    m->hp_range.push_back(lvl["hpRange"]["minimum"].get<int>());
                    m->hp_range.push_back(lvl["hpRange"]["maximum"].get<int>());
                    m->durability_range.push_back(lvl["durabilityRange"]["minimum"].get<int>());
                    m->durability_range.push_back(lvl["durabilityRange"]["maximum"].get<int>());
                }
                return m;
            }
        }
    }


    return NULL;
}

