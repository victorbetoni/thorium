#ifndef RECIPE_H
#define RECIPE_H

#include <vector>
#include <string>
#include <regex>
#include <map>

typedef float EffectivenessArray[6];

const short AFFECTED_SLOTS[7][7][3] = {
	{ {}, {2}, {}, {3,5}, {2,3}, {4,5,6} }, // index 0
	{ {1},{}, {}, {4,6}, {1,4}, {3,5,6} },  // index 1
	{ {}, {4}, {5}, {1}, {1,4,5}, {2,6} },  // index 2
	{ {3}, {}, {6}, {2}, {2,3,6}, {1,5} },  // index 3
	{ {}, {6}, {}, {3,1}, {3,6}, {} },      // index 4
	{ {5}, {}, {}, {2,4}, {4,5}, {1,2,3} }  // index 5
};

__host__ char* to_device_string(const std::string& str) {
	char* deviceStr = new char[str.size() + 1];
	std::copy(str.begin(), str.end(), deviceStr);
	deviceStr[str.size()] = '\0';
	return deviceStr;
}

__host__  char** to_device_vector(std::vector<std::string> vec) {
	char** charArray = new char* [vec.size()];
	for (size_t i = 0; i < vec.size(); ++i) {
		charArray[i] = to_device_string(vec[i]);
	}
	return charArray;
}

__host__  std::vector<std::string> split(const std::string& str, const std::string& delimiter) {
	std::vector<std::string> result;
	std::regex rgx(delimiter);
	std::sregex_token_iterator iter(str.begin(), str.end(), rgx, -1);
	std::sregex_token_iterator end;

	for (; iter != end; ++iter) {
		result.push_back(*iter);
	}

	return result;
}

__device__ void sum_effectiveness_array(EffectivenessArray* arr1, EffectivenessArray* arr2) {
	*arr1[0] += *arr2[0];
	*arr1[1] += *arr2[1];
	*arr1[2] += *arr2[2];
	*arr1[3] += *arr2[3];
	*arr1[4] += *arr2[4];
	*arr1[5] += *arr2[5];
}

#endif