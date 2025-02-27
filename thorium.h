#ifndef RECIPE_H
#define RECIPE_H

#include <vector>
#include <string>
#include <regex>

typedef float EffectivenessArray[6];

__device__ const short AFFECTED_SLOTS[7][7][3] = {
	{ {}, {2}, {}, {3,5}, {2,3}, {4,5,6} }, // index 0
	{ {1},{}, {}, {4,6}, {1,4}, {3,5,6} },  // index 1
	{ {}, {4}, {5}, {1}, {1,4,5}, {2,6} },  // index 2
	{ {3}, {}, {6}, {2}, {2,3,6}, {1,5} },  // index 3
	{ {}, {6}, {}, {3,1}, {3,6}, {} },      // index 4
	{ {5}, {}, {}, {2,4}, {4,5}, {1,2,3} }  // index 5
};

char** to_device_vector(std::vector<std::string> vec) {
	char** charArray = new char* [vec.size()];
	for (size_t i = 0; i < vec.size(); ++i) {
		charArray[i] = to_device_string(vec[i]);
	}
	return charArray;
}

char* to_device_string(std::string str) {
	char* dev_str = new char[str.size()];
	std::strcpy(dev_str, str.c_str());
	return dev_str;
}

std::vector<std::string> split(const std::string& str, const std::string& delimiter) {
	std::vector<std::string> result;
	std::regex rgx(delimiter);
	std::sregex_token_iterator iter(str.begin(), str.end(), rgx, -1);
	std::sregex_token_iterator end;

	for (; iter != end; ++iter) {
		result.push_back(*iter);
	}

	return result;
}

#endif