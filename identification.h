#ifndef IDENTIFICATION_H
#define IDENTIFICATION_H

#include "thorium.h"
#include <string>

struct DeviceIdentification {
	const char* id;
	int maximum;
	int minimum;
};

struct HostIdentification {
	std::string id;
	int maximum;
	int minimum;

	NLOHMANN_DEFINE_TYPE_INTRUSIVE(
		HostIdentification,
		maximum,
		minimum
	)

};

DeviceIdentification* to_device_identification(HostIdentification* id) {
	DeviceIdentification* dev_id;
	dev_id->id = id->id.c_str();
	dev_id->maximum = id->maximum;
	dev_id->minimum = id->minimum;
	return dev_id;
}

#endif // !IDENTIFICATION_H