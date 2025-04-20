#include "V4L2Device.hpp"

#include <cassert>
#include <iomanip>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <stdexcept>
#include <iostream>




V4L2Device::V4L2Device(const std::string& path) : path(path) {}

boost::optional<HarrierDeviceState> V4L2Device::getDeviceHandle() const {
    int fd = open(path.c_str(), O_RDWR);
    if(fd < 0) {
        throw std::runtime_error("Invalid UVC device handle.");
    } 

    return HarrierDeviceState(fd, true);
}

HarrierDeviceState::HarrierDeviceState(const int deviceDescriptor, bool verboseState) : deviceHandle(deviceDescriptor), verboseQueries(verboseState) {
    formats = {};
    caps = {};
    formats.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
}

HarrierDeviceState::~HarrierDeviceState() {
    if (deviceHandle >= 0) close(deviceHandle);
}

void HarrierDeviceState::queryCapabilities() {
    if(ioctl(deviceHandle, VIDIOC_QUERYCAP, &caps) < 0) throw std::runtime_error("Failed to query capabilities.");

    if(verboseQueries) {
        std::cout << "Driver: " << caps.driver << "\n"
                  << "Card: " << caps.card << "\n"
                  << "Bus Info: " << caps.bus_info << "\n"
                  << "Version: " << ((caps.version >> 16) & 0xFF) << "."
                  << ((caps.version >> 8) & 0xFF) << "."
                  << (caps.version & 0xFF) << "\n"
                  << "Capabilities: " << caps.capabilities << "\n";
    }

    if(not (caps.capabilities & V4L2_CAP_VIDEO_CAPTURE)) throw std::runtime_error("Device does not support video capture.\n");
}

void HarrierDeviceState::queryFormats() {
    do {
        if(verboseQueries) {
            std::cout << "Available formats: \n"
                      << formats.index << " " << formats.description
                      << ": " << std::hex << formats.pixelformat << std::dec << "\n";
        }
    } while(ioctl(deviceHandle, VIDIOC_ENUM_FMT, &formats) == 0);
}
