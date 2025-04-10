#include "HarrierHandler.hpp"
#include "V4L2Device.hpp"
#include "HarrierVisca.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <memory>
#include <type_traits>
#include <linux/videodev2.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>


#define CLEARMEM(x) memset(x, 0, sizeof(x))

void printCommsBuffer(const std::string& name, const uint8_t* buffer, size_t size) {
    std::cout << name << "[" << size << "]: ";
        for (size_t i = 0; i < size; i++) {
            printf(" %02X ", buffer[i]);
        }
        std::cout << "\n";
}


int main(int argc, char** argv) {
    try {
        int deviceHandle = -1;
        HarrierCommsError_t status = HarrierCommsUSBOpenByIndex(&deviceHandle, 0);
        if (status != HarrierCommsOK) {
            std::cerr << "Failed to retrieve Harrier's USB handle, error code: " << status << std::endl;
            return 1;
        }

        Harrier harrier36x(deviceHandle);
        harrier36x.sendCommand(harrier36x.createViscaCommandPacket(HARRIER_POWER_CYCLE));

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
