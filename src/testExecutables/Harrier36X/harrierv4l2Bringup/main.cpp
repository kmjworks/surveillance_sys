#include "HarrierCommsUSB.h"
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


class BufferPrinter {
public:
    static void print(const std::string& name, const uint8_t* buffer, size_t size) {
        std::cout << name << "[" << size << "]: ";
        for (size_t i = 0; i < size; i++) {
            printf(" %02X ", buffer[i]);
        }
        std::cout << std::endl;
    }
};


class HarrierException : public std::runtime_error {
public:
    explicit HarrierException(const std::string& message, HarrierCommsError_t status) 
        : std::runtime_error(message + ": " + std::to_string(status)), status_(status) {}
    
    HarrierCommsError_t getStatus() const { return status_; }
    
private:
    HarrierCommsError_t status_;
};


class V4L2Camera {
public:
    explicit V4L2Camera(const std::string& device = "/dev/video0") 
        : device_path_(device), fd_(-1) {
        openDevice();
    }

    ~V4L2Camera() {
        if (fd_ >= 0) {
            close(fd_);
        }
    }
    
    bool isOpen() const { return fd_ >= 0; }
    

    void queryCameraCapabilities() {
        struct v4l2_capability caps = {};
        
        if (ioctl(fd_, VIDIOC_QUERYCAP, &caps) < 0) {
            throw std::runtime_error("Failed to query camera capabilities");
        }
        
        std::cout << "Driver: " << caps.driver << "\n";
        std::cout << "Card: " << caps.card << "\n";
        std::cout << "Bus Info: " << caps.bus_info << "\n";
        std::cout << "Version: " << ((caps.version >> 16) & 0xFF) << "."
                  << ((caps.version >> 8) & 0xFF) << "."
                  << (caps.version & 0xFF) << "\n";
        std::cout << "Capabilities: 0x" << std::hex << caps.capabilities << std::dec << "\n";
        
        if (!(caps.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
            throw std::runtime_error("Device does not support video capture");
        }
    }
    
    void queryFormats() {
        struct v4l2_fmtdesc fmt = {};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        
        std::cout << "Available formats:" << std::endl;
        while (ioctl(fd_, VIDIOC_ENUM_FMT, &fmt) == 0) {
            std::cout << "  " << fmt.index << ": " << fmt.description 
                      << " (0x" << std::hex << fmt.pixelformat << std::dec << ")" << std::endl;
            fmt.index++;
        }
    }

private:
    void openDevice() {
        fd_ = open(device_path_.c_str(), O_RDWR);
        if (fd_ < 0) {
            throw std::runtime_error("Failed to open camera device: " + device_path_);
        }
    }

    std::string device_path_;
    int fd_;
};


template<typename T = uint8_t*>
class CommandHandler {
public:
    struct CommandType {
        enum class Category {
            INQUIRY,
            CONTROL,
            SETUP,
            UNKNOWN
        };
        
        Category category;
        std::string name;
        std::string description;
    };

    explicit CommandHandler(HarrierUSBHandle handle) : handle_(handle) {}
    
    template<size_t N>
    void executeCommand(const uint8_t (&command)[N], const std::string& cmdName) {
        executeCommandImpl(command, N, getCommandType(command, N), cmdName);
    }
    
    void executeCommand(const uint8_t* command, size_t size, const std::string& cmdName) {
        executeCommandImpl(command, size, getCommandType(command, size), cmdName);
    }
    
    size_t getCommandCount() const { return command_count_; }

private:
    
    void executeCommandImpl(const void* command, size_t commandSize, 
                           const CommandType& cmdType, const std::string& cmdName) {
        uint8_t reply[16];
        unsigned char bytes = 0;
        
        CLEARMEM(reply);
        bytes = 6;
        
        std::cout << "Command :  " << cmdType.name << " " << cmdName 
                  << " (size: " << commandSize << ")" << std::endl;
        
        HarrierCommsError_t status;
        status = HarrierCommsUSBTransmit(handle_, command, commandSize);
        if (status != HarrierCommsOK) {
            throwOnInternalCmdError("Error transmitting command", status);
        }
        
        status = HarrierCommsUSBReceive(handle_, reply, sizeof(reply), &bytes, 200);
        if (status != HarrierCommsOK) {
            throwOnInternalCmdError("Error receiving response", status);
        }
        
        BufferPrinter::print(cmdName, reply, bytes);
        command_count_++;
    }

    void throwOnInternalCmdError(const std::string& failedCommand, HarrierCommsError_t status) {
        throw HarrierException(failedCommand, status);
    }


    CommandType getCommandType(const uint8_t* command, size_t size) {
        CommandType result{CommandType::Category::UNKNOWN, "Unknown", "Unknown command type"};
        
        if (size < 3) {
            return result;
        }
        
        if (command[2] == 0x09) {
            result.category = CommandType::Category::INQUIRY;
            result.name = "INQUIRY";
            result.description = "Query camera information";
        } else if (command[1] == 0x01) {
            result.category = CommandType::Category::CONTROL;
            result.name = "CONTROL";
            result.description = "Camera control command";
        } else if (size >= 4 && (command[2] == 0x0A)) {
            result.category = CommandType::Category::SETUP;
            result.name = "SETUP";
            result.description = "Camera setup/configuration";
        }
        
        return result;
    }

    HarrierUSBHandle handle_;
    size_t command_count_ = 0;
};


class HarrierUSB {
public:
    HarrierUSB() : handle_(-1) {
        openDevice();
    }
    
    ~HarrierUSB() {
        close();
    }
    
    void openDevice(int index = 0) {
        HarrierCommsError_t status = HarrierCommsUSBOpenByIndex(&handle_, index);
        if (status != HarrierCommsOK) {
            throw HarrierException("Failed to open Harrier USB device", status);
        }
        std::cout << "[+] Harrier UVC device opened \n";
        
        uint8_t clearBuffer[128];
        unsigned char bytes = sizeof(clearBuffer);
        CLEARMEM(clearBuffer);
        HarrierCommsUSBReceive(handle_, clearBuffer, sizeof(clearBuffer), &bytes, 100);
    }
    
    void close() {
        if (handle_ >= 0) {
            HarrierCommsError_t status = HarrierCommsUSBClose(handle_);
            if (status != HarrierCommsOK) {
                std::cerr << "Warning: Failed to close Harrier USB device: " << status << "\n";
            }
            handle_ = -1;
        }
    }
    
    HarrierUSBHandle getHandle() const {
        if (handle_ < 0) {
            throw std::runtime_error("Invalid Harrier USB handle");
        }
        return handle_;
    }
    
private:
    HarrierUSBHandle handle_;
};

int main(int argc, char **argv) {
    try {
        V4L2Camera camera;
        std::cout << "V4L2 camera opened successfully" << "\n";
        
        camera.queryCameraCapabilities();
        camera.queryFormats();
        
        HarrierUSB harrier;
        
        CommandHandler<> cmdHandler(harrier.getHandle());
        
        cmdHandler.executeCommand(HARRIER_INTERFACE_FW_VER, "HARRIER FW VERSION");
        cmdHandler.executeCommand(HARRIER_INTERFACE_HW_VER, "HARRIER HW VERSION");
        cmdHandler.executeCommand(HARRIER_INTERFACE_HW_STATUS, "HARRIER HW STATUS");
        
        cmdHandler.executeCommand(HARRIER_AUTO_FOCUS, "HARRIER AUTO FOCUS ON");

        std::cout << "Commands executed : " << cmdHandler.getCommandCount() << "\n";
        
        return 0;
    } catch (const HarrierException& e) {
        std::cerr << "Harrier Error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
