#include "ViscaProtocol.hpp"
#include "Serial.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace harrier {
/*
These constants can be found in the docs of Active Silicon
*/
constexpr uint8_t VISCA_HEADER_CAMERA = 0x81;             // Camera address 1
constexpr uint8_t VISCA_HEADER_INTERFACE = 0x82;          // Interface board address 2
constexpr uint8_t VISCA_HEADER_CAMERA_INQUIRY = 0x89;     // Inquiry for camera
constexpr uint8_t VISCA_HEADER_INTERFACE_INQUIRY = 0x8A;  // Inquiry for interface board
constexpr uint8_t VISCA_TERMINATOR = 0xFF;                // Command terminator
constexpr uint8_t VISCA_ACK_SOCKET_MASK = 0x30;           // Mask for socket number in ACK

std::string formatHex(const std::vector<uint8_t>& bytes) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (uint8_t byte : bytes) {
        ss << std::setw(2) << static_cast<int>(byte) << " ";
    }
    return ss.str();
}

struct CommandInfo {
    std::chrono::steady_clock::time_point timestamp;
    ViscaCommandType type;
    std::chrono::milliseconds timeout;
    std::promise<ViscaResponse> promise;
    ViscaProtocol::ResponseCallback callback;
    std::vector<uint8_t> command;
    uint8_t socketNumber = 0;  // Set when ACK is received

    CommandInfo(ViscaCommandType type, std::chrono::milliseconds timeout,
                const std::vector<uint8_t>& cmd)
        : timestamp(std::chrono::steady_clock::now()), type(type), timeout(timeout), command(cmd) {}
};

struct ViscaProtocol::Impl {
    std::shared_ptr<Serial> serialPort;
    std::mutex mutex;
    std::condition_variable cv;
    std::thread readThread;
    std::atomic<bool> running{false};
    std::chrono::milliseconds vSyncPeriod{33};  // Default to 30fps (33.3ms)

    std::queue<std::shared_ptr<CommandInfo>> pendingCommands;
    std::map<uint8_t, std::shared_ptr<CommandInfo>> activeCommands;  // Socket number -> CommandInfo

    void processResponse(const std::vector<uint8_t>& response) {
        if (response.empty() || response.back() != VISCA_TERMINATOR) {
            // Incomplete response, just ignore it
            return;
        }

        ViscaResponse viscaResponse;

        if (response.size() >= 3 && response[0] >= 0x90 && response[0] <= 0x9F) {
            if (response[1] == 0x4F || response[1] == 0x5F) {
                // Data from inquiry
                viscaResponse.type = ViscaResponse::Type::Data;
                viscaResponse.data = response;
            } else if (response[1] == 0x41) {
                // ACK
                viscaResponse.type = ViscaResponse::Type::Acknowledge;
                viscaResponse.socketNumber = (response[1] & VISCA_ACK_SOCKET_MASK) >> 4;
                handleAck(viscaResponse);
            } else if (response[1] == 0x51) {
                // Completion
                viscaResponse.type = ViscaResponse::Type::Completion;
                handleCompletion(viscaResponse);
            } else if (response[1] == 0x60) {
                // Error
                viscaResponse.type = ViscaResponse::Type::Error;
                viscaResponse.errorCode = (response.size() >= 4) ? response[2] : 0;
                handleError(viscaResponse);
            }
        }
    }

    void handleAck(const ViscaResponse& response) {
        std::lock_guard<std::mutex> lock(mutex);

        if (pendingCommands.empty()) {
            // No pending commands, ignore
            return;
        }

        auto command = pendingCommands.front();
        pendingCommands.pop();

        // Set socket number
        command->socketNumber = response.socketNumber;

        // Store in active commands
        activeCommands[response.socketNumber] = command;

        if (command->callback) {
            command->callback(response);
        }
    }

    void handleCompletion(const ViscaResponse& response) {
        std::lock_guard<std::mutex> lock(mutex);

        for (auto it = activeCommands.begin(); it != activeCommands.end(); ++it) {
            auto& command = it->second;

            if (command->promise.valid()) {
                command->promise.set_value(response);
            }

            if (command->callback) {
                command->callback(response);
            }

            activeCommands.erase(it);
            break;
        }
    }

    void handleError(const ViscaResponse& response) {
        std::lock_guard<std::mutex> lock(mutex);

        for (auto it = activeCommands.begin(); it != activeCommands.end(); ++it) {
            auto& command = it->second;

            if (command->promise.valid()) {
                command->promise.set_value(response);
            }

            if (command->callback) {
                command->callback(response);
            }

            activeCommands.erase(it);
            break;
        }
    }

    void processInquiryResponse(const std::vector<uint8_t>& response) {
        std::lock_guard<std::mutex> lock(mutex);

        if (activeCommands.empty()) {
            return;
        }

        for (auto it = activeCommands.begin(); it != activeCommands.end(); ++it) {
            auto& command = it->second;

            if (command->type == ViscaCommandType::Inquiry) {
                ViscaResponse viscaResponse;
                viscaResponse.type = ViscaResponse::Type::Data;
                viscaResponse.data = response;

                if (command->promise.valid()) {
                    command->promise.set_value(viscaResponse);
                }

                if (command->callback) {
                    command->callback(viscaResponse);
                }

                activeCommands.erase(it);
                break;
            }
        }
    }

    void checkTimeouts() {
        std::lock_guard<std::mutex> lock(mutex);

        auto now = std::chrono::steady_clock::now();

        for (auto it = activeCommands.begin(); it != activeCommands.end();) {
            auto& command = it->second;

            if (now - command->timestamp > command->timeout) {
                // Timestamp is bigger than the set timeout threshold, so the command timed out most
                // likely in this case

                ViscaResponse response;
                response.type = ViscaResponse::Type::Error;
                response.errorCode = 0xFE;

                if (command->promise.valid()) {
                    command->promise.set_value(response);
                }

                if (command->callback) {
                    command->callback(response);
                }

                it = activeCommands.erase(it);
            } else {
                ++it;
            }
        }
    }

    void readThreadFunc() {
        std::vector<uint8_t> buffer;
        buffer.reserve(32);

        while (running) {
            try {
                auto data = serialPort->read(32);

                if (!data.empty()) {
                    buffer.insert(buffer.end(), data.begin(), data.end());

                    size_t pos = 0;
                    while (pos < buffer.size()) {
                        auto terminatorPos =
                            std::find(buffer.begin() + pos, buffer.end(), VISCA_TERMINATOR);

                        if (terminatorPos != buffer.end()) {
                            size_t responseSize = terminatorPos - buffer.begin() - pos + 1;
                            std::vector<uint8_t> response(buffer.begin() + pos,
                                                          buffer.begin() + pos + responseSize);
                            processResponse(response);

                            pos += responseSize;
                        } else {
                            break;
                        }
                    }

                    if (pos > 0) {
                        buffer.erase(buffer.begin(), buffer.begin() + pos);
                    }
                }

                checkTimeouts();

                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            } catch (const SerialPortError& e) {
                std::cerr << "Serial port error in VISCA read thread: " << e.what() << std::endl;

                // Short sleep before retrying
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }

    void startReadThread() {
        running = true;
        readThread = std::thread([this]() { readThreadFunc(); });
    }

    void stopReadThread() {
        running = false;
        if (readThread.joinable()) {
            readThread.join();
        }
    }
};

ViscaProtocol::ViscaProtocol(std::shared_ptr<Serial> serialPort)
    : ptrImpl(std::make_unique<Impl>()) {
    ptrImpl->serialPort = serialPort;
    ptrImpl->startReadThread();
}

ViscaProtocol::~ViscaProtocol() { ptrImpl->stopReadThread(); }

std::shared_ptr<ViscaProtocol> ViscaProtocol::create(std::shared_ptr<Serial> serialPort) {
    return std::shared_ptr<ViscaProtocol>(new ViscaProtocol(serialPort));
}

}  // namespace harrier