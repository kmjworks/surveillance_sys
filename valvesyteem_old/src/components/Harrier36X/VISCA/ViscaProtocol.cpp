// ViscaProtocol.cpp
#include "ViscaProtocol.hpp"
#include "Serial.hpp"

#include <algorithm>
#include <iostream>
#include <queue>
#include <sstream>

// Boost includes
#include <boost/asio.hpp>
#include <boost/bind/bind.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/format.hpp>
#include <boost/thread.hpp>

namespace harrier {

// VISCA protocol constants
constexpr uint8_t VISCA_HEADER_CAMERA = 0x81;             // Camera address 1
constexpr uint8_t VISCA_HEADER_INTERFACE = 0x82;          // Interface board address 2
constexpr uint8_t VISCA_HEADER_CAMERA_INQUIRY = 0x89;     // Inquiry for camera
constexpr uint8_t VISCA_HEADER_INTERFACE_INQUIRY = 0x8A;  // Inquiry for interface board
constexpr uint8_t VISCA_TERMINATOR = 0xFF;                // Command terminator
constexpr uint8_t VISCA_ACK_SOCKET_MASK = 0x30;           // Mask for socket number in ACK

std::string formatHex(const std::vector<uint8_t>& bytes) {
    std::stringstream ss;
    for (uint8_t byte : bytes) {
        ss << boost::format("%02x ") % static_cast<int>(byte);
    }
    return ss.str();
}

struct CommandInfo {
    boost::chrono::steady_clock::time_point timestamp;
    ViscaCommandType type;
    std::chrono::milliseconds timeout;
    boost::promise<ViscaResponse> promise;
    ViscaProtocol::ResponseCallback callback;
    std::vector<uint8_t> command;
    uint8_t socketNumber = 0;  // Set when ACK is received

    CommandInfo(ViscaCommandType type, std::chrono::milliseconds timeout,
                const std::vector<uint8_t>& cmd)
        : timestamp(boost::chrono::steady_clock::now()),
          type(type),
          timeout(timeout),
          command(cmd) {}
};

struct ViscaProtocol::Impl : public boost::enable_shared_from_this<ViscaProtocol::Impl> {
    boost::shared_ptr<Serial> serialPort;
    boost::mutex mutex;
    boost::condition_variable cv;
    boost::asio::io_context io;
    boost::asio::executor_work_guard<boost::asio::io_context::executor_type> work;
    boost::thread_group threadPool;
    std::atomic<bool> running{false};
    std::chrono::milliseconds vSyncPeriod{33};  // Default to 30fps (33.3ms)

    std::queue<boost::shared_ptr<CommandInfo>> pendingCommands;
    boost::mutex pendingCommandsMutex;

    std::map<uint8_t, boost::shared_ptr<CommandInfo>> activeCommands;

    boost::circular_buffer<uint8_t> receiveBuffer{1024};

    boost::asio::steady_timer timeoutTimer;

    boost::thread readThread;

    Impl(boost::shared_ptr<Serial> port)
        : serialPort(port), work(boost::asio::make_work_guard(io)), timeoutTimer(io) {}

    void start() {
        running = true;

        // Start the IO context with multiple worker threads
        const int numThreads =
            std::max(1, static_cast<int>(boost::thread::hardware_concurrency()) - 1);
        for (int i = 0; i < numThreads; ++i) {
            threadPool.create_thread(boost::bind(&boost::asio::io_context::run, &io));
        }

        scheduleTimeoutCheck();

        readThread = boost::thread(&Impl::readThreadFunc, this);
    }

    void stop() {
        running = false;

        // Cancel all pending operations
        timeoutTimer.cancel();

        // Join read thread
        if (readThread.joinable()) {
            readThread.join();
        }

        work.reset();
        io.stop();

        // Wait for all threads to complete
        threadPool.join_all();
    }

    void scheduleTimeoutCheck() {
        timeoutTimer.expires_after(boost::asio::chrono::milliseconds(100));
        timeoutTimer.async_wait(
            boost::bind(&Impl::checkTimeouts, this, boost::asio::placeholders::error));
    }

    void readThreadFunc() {
        while (running) {
            try {
                std::vector<uint8_t> receivedData = serialPort->read(32);

                if (!receivedData.empty()) {
                    processReceivedData(receivedData, receivedData.size());
                }

                // Short sleep to avoid busy looping
                boost::this_thread::sleep_for(boost::chrono::milliseconds(5));
            } catch (const SerialPortError& e) {
                std::cerr << "Serial port error in read thread: " << e.what() << std::endl;
                boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
            }
        }
    }

    void processReceivedData(const std::vector<uint8_t>& buffer, std::size_t bytesRead) {
        boost::lock_guard<boost::mutex> lock(mutex);

        // Add received data to buffer
        for (size_t i = 0; i < bytesRead; ++i) {
            receiveBuffer.push_back(buffer[i]);
        }

        // Process complete responses
        processCompleteResponses();
    }

    void processCompleteResponses() {
        size_t startPos = 0;
        while (startPos < receiveBuffer.size()) {
            // Look for VISCA terminator
            auto it =
                std::find(receiveBuffer.begin() + startPos, receiveBuffer.end(), VISCA_TERMINATOR);

            if (it != receiveBuffer.end()) {
                size_t responseLength = std::distance(receiveBuffer.begin() + startPos, it) + 1;

                // Extract complete response
                std::vector<uint8_t> response;
                response.reserve(responseLength);

                for (size_t i = 0; i < responseLength; ++i) {
                    response.push_back(receiveBuffer[startPos + i]);
                }

                processResponse(response);

                startPos += responseLength;
            } else {
                break;
            }
        }

        if (startPos > 0) {
            receiveBuffer.erase_begin(startPos);
        }
    }

    void processResponse(const std::vector<uint8_t>& response) {
        if (response.empty() || response.back() != VISCA_TERMINATOR) {
            // Incomplete response, ignore
            return;
        }

        ViscaResponse viscaResponse;

        // Determine response type
        if (response.size() >= 3 && response[0] >= 0x90 && response[0] <= 0x9F) {
            if (response[1] == 0x4F || response[1] == 0x5F) {
                // Data from inquiry
                viscaResponse.type = ViscaResponse::Type::Data;
                viscaResponse.data = response;
                processInquiryResponse(response);
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
        boost::lock_guard<boost::mutex> lock(mutex);
        boost::lock_guard<boost::mutex> pendingLock(pendingCommandsMutex);

        if (pendingCommands.empty()) {
            // No pending commands, ignore
            return;
        }

        auto command = pendingCommands.front();
        pendingCommands.pop();

        command->socketNumber = response.socketNumber;

        activeCommands[response.socketNumber] = command;

        if (command->callback) {
            io.post([commandCopy = command, responseCopy = response]() {
                commandCopy->callback(responseCopy);
            });
        }
    }

    void handleCompletion(const ViscaResponse& response) {
        boost::lock_guard<boost::mutex> lock(mutex);

        for (auto it = activeCommands.begin(); it != activeCommands.end(); ++it) {
            auto command = it->second;

            command->promise.set_value(response);

            if (command->callback) {
                io.post([commandCopy = command, responseCopy = response]() {
                    commandCopy->callback(responseCopy);
                });
            }

            activeCommands.erase(it);
            break;
        }
    }

    void handleError(const ViscaResponse& response) {
        boost::lock_guard<boost::mutex> lock(mutex);

        for (auto it = activeCommands.begin(); it != activeCommands.end(); ++it) {
            auto command = it->second;

            command->promise.set_value(response);

            if (command->callback) {
                io.post([commandCopy = command, responseCopy = response]() {
                    commandCopy->callback(responseCopy);
                });
            }

            activeCommands.erase(it);
            break;
        }
    }

    void processInquiryResponse(const std::vector<uint8_t>& response) {
        boost::lock_guard<boost::mutex> lock(mutex);

        if (activeCommands.empty()) {
            return;
        }

        for (auto it = activeCommands.begin(); it != activeCommands.end(); ++it) {
            auto command = it->second;

            if (command->type == ViscaCommandType::Inquiry) {
                ViscaResponse viscaResponse;
                viscaResponse.type = ViscaResponse::Type::Data;
                viscaResponse.data = response;

                command->promise.set_value(viscaResponse);

                if (command->callback) {
                    io.post([commandCopy = command, responseCopy = viscaResponse]() {
                        commandCopy->callback(responseCopy);
                    });
                }

                activeCommands.erase(it);
                break;
            }
        }
    }

    void checkTimeouts(const boost::system::error_code& error) {
        if (error == boost::asio::error::operation_aborted) {
            return;  // Timer was cancelled
        }

        boost::lock_guard<boost::mutex> lock(mutex);
        auto now = boost::chrono::steady_clock::now();

        // Check active commands for timeouts
        for (auto it = activeCommands.begin(); it != activeCommands.end();) {
            auto command = it->second;

            auto timeout = std::chrono::milliseconds(command->timeout);
            auto elapsed =
                boost::chrono::duration_cast<boost::chrono::milliseconds>(now - command->timestamp);

            if (elapsed.count() > timeout.count()) {
                // The command most probably timed out in this case
                ViscaResponse response;
                response.type = ViscaResponse::Type::Error;
                response.errorCode = 0xFE;  // Custom timeout error code

                command->promise.set_value(response);

                if (command->callback) {
                    io.post([commandCopy = command, responseCopy = response]() {
                        commandCopy->callback(responseCopy);
                    });
                }

                it = activeCommands.erase(it);
            } else {
                ++it;
            }
        }

        if (running) {
            scheduleTimeoutCheck();
        }
    }

    void pushPendingCommand(boost::shared_ptr<CommandInfo> command) {
        boost::lock_guard<boost::mutex> lock(pendingCommandsMutex);
        pendingCommands.push(command);
    }
};

class ViscaProtocolCreator : public ViscaProtocol {
public:
    static boost::shared_ptr<ViscaProtocol> create(boost::shared_ptr<Serial> serialPort) {
        return boost::shared_ptr<ViscaProtocol>(new ViscaProtocol(serialPort));
    }
};

ViscaProtocol::ViscaProtocol(boost::shared_ptr<Serial> serialPort)
    : pImpl(boost::make_shared<Impl>(serialPort)) {
    pImpl->start();
}

ViscaProtocol::~ViscaProtocol() { pImpl->stop(); }

boost::shared_ptr<ViscaProtocol> ViscaProtocol::create(boost::shared_ptr<Serial> serialPort) {
    return ViscaProtocolCreator::create(serialPort);
}

boost::unique_future<ViscaResponse> ViscaProtocol::sendCommand(const std::vector<uint8_t>& command,
                                                               ViscaCommandType type,
                                                               std::chrono::milliseconds timeout) {
    auto commandInfo = boost::make_shared<CommandInfo>(type, timeout, command);
    boost::unique_future<ViscaResponse> future = commandInfo->promise.get_future();

    // Send the command
    try {
        pImpl->serialPort->write(command);

        // Queue the command info
        pImpl->pushPendingCommand(commandInfo);
    } catch (const SerialPortError& e) {
        // Handle write error
        ViscaResponse errorResponse;
        errorResponse.type = ViscaResponse::Type::Error;
        errorResponse.errorCode =
            0xFF;  // Custom error code for write failure - no specific standard for these codes yet

        commandInfo->promise.set_value(errorResponse);
    }

    return future;
}

void ViscaProtocol::sendCommandAsync(const std::vector<uint8_t>& command, ResponseCallback callback,
                                     ViscaCommandType type, std::chrono::milliseconds timeout) {
    auto commandInfo = boost::make_shared<CommandInfo>(type, timeout, command);
    commandInfo->callback = callback;

    try {
        pImpl->serialPort->write(command);

        pImpl->pushPendingCommand(commandInfo);
    } catch (const SerialPortError& e) {
        ViscaResponse errorResponse;
        errorResponse.type = ViscaResponse::Type::Error;
        errorResponse.errorCode = 0xFF;

        if (callback) {
            callback(errorResponse);
        }
    }
}

std::vector<uint8_t> ViscaProtocol::createCameraCommand(const std::vector<uint8_t>& payload) {
    std::vector<uint8_t> command;
    command.reserve(payload.size() + 2);

    command.push_back(VISCA_HEADER_CAMERA);
    command.insert(command.end(), payload.begin(), payload.end());
    command.push_back(VISCA_TERMINATOR);

    return command;
}

std::vector<uint8_t> ViscaProtocol::createInterfaceCommand(const std::vector<uint8_t>& payload) {
    std::vector<uint8_t> command;
    command.reserve(payload.size() + 2);

    command.push_back(VISCA_HEADER_INTERFACE);
    command.insert(command.end(), payload.begin(), payload.end());
    command.push_back(VISCA_TERMINATOR);

    return command;
}

std::vector<uint8_t> ViscaProtocol::createCameraInquiry(const std::vector<uint8_t>& payload) {
    std::vector<uint8_t> command;
    command.reserve(payload.size() + 2);

    command.push_back(VISCA_HEADER_CAMERA_INQUIRY);
    command.insert(command.end(), payload.begin(), payload.end());
    command.push_back(VISCA_TERMINATOR);

    return command;
}

std::vector<uint8_t> ViscaProtocol::createInterfaceInquiry(const std::vector<uint8_t>& payload) {
    std::vector<uint8_t> command;
    command.reserve(payload.size() + 2);

    command.push_back(VISCA_HEADER_INTERFACE_INQUIRY);
    command.insert(command.end(), payload.begin(), payload.end());
    command.push_back(VISCA_TERMINATOR);

    return command;
}

void ViscaProtocol::setVSyncPeriod(std::chrono::milliseconds period) {
    pImpl->vSyncPeriod = period;
}

std::chrono::milliseconds ViscaProtocol::getVSyncPeriod() const { return pImpl->vSyncPeriod; }

void ViscaProtocol::cancelAll() {
    boost::lock_guard<boost::mutex> lock(pImpl->mutex);
    boost::lock_guard<boost::mutex> pendingLock(pImpl->pendingCommandsMutex);

    ViscaResponse errorResponse;
    errorResponse.type = ViscaResponse::Type::Error;
    errorResponse.errorCode = 0xFD;  // Custom error code for cancellation

    // Cancel pending commands (it basically just drains the queue so that the queue won't be
    // overflown with pending commands)
    while (!pImpl->pendingCommands.empty()) {
        auto command = pImpl->pendingCommands.front();
        pImpl->pendingCommands.pop();

        command->promise.set_value(errorResponse);

        if (command->callback) {
            auto commandCopy = command;
            auto responseCopy = errorResponse;
            pImpl->io.post([commandCopy, responseCopy]() { commandCopy->callback(responseCopy); });
        }
    }

    // Cancel active commands
    for (auto it = pImpl->activeCommands.begin(); it != pImpl->activeCommands.end(); ++it) {
        auto command = it->second;
        command->promise.set_value(errorResponse);

        if (command->callback) {
            auto commandCopy = command;
            auto responseCopy = errorResponse;
            pImpl->io.post([commandCopy, responseCopy]() { commandCopy->callback(responseCopy); });
        }
    }

    pImpl->activeCommands.clear();
}

}  // namespace harrier