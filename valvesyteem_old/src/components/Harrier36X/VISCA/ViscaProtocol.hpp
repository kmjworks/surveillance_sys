#pragma once

#include <chrono>
#include <functional>
#include <string>
#include <vector>

// Boost includes
#include <boost/asio.hpp>
#include <boost/bind/bind.hpp>
#include <boost/optional.hpp>
#include <boost/signals2.hpp>
#include <boost/thread/future.hpp>

namespace harrier {

class Serial;

enum class ViscaCommandType { Command, Inquiry, InterfaceCommand };

struct ViscaResponse {
    enum class Type { Acknowledge, Completion, Error, Data };

    Type type;
    uint8_t socketNumber = 0;   // For ACK responses
    std::vector<uint8_t> data;  // For data responses
    uint8_t errorCode = 0;      // For error responses

    // Helper methods to check response type
    bool isAcknowledge() const { return type == Type::Acknowledge; }
    bool isCompletion() const { return type == Type::Completion; }
    bool isError() const { return type == Type::Error; }
    bool isData() const { return type == Type::Data; }
};

class ViscaError : public std::runtime_error {
public:
    explicit ViscaError(const std::string& message) : std::runtime_error(message) {}
};

class ViscaProtocol : public boost::enable_shared_from_this<ViscaProtocol> {
public:
    using ResponseCallback = std::function<void(const ViscaResponse&)>;

    static boost::shared_ptr<ViscaProtocol> create(boost::shared_ptr<Serial> serialPort);

    ViscaProtocol(boost::shared_ptr<Serial> serialPort);
    ~ViscaProtocol();

    // Delete copy constructor and assignment operator
    ViscaProtocol(const ViscaProtocol&) = delete;
    ViscaProtocol& operator=(const ViscaProtocol&) = delete;

    boost::unique_future<ViscaResponse> sendCommand(
        const std::vector<uint8_t>& command, ViscaCommandType type = ViscaCommandType::Command,
        std::chrono::milliseconds timeout = std::chrono::milliseconds(1000));

    void sendCommandAsync(const std::vector<uint8_t>& command, ResponseCallback callback,
                          ViscaCommandType type = ViscaCommandType::Command,
                          std::chrono::milliseconds timeout = std::chrono::milliseconds(1000));

    static std::vector<uint8_t> createCameraCommand(const std::vector<uint8_t>& payload);
    static std::vector<uint8_t> createCameraInquiry(const std::vector<uint8_t>& payload);
    static std::vector<uint8_t> createInterfaceCommand(const std::vector<uint8_t>& payload);
    static std::vector<uint8_t> createInterfaceInquiry(const std::vector<uint8_t>& payload);

    void setVSyncPeriod(std::chrono::milliseconds period);
    std::chrono::milliseconds getVSyncPeriod() const;
    void cancelAll();

private:
    struct Impl;
    boost::shared_ptr<Impl> pImpl;
};

}  // namespace harrier
