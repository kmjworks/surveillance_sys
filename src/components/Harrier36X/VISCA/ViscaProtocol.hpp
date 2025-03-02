#ifndef HARRIER_VISCA_PROTOCOL_HPP
#define HARRIER_VISCA_PROTOCOL_HPP

#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace harrier {
class Serial;

enum class ViscaCommandType { Command, Inquiry, InterfaceCommand };

struct ViscaResponse {
    enum class Type { Acknowledge, Completion, Error, Data };

    Type type;

    uint8_t socketNumber = 0;
    std::vector<uint8_t> data;
    uint8_t errorCode = 0;

    bool isAcknowledge() const { return type == Type::Acknowledge; }
    bool isCompletion() const { return type == Type::Completion; }
    bool isError() const { return type == Type::Error; }
    bool isData() const { return type == Type::Data; }
};

class ViscaError : public std::runtime_error {
public:
    explicit ViscaError(const std::string& message) : std::runtime_error(message) {}
};

class ViscaProtocol {
    using ResponseCallback = std::function<void(const ViscaResponse&)>;

    static std::shared_ptr<ViscaProtocol> create(std::shared_ptr<Serial> serialPort);

    ~ViscaProtocol();

    ViscaProtocol(const ViscaProtocol&) = delete;
    ViscaProtocol& operator=(const ViscaProtocol&) = delete;

    std::future<ViscaResponse> sendCommand(
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
    ViscaProtocol(std::shared_ptr<Serial> serialPort);

    struct Impl;
    std::unique_ptr<Impl> ptrImpl;
};

}  // namespace harrier

#endif  // HARRIER_VISCA_PROTOCOL_HPP