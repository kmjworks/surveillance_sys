#ifndef HARRIER_SERIAL_HPP
#define HARRIER_SERIAL_HPP

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace harrier {
class SerialPortError : public std::runtime_error {
public:
    explicit SerialPortError(const std::string& message) : std::runtime_error(message) {}
};

/**
 * @brief Configuration for serial port
 */
struct SerialPortConfig {
    int baudRate = 9600;
    int dataBits = 8;
    int stopBits = 1;
    enum class Parity : uint8_t { None, Odd, Even } parity = Parity::None;
    enum class FlowControl : uint8_t { None, Hardware, Software } flowControl = FlowControl::None;
    std::chrono::milliseconds readTimeout = std::chrono::milliseconds(1000);
    std::chrono::milliseconds writeTimeout = std::chrono::milliseconds(1000);
};

class Serial {
public:
    using ReadCallback = std::function<void(const std::vector<uint8_t>&, std::error_code)>;

    static std::shared_ptr<Serial> create(const std::string& devicePath,
                                          const SerialPortConfig& config = {});

    ~Serial();

    Serial(const Serial&) = delete;
    Serial& operator=(const Serial&) = delete;

    void open();

    void close();

    bool isOpen() const;

    std::size_t write(const std::vector<uint8_t>& data);
    std::vector<uint8_t> read(std::size_t maxBytes);
    void flush();

    std::vector<uint8_t> readUntil(uint8_t delimiter, std::size_t maxBytes = 1024);
    void setReadCallback(ReadCallback callback, std::size_t minBytes);
    void cancelReadCallback();

    SerialPortConfig getConfig() const;
    void setConfig(const SerialPortConfig& config);

private:
    Serial(const std::string& devicePath, const SerialPortConfig& config);
    struct Impl;
    std::unique_ptr<Impl> ptrImpl;
};

}  // namespace harrier

#endif  // HARRIER_SERIAL_HPP