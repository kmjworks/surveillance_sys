#pragma once

#include <boost/chrono.hpp>
#include <boost/function.hpp>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <string>
#include <vector>
namespace harrier {
class SerialPortError : public std::runtime_error {
public:
    explicit SerialPortError(const std::string& message) : std::runtime_error(message) {}
};

struct SerialPortConfig {
    int baudRate = 9600;
    int dataBits = 8;
    int stopBits = 1;

    enum class Parity : uint8_t { None, Odd, Even };
    Parity parity = Parity::None;

    enum class FlowControl : uint8_t { None, Hardware, Software };
    FlowControl flowControl = FlowControl::None;

    boost::chrono::milliseconds readTimeout{1000};
    boost::chrono::milliseconds writeTimeout{1000};
};

class Serial : private boost::noncopyable {
public:
    using ReadCallback =
        boost::function<void(const std::vector<uint8_t>&, const boost::system::error_code&)>;

    explicit Serial(const std::string& devicePath, int baudRate = 9600);

    Serial(const std::string& devicePath, const SerialPortConfig& config);

    ~Serial();

    static boost::shared_ptr<Serial> create(const std::string& devicePath,
                                            const SerialPortConfig& config = SerialPortConfig());

    void open();

    void close();

    bool isOpen() const;

    std::size_t write(const std::vector<uint8_t>& data);
    std::vector<uint8_t> read(std::size_t maxBytes = 256);
    void flush() const;

    std::vector<uint8_t> readUntil(uint8_t delimiter, std::size_t maxBytes = 1024);
    void setReadCallback(ReadCallback callback, std::size_t minBytes);
    void cancelReadCallback();

    SerialPortConfig getConfig() const;
    void setConfig(const SerialPortConfig& config);

private:
    struct Impl;
    std::unique_ptr<Impl> ptrImpl;
};

}  // namespace harrier
