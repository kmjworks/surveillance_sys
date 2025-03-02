#include "Serial.hpp"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

namespace harrier {
struct Serial::Impl {
    std::string devicePath;
    SerialPortConfig config;
    int fd = -1;
    std::atomic<bool> isRunning{false};
    std::thread readThread;
    ReadCallback readCallback;
    std::size_t minReadBytes = 1;
    std::mutex mutex;
    std::condition_variable cv;

    Impl(const std::string& path, const SerialPortConfig& cfg) : devicePath(path), config(cfg) {}

    ~Impl() {
        if (fd >= 0) {
            stopReadThread();
            ::close(fd);
            fd = -1;
        }
    }

    void startReadThread() {
        isRunning = true;
        readThread = std::thread([this]() { readThreadFunc(); });
    }

    void stopReadThread() {
        if (readThread.joinable()) {
            isRunning = false;
            cv.notify_all();
            readThread.join();
        }
    }

    void readThreadFunc() {
        std::vector<uint8_t> buffer(1024);

        pollfd pfd = {fd, POLLIN, 0};

        while (isRunning) {
            int ret = ::poll(&pfd, 1, 100);

            if (ret < 0) {
                if (errno == EINTR)
                    continue;
                readCallback({}, std::error_code(errno, std::system_category()));
                break;
            } else if (ret > 0 && (pfd.revents & POLLIN)) {
                int bytesAvailable;
                if (::ioctl(fd, FIONREAD, &bytesAvailable) < 0) {
                    readCallback({}, std::error_code(errno, std::system_category()));
                    continue;
                }

                if (static_cast<std::size_t>(bytesAvailable) >= minReadBytes) {
                    buffer.resize(bytesAvailable);
                    ssize_t bytesRead = ::read(fd, buffer.data(), bytesAvailable);

                    if (bytesRead < 0) {
                        readCallback({}, std::error_code(errno, std::system_category()));
                    } else if (bytesRead > 0) {
                        buffer.resize(bytesRead);
                        readCallback(buffer, std::error_code());
                    }
                }
            }
        }
    }
};

Serial::Serial(const std::string& devicePath, const SerialPortConfig& config)
    : ptrImpl(std::make_unique<Impl>(devicePath, config)) {}

Serial::~Serial() { close(); }

std::shared_ptr<Serial> Serial::create(const std::string& devicePath,
                                       const SerialPortConfig& config) {
    std::shared_ptr<Serial> port(new Serial(devicePath, config));
    port->open();
    return port;
}

void Serial::open() {
    if (isOpen())
        return;

    ptrImpl->fd = ::open(ptrImpl->devicePath.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (ptrImpl->fd < 0) {
        throw SerialPortError("Failed to open device port " + ptrImpl->devicePath + ":" +
                              strerror(errno));
    }

    setConfig(ptrImpl->config);
}

void Serial::close() {
    if (isOpen()) {
        ptrImpl->stopReadThread();
        ::close(ptrImpl->fd);
        ptrImpl->fd = -1;
    }
}

bool Serial::isOpen() const { return ptrImpl->fd >= 0; }

std::size_t Serial::write(const std::vector<uint8_t>& data) {
    if (!isOpen()) {
        throw SerialPortError("Serial port not open.");
    }

    fd_set wfds;
    struct timeval tval;

    FD_ZERO(&wfds);
    FD_SET(ptrImpl->fd, &wfds);

    tval.tv_sec = ptrImpl->config.writeTimeout.count() / 1000;
    tval.tv_usec = (ptrImpl->config.writeTimeout.count() % 1000) * 1000;

    int ret = select(ptrImpl->fd + 1, nullptr, &wfds, nullptr, &tval);
    if (ret < 0) {
        throw SerialPortError("Select failed during write: " + std::string(strerror(errno)));
    }
    if (ret == 0) {
        throw SerialPortError("Write timeout");
    }

    ssize_t bytesWritten = ::write(ptrImpl->fd, data.data(), data.size());
    if (bytesWritten < 0) {
        throw SerialPortError("Write error: " + std::string(strerror(errno)));
    }

    return static_cast<std::size_t>(bytesWritten);
}

std::vector<uint8_t> Serial::read(std::size_t maxBytes) {
    if (!isOpen()) {
        throw SerialPortError("Serial port not open");
    }

    // Set up timeout using select
    fd_set rfds;
    struct timeval tval;

    FD_ZERO(&rfds);
    FD_SET(ptrImpl->fd, &rfds);

    tval.tv_sec = ptrImpl->config.readTimeout.count() / 1000;
    tval.tv_usec = (ptrImpl->config.readTimeout.count() % 1000) * 1000;

    int ret = select(ptrImpl->fd + 1, &rfds, nullptr, nullptr, &tval);
    if (ret < 0) {
        throw SerialPortError("Select failed during read: " + std::string(strerror(errno)));
    }
    if (ret == 0) {
        // Timeout, return empty buffer
        return {};
    }

    std::vector<uint8_t> buffer(maxBytes);
    ssize_t bytesRead = ::read(ptrImpl->fd, buffer.data(), maxBytes);

    if (bytesRead < 0) {
        throw SerialPortError("Read error: " + std::string(strerror(errno)));
    }

    buffer.resize(bytesRead);
    return buffer;
}

std::vector<uint8_t> Serial::readUntil(uint8_t delimiter, std::size_t maxBytes) {
    if (!isOpen()) {
        throw SerialPortError("Serial port not open.");
    }

    std::vector<uint8_t> result;
    auto timeout = std::chrono::steady_clock::now() + ptrImpl->config.readTimeout;

    while (result.size() < maxBytes) {
        if (std::chrono::steady_clock::now() > timeout) {
            if (result.empty()) {
                throw SerialPortError("Read timeout");
            }
            break;
        }

        // Check for available data
        fd_set rfds;
        struct timeval tval;

        FD_ZERO(&rfds);
        FD_SET(ptrImpl->fd, &rfds);

        // Use small timeout for the polling loop
        tval.tv_sec = 0;
        tval.tv_usec = 100000;  // 100ms

        int ret = select(ptrImpl->fd + 1, &rfds, nullptr, nullptr, &tval);
        if (ret < 0) {
            throw SerialPortError("Select failed: " + std::string(strerror(errno)));
        }
        if (ret == 0) {
            // No data available yet, continue polling
            continue;
        }

        // Read one byte at a time
        uint8_t byte;
        ssize_t bytesRead = ::read(ptrImpl->fd, &byte, 1);

        if (bytesRead < 0) {
            throw SerialPortError("Read error: " + std::string(strerror(errno)));
        }
        if (bytesRead == 0) {
            // No data read, try again
            continue;
        }

        result.push_back(byte);

        if (byte == delimiter) {
            break;
        }
    }

    return result;
}

void Serial::setReadCallback(ReadCallback callback, std::size_t minBytes) {
    if (!isOpen()) {
        throw SerialPortError("Serial port not open");
    }

    ptrImpl->stopReadThread();
    ptrImpl->readCallback = std::move(callback);
    ptrImpl->minReadBytes = minBytes;

    if (ptrImpl->readCallback) {
        ptrImpl->startReadThread();
    }
}

void Serial::cancelReadCallback() {
    ptrImpl->stopReadThread();
    ptrImpl->readCallback = nullptr;
}

void Serial::flush() const {
    if (!isOpen()) {
        throw SerialPortError("Serial port not open");
    }

    ::tcflush(ptrImpl->fd, TCIOFLUSH);
}

SerialPortConfig Serial::getConfig() const { return ptrImpl->config; }

void Serial::setConfig(const SerialPortConfig& config) {
    if (!isOpen()) {
        ptrImpl->config = config;
        return;
    }

    struct termios tty;
    memset(&tty, 0, sizeof(tty));

    if (::tcgetattr(ptrImpl->fd, &tty) != 0) {
        throw SerialPortError("Failed to get terminal attributes: " + std::string(strerror(errno)));
    }

    speed_t baudRate;
    switch (config.baudRate) {
        case 4800:
            baudRate = B4800;
            break;
        case 9600:
            baudRate = B9600;
            break;
        case 19200:
            baudRate = B19200;
            break;
        case 38400:
            baudRate = B38400;
            break;
        case 57600:
            baudRate = B57600;
            break;
        case 115200:
            baudRate = B115200;
            break;
        default:
            throw SerialPortError("Unsupported baud rate: " + std::to_string(config.baudRate));
    }

    cfsetispeed(&tty, baudRate);
    cfsetospeed(&tty, baudRate);

    tty.c_cflag &= ~CSIZE;
    switch (config.dataBits) {
        case 5:
            tty.c_cflag |= CS5;
            break;
        case 6:
            tty.c_cflag |= CS6;
            break;
        case 7:
            tty.c_cflag |= CS7;
            break;
        case 8:
            tty.c_cflag |= CS8;
            break;
        default:
            throw SerialPortError("Unsupported data bits: " + std::to_string(config.dataBits));
    }

    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_iflag &= ~(INPCK | ISTRIP);
    switch (config.parity) {
        case SerialPortConfig::Parity::None:
            break;
        case SerialPortConfig::Parity::Odd:
            tty.c_cflag |= (PARENB | PARODD);
            tty.c_iflag |= (INPCK | ISTRIP);
            break;
        case SerialPortConfig::Parity::Even:
            tty.c_cflag |= PARENB;
            tty.c_iflag |= (INPCK | ISTRIP);
            break;
        default:
            throw SerialPortError("Unsupported parity setting.");
    }

    tty.c_cflag &= ~CSTOPB;
    if (config.stopBits == 2) {
        tty.c_cflag |= CSTOPB;
    } else if (config.stopBits != 1) {
        throw SerialPortError("Unsupported stop bits: " + std::to_string(config.stopBits));
    }

    tty.c_cflag &= ~CRTSCTS;
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    switch (config.flowControl) {
        case SerialPortConfig::FlowControl::None:
            break;
        case SerialPortConfig::FlowControl::Hardware:
            tty.c_cflag |= CRTSCTS;
            break;
        case SerialPortConfig::FlowControl::Software:
            tty.c_iflag |= (IXON | IXOFF);
            break;
        default:
            throw SerialPortError("Unsupported flow control setting");
    }

    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ECHONL | ISIG | IEXTEN);
    tty.c_iflag &= ~(IGNBRK | BRKINT | ICRNL | INLCR | PARMRK | INPCK | ISTRIP);
    tty.c_oflag &= ~(OCRNL | ONLCR | ONLRET | ONOCR | OFILL | OPOST);

    tty.c_cc[VMIN] = 0;   // Min number of characters for read
    tty.c_cc[VTIME] = 1;  // 0.1 seconds read timeout

    if (::tcsetattr(ptrImpl->fd, TCSANOW, &tty) != 0) {
        throw SerialPortError("Failed to set terminal attributes: " + std::string(strerror(errno)));
    }

    ptrImpl->config = config;
}
}  // namespace harrier