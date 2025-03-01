#include "Serial.hpp"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <string.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <system_error>
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
    }
};

}  // namespace harrier