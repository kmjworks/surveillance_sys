#pragma once

#include <boost/enable_shared_from_this.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread/mutex.hpp>
#include <string>
#include <utility>
#include "../CAM_Harrier36X/Harrier36X.hpp"

namespace harrier {
class HarrierUSBHDMI : public boost::enable_shared_from_this<HarrierUSBHDMI> {
public:
    enum class VideoFormat {
        DEFAULT_CAMERA_MODE = 0,
        FORMAT_1080P60 = 1,
        FORMAT_1080P59_94 = 2,
        FORMAT_1080P50 = 3,
        FORMAT_1080P30 = 4,
        FORMAT_1080P29 = 5,
        FORMAT_1080P25 = 6,
        FORMAT_1080I60 = 7,
        FORMAT_1080I59_94 = 8,
        FORMAT_1080I50 = 9,
        FORMAT_720P60 = 10,
        FORMAT_720P59_94 = 11,
        FORMAT_720P50 = 12,
        FORMAT_720P30 = 13,
        FORMAT_720P29 = 14,
        FORMAT_720P25 = 15
    };

    enum class LvdsMode { SINGLE, DUAL };

    enum class SyncOutputMode {
        VSYNC = 0,
        VSYNC_INVERTED = 1,
        HSYNC = 2,
        HSYNC_INVERTED = 3,
        FSYNC = 4,
        FSYNC_INVERTED = 5,
        LOGIC_LOW = 6,
        LOGIC_HIGH = 7,
        PWM = 8
    };

    using InterfaceCallback = boost::function<void(bool success, const std::string& errorMessage)>;

    static boost::shared_ptr<HarrierUSBHDMI> create(boost::shared_ptr<Harrier36X> camera) {
        return boost::shared_ptr<HarrierUSBHDMI>(new HarrierUSBHDMI(std::move(camera)));
    }

    ~HarrierUSBHDMI() = default;

private:
    explicit HarrierUSBHDMI(boost::shared_ptr<Harrier36X> camera);
    bool checkConnection();
    void setLastError(const std::string& error);
    bool sendInterfaceCommand(const std::vector<uint8_t>& payload,
                              std::chrono::milliseconds timeout = std::chrono::milliseconds(1000));

    bool processInterfaceResponse(const ViscaResponse& response);

    boost::shared_ptr<Harrier36X> camera;
    std::string lastError;
    mutable boost::mutex mutex;
};
}  // namespace harrier