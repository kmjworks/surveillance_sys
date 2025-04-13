#pragma once

#include "ViscaProtocol.hpp"

#include <boost/enable_shared_from_this.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread/mutex.hpp>
#include <string>

namespace harrier {

struct RuntimeCameraParameters;

class Harrier36X : public boost::enable_shared_from_this<Harrier36X> {
public:
    enum class ZoomDirection : uint8_t { STOP, IN, OUT };
    enum class FocusDirection : uint8_t { STOP, NEAR, FAR };
    enum class FocusMode : uint8_t { AUTO, MANUAL, ONE_PUSH };
    enum class ExposureMode : uint8_t { AUTO, MANUAL, SHUTTER_PRIORITY, IRIS_PRIORITY, BRIGHT };
    enum class DayNightMode : uint8_t { AUTO, DAY, NIGHT, EXTERNAL };
    enum class BacklightMode : uint8_t { OFF, ON, SPOT };
    enum class WhiteBalanceMode : uint8_t {
        AUTO,
        INDOOR,
        OUTDOOR,
        ONE_PUSH,
        MANUAL,
        AUTO_EXTENDED
    };

    using CameraCallback = boost::function<void(bool success, const std::string& errorMessage)>;
    using CommandCompletionHandler = boost::function<void(const ViscaResponse&)>;

    static boost::shared_ptr<Harrier36X> create(boost::shared_ptr<ViscaProtocol> protocol) {
        return boost::shared_ptr<Harrier36X>(new Harrier36X(protocol));
    }
    ~Harrier36X();
    bool connect(void);
    void disconnect(void);
    bool isConnected(void) const;
    std::string getLastError(void) const;
    boost::shared_ptr<ViscaProtocol> getProtocol() const { return protocol; }
    void cancelAllPendingCommands(void);
    void setRuntimeParameters(void);

private:
    explicit Harrier36X(boost::shared_ptr<ViscaProtocol> protocol);
    bool checkConnection();
    void setLastError(const std::string& error);
    bool processCommandResponse(const ViscaResponse& response);
    RuntimeCameraParameters getRuntimeParameters(void);

    boost::shared_ptr<ViscaProtocol> protocol;
    bool connected;
    std::string lastError;
    mutable boost::mutex mutex;
};

struct RuntimeCameraParameters {
    Harrier36X::ZoomDirection zoomDirection;
    Harrier36X::ExposureMode exposureMode;
    Harrier36X::BacklightMode backlightEnabled;
    Harrier36X::DayNightMode dayNightMode;
};

RuntimeCameraParameters runtimeParams;

}  // namespace harrier
