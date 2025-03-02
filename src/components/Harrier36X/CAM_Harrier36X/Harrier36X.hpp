#pragma once

#include "ViscaProtocol.hpp"

#include <boost/enable_shared_from_this.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread/mutex.hpp>
#include <string>

namespace harrier {
class Harrier36X : public boost::enable_shared_from_this<Harrier36X> {
public:
    enum class ZoomDirection { STOP, IN, OUT };
    enum class FocusDirection { STOP, NEAR, FAR };
    enum class FocusMode { AUTO, MANUAL, ONE_PUSH };
    enum class ExposureMode { AUTO, MANUAL, SHUTTER_PRIORITY, IRIS_PRIORITY, BRIGHT };
    enum class DayNightMode { AUTO, DAY, NIGHT, EXTERNAL };
    enum class BacklightMode { OFF, ON, SPOT };
    enum class WhiteBalanceMode { AUTO, INDOOR, OUTDOOR, ONE_PUSH, MANUAL, AUTO_EXTENDED };

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

private:
    explicit Harrier36X(boost::shared_ptr<ViscaProtocol> protocol);
    bool checkConnection();
    void setLastError(const std::string& error);
    bool processCommandResponse(const ViscaResponse& response);

    boost::shared_ptr<ViscaProtocol> protocol;
    bool connected;
    std::string lastError;
    mutable boost::mutex mutex;
};
}  // namespace harrier