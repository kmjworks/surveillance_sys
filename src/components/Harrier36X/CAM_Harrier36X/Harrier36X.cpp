#include "Harrier36X.hpp"
#include <boost/bind.hpp>
#include <boost/format.hpp>
#include <boost/thread/lock_guard.hpp>
#include <iostream>
#include <sstream>

namespace harrier {
Harrier36X::Harrier36X(boost::shared_ptr<ViscaProtocol> protocol)
    : protocol(protocol), connected(false) {}
Harrier36X::~Harrier36X() { disconnect(); }
bool Harrier36X::connect() {
    boost::lock_guard<boost::mutex> lock(mutex);
    if (connected) {
        return true;
    }

    try {
        auto future = protocol->sendCommand(ViscaProtocol::createCameraCommand({0x01, 0x00, 0x01}));
        auto response = future.get();

        if (response.isError()) {
            setLastError("Failed to clear command buffers: Error code " +
                         std::to_string(response.errorCode));
            return false;
        }

        auto zoomQuery = protocol->sendCommand(ViscaProtocol::createCameraInquiry({0x04, 0x47}),
                                               ViscaCommandType::Inquiry);

        auto queryResponse = zoomQuery.get();
        if (queryResponse.isError()) {
            setLastError("Failed to communicate with camera: Error code " +
                         std::to_string(queryResponse.errorCode));
            return false;
        }

        connected = true;
        return true;
    } catch (const std::exception& e) {
        setLastError(std::string("Exception during connection: ") + e.what());
        return false;
    }
}

void Harrier36X::disconnect() {
    boost::lock_guard<boost::mutex> lock(mutex);

    if (!connected) {
        return;
    }

    protocol->cancelAll();

    connected = false;
}

bool Harrier36X::isConnected() const {
    boost::lock_guard<boost::mutex> lock(mutex);
    return connected;
}

std::string Harrier36X::getLastError() const {
    boost::lock_guard<boost::mutex> lock(mutex);
    return lastError;
}

void Harrier36X::cancelAllPendingCommands() { protocol->cancelAll(); }

bool Harrier36X::processCommandResponse(const ViscaResponse& response) {
    if (response.isError()) {
        setLastError("Command failed with error code: " + std::to_string(response.errorCode));
        return false;
    }

    if (!response.isCompletion()) {
        setLastError("Unexpected response type: " + std::to_string(response.errorCode));
        return false;
    }

    return true;
}

void Harrier36X::setRuntimeParameters() {
    RuntimeCameraParameters params = getRuntimeParameters();
    params.backlightEnabled = Harrier36X::BacklightMode::SPOT;
    params.dayNightMode = Harrier36X::DayNightMode::AUTO;
    params.exposureMode = Harrier36X::ExposureMode::AUTO;
    params.zoomDirection = Harrier36X::ZoomDirection::IN;
}

RuntimeCameraParameters Harrier36X::getRuntimeParameters() { return runtimeParams; }

}  // namespace harrier
