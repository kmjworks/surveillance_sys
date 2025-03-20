
#include "HarrierUSBHDMI.hpp"
#include <boost/format.hpp>
#include <boost/optional/optional.hpp>
#include <boost/thread/lock_guard.hpp>
#include <iostream>
#include <sstream>
#include <utility>
#include "ViscaProtocol.hpp"

namespace harrier {
HarrierUSBHDMI::HarrierUSBHDMI(boost::shared_ptr<Harrier36X> camera) : camera(std::move(camera)) {}

boost::optional<std::string> HarrierUSBHDMI::getLastError() const {
    boost::lock_guard<boost::mutex> lock(mutex);
    return lastError;
}

bool HarrierUSBHDMI::setVideoFormat(VideoFormat format) {
    if (!checkConnection()) {
        return false;
    }

    // Skip for default camera mode
    if (format == VideoFormat::DEFAULT_CAMERA_MODE) {
        return true;
    }

    // Check if we need to set LVDS mode based on the format
    bool requiresDualLvds =
        (format == VideoFormat::FORMAT_1080P60 || format == VideoFormat::FORMAT_1080P59_94 ||
         format == VideoFormat::FORMAT_1080P50);

    if (requiresDualLvds) {
        if (!setLvdsMode(LvdsMode::DUAL)) {
            return false;
        }
    } else {
        if (!setLvdsMode(LvdsMode::SINGLE)) {
            return false;
        }
    }

    uint8_t monitoringMode;
    switch (format) {
        case VideoFormat::FORMAT_1080P59_94:
            monitoringMode = 0x13;
            break;
        case VideoFormat::FORMAT_1080P50:
            monitoringMode = 0x14;
            break;
        case VideoFormat::FORMAT_1080P30:
            monitoringMode = 0x06;
            break;
        case VideoFormat::FORMAT_1080P25:
            monitoringMode = 0x08;
            break;
        case VideoFormat::FORMAT_1080I60:
            monitoringMode = 0x00;
            break;
        case VideoFormat::FORMAT_1080I59_94:
            monitoringMode = 0x00;
            break;
        case VideoFormat::FORMAT_1080I50:
            monitoringMode = 0x01;
            break;
        case VideoFormat::FORMAT_720P60:
            monitoringMode = 0x09;
            break;
        case VideoFormat::FORMAT_720P59_94:
            monitoringMode = 0x09;
            break;
        case VideoFormat::FORMAT_720P50:
            monitoringMode = 0x0C;
            break;
        case VideoFormat::FORMAT_720P30:
            monitoringMode = 0x0E;
            break;
        case VideoFormat::FORMAT_720P29:
            monitoringMode = 0x0E;
            break;
        case VideoFormat::FORMAT_720P25:
            monitoringMode = 0x11;
            break;
        default:
            return false;
    }

    // Set monitoring mode register (72h)
    std::vector<uint8_t> payload = {0x01, 0x04, 0x24, 0x72, 0x00, monitoringMode};

    // Send command to camera (not interface board)
    try {
        auto future = camera->getProtocol()->sendCommand(
            ViscaProtocol::createCameraCommand(payload), ViscaCommandType::Command);
        auto response = future.get();

        if (response.isError()) {
            setLastError("Failed to set video format: Error code " +
                         std::to_string(response.errorCode));
            return false;
        }

        // Reset camera to apply the changes
        // return camera->reset();
    } catch (const std::exception& e) {
        setLastError(std::string("Exception during set video format: ") + e.what());
        return false;
    }

    return true;
}

bool HarrierUSBHDMI::resetInterfaceBoard() {
    if (!checkConnection()) {
        return false;
    }

    std::vector<uint8_t> payload = {0x01, 0x0A, 0x00};

    return sendInterfaceCommand(payload, std::chrono::milliseconds(5000));
}

boost::optional<uint8_t> HarrierUSBHDMI::getBoardHealth() {
    if (!checkConnection()) {
        return boost::none;
    }

    // Query interface board health
    std::vector<uint8_t> payload = {0x09, 0x0A, 0x02};

    try {
        auto future = camera->getProtocol()->sendCommand(
            ViscaProtocol::createInterfaceInquiry(payload), ViscaCommandType::Inquiry);

        auto response = future.get();

        if (response.isError() || !response.isData()) {
            setLastError("Failed to get board health: Error code " +
                         std::to_string(response.errorCode));
            return boost::none;
        }

        // Parse response (expected format: A0 50 r1 r2 FF)
        const auto& data = response.data;
        if (data.size() >= 5) {
            return data[2];
        }

        setLastError("Invalid response format for board health inquiry");
        return boost::none;
    } catch (const std::exception& e) {
        setLastError(std::string("Exception during get board health: ") + e.what());
        return boost::none;
    }
}

boost::optional<std::string> HarrierUSBHDMI::getFirmwareVersion() {
    if (!checkConnection()) {
        return boost::none;
    }

    std::vector<uint8_t> payload = {0x09, 0x0A, 0x00};

    try {
        auto future = camera->getProtocol()->sendCommand(
            ViscaProtocol::createInterfaceInquiry(payload), ViscaCommandType::Inquiry);

        auto response = future.get();

        if (response.isError() || !response.isData()) {
            setLastError("Failed to get firmware version: Error code " +
                         std::to_string(response.errorCode));
            return boost::none;
        }

        // Parse response (expected format: A0 50 r1 r2 r3 FF)
        const auto& data = response.data;
        if (data.size() >= 6) {
            std::stringstream ss;
            ss << static_cast<int>(data[2]) << "." << static_cast<int>(data[3]) << "."
               << static_cast<int>(data[4]);
            return ss.str();
        }

        setLastError("Invalid response format for firmware version inquiry");
        return boost::none;
    } catch (const std::exception& e) {
        setLastError(std::string("Exception during get firmware version: ") + e.what());
        return boost::none;
    }
}

boost::optional<int> HarrierUSBHDMI::getBoardTemperature() {
    if (!checkConnection()) {
        return boost::none;
    }

    // Query interface board health (includes temperature)
    std::vector<uint8_t> payload = {0x09, 0x0A, 0x02};

    try {
        auto future = camera->getProtocol()->sendCommand(
            ViscaProtocol::createInterfaceInquiry(payload), ViscaCommandType::Inquiry);

        auto response = future.get();

        if (response.isError() || !response.isData()) {
            setLastError("Failed to get board temperature: Error code " +
                         std::to_string(response.errorCode));
            return boost::none;
        }

        // Parse response (expected format: A0 50 r1 r2 FF)
        const auto& data = response.data;
        if (data.size() >= 5) {
            // Temperature is in r2 with a +60°C offset
            return static_cast<int>(data[3]) - 60;
        }

        setLastError("Invalid response format for board temperature inquiry");
        return boost::none;
    } catch (const std::exception& e) {
        setLastError(std::string("Exception during board temperature inquiry: ") + e.what());
        return boost::none;
    }
}

}  // namespace harrier