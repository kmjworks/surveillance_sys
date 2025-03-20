#include <boost/make_shared.hpp>
#include <iostream>
#include "Harrier36X.hpp"
#include "Serial.hpp"
#include "ViscaProtocol.hpp"

int main(void) {
    auto serialConfig = harrier::SerialPortConfig();
    serialConfig.baudRate = 9600;
    serialConfig.dataBits = 8;
    serialConfig.stopBits = 1;
    serialConfig.parity = harrier::SerialPortConfig::Parity::None;
    serialConfig.flowControl = harrier::SerialPortConfig::FlowControl::None;

    auto port = harrier::Serial::create("/dev/ttyUSB0", serialConfig);
    auto protocol = harrier::ViscaProtocol::create(port);
    auto camera = harrier::Harrier36X::create(protocol);

    if (!camera->connect()) {
        std::cerr << "Failed to connect to camera: " << camera->getLastError() << std::endl;
        return 1;
    }

    std::cout << "Connected to camera successfully" << std::endl;

    return 0;
}