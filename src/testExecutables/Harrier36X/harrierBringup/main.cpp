#include <boost/make_shared.hpp>
#include <boost/smart_ptr/make_shared_object.hpp>
#include <iostream>
#include "Harrier36X.hpp"
#include "Serial.hpp"
#include "ViscaProtocol.hpp"

using namespace harrier;
int main(void) {
    SerialPortConfig config;
    config.baudRate = 9600;
    config.dataBits = 8;
    config.stopBits = 1;
    config.parity = SerialPortConfig::Parity::None;
    config.readTimeout = std::chrono::milliseconds(500);

    auto port = Serial::create("/dev/ttyUSB0", config);

    port->open();

    auto protocol = ViscaProtocol::create(port);

    return 0;
}