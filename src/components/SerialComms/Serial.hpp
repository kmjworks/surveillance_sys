#pragma once

#include <memory>
#include <string>
#include <vector>

class Serial {
public:
    enum class BaudRate { B9600, B19200, B38400, B57600, B115200 };
    enum class DataBits { DATABITS_7, DATABITS_8 };
}