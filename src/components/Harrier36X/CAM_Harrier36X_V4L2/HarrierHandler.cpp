#include "HarrierHandler.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <thread>

using namespace std::chrono; 

Harrier::Harrier(HarrierUSBHandle handle) : commsInternal {handle, HarrierCommsOK, std::vector<uint8_t>(128)}  {
   unsigned char bytesTmp = 0;
   
   std::cout << "Harrier constructor: initializing with handle " << handle << "\n";
   if (handle < 0) {
      throw std::runtime_error("Invalid Harrier USB handle (negative value)");
   }
   
   HarrierCommsError_t status = HarrierCommsUSBReceive(commsInternal.device, commsInternal.onReplyBuffer.data(),   commsInternal.onReplyBuffer.capacity(), &bytesTmp,     100);
}

Harrier::~Harrier() {
    if(commsInternal.device >= 0) {
        internal::CommunicationState status = HarrierCommsUSBClose(commsInternal.device);
        if(status != HarrierCommsOK) {
            std::cerr << "Warning: Failed to close Harrier USB device: " << status << "\n";
        }
    }
}

internal::ViscaPacket Harrier::createViscaCommandPacket(const uint8_t* command) {
    internal::ViscaPacket cmdPacket;
    cmdPacket.packet = command;
    cmdPacket.cmdType = internal::CommandType::CONTROL;
    
    size_t size = 0;
    while (size < 32) {
        if (command[size] == 0xFF) {
            cmdPacket.packetSize = size + 1;
            getTargetCmdProperties(cmdPacket);
            return cmdPacket;
        }
        size++;
    }
    
    throw std::runtime_error("Invalid VISCA command: too long or missing termination");
}

internal::ViscaPacket Harrier::createViscaInquiryPacket(const uint8_t* command) {
    internal::ViscaPacket inqPacket;
    inqPacket.packet = command;
    inqPacket.cmdType = internal::CommandType::INQUIRY;
    
    size_t size = 0;
    while (size < 32) {
        if (command[size] == 0xFF) {
            inqPacket.packetSize = size + 1;
            getTargetCmdProperties(inqPacket);
            return inqPacket;
        }
        size++;
    }
    
    throw std::runtime_error("Invalid VISCA inquiry: too long or missing termination");
}

void Harrier::sendCommand(const internal::ViscaPacket& cmdPacket) {
    internal::CommunicationState status = HarrierCommsUSBTransmit(commsInternal.device,cmdPacket.packet, cmdPacket.packetSize);
    throwOnInternalCommsError(status, internal::CommandType::CONTROL);

    unsigned char cmdReplyBytes = 0;
    commsInternal.onReplyBuffer.resize(128);
    {
        std::lock_guard<std::mutex> lock(commsMtx);
        status = HarrierCommsUSBReceive(commsInternal.device, commsInternal.onReplyBuffer.data(), commsInternal.onReplyBuffer.size(), &cmdReplyBytes, 200);
        throwOnInternalCommsError(status, internal::CommandType::CONTROL);
    }
    
    std::this_thread::sleep_for(milliseconds(100));
}

void Harrier::sendInquiry(const internal::ViscaPacket& inqPacket) {
    internal::CommunicationState status = HarrierCommsUSBTransmit(commsInternal.device, inqPacket.packet, inqPacket.packetSize);
    throwOnInternalCommsError(status, internal::CommandType::INQUIRY);
    
    unsigned char bytesReceived = 0;
    commsInternal.onReplyBuffer.resize(128);
    {
        std::lock_guard<std::mutex> lock(commsMtx);
        status = HarrierCommsUSBReceive(commsInternal.device,  commsInternal.onReplyBuffer.data(), commsInternal.onReplyBuffer.size(), &bytesReceived, 500);
        throwOnInternalCommsError(status, internal::CommandType::INQUIRY);
    }
    commsInternal.onReplyBuffer.resize(bytesReceived);
}

void Harrier::getTargetCmdProperties(internal::ViscaPacket& targetPacket) {
    if (targetPacket.packetSize < 3) {
        throw std::runtime_error("VISCA packet too short");
    }
    
    uint8_t header = targetPacket.packet[0];
    if ((header & 0x80) == 0) {
        throw std::runtime_error("Invalid VISCA packet header");
    }
}

void Harrier::populateTargetCmd(uint8_t& targetCmd) {
    targetCmd = (targetCmd & 0xF0) | 0x01;
}

void Harrier::throwOnInternalCommsError(internal::CommunicationState commandStatus, internal::CommandType cmdType) {
    if(commandStatus != HarrierCommsOK) {
        std::stringstream errStream;
        errStream << "Harrier " << ((cmdType == internal::CommandType::CONTROL) ? "Control " : "Inquiry ") << " failed with error code: " << commandStatus;
        throw std::runtime_error(errStream.str());
    }
}
