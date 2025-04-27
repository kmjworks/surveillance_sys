#include "HarrierCommsUSB.h"
#include <boost/optional/optional.hpp>
#include <mutex>


namespace internal {

    enum CommandType : uint8_t {
        INQUIRY,
        CONTROL
    };

    struct ViscaPacket {
        const uint8_t* packet;
        size_t packetSize;
        CommandType cmdType;
    };

    using DeviceHandle = HarrierUSBHandle;
    using CommunicationState = HarrierCommsError_t;
    using RxBuffer = std::vector<uint8_t>;

    struct CommunicationInterface {
        DeviceHandle device;
        CommunicationState status;
        RxBuffer onReplyBuffer;
    };
}




class Harrier {
    public:
        Harrier(HarrierUSBHandle handle);
        ~Harrier();

        internal::ViscaPacket createViscaCommandPacket(const uint8_t* command);
        internal::ViscaPacket createViscaInquiryPacket(const uint8_t* command);
        void sendCommand(const internal::ViscaPacket& cmdPacket);
        void sendInquiry(const internal::ViscaPacket& inqPacket);

    private:
        void getTargetCmdProperties(internal::ViscaPacket& targetPacket);
        void populateTargetCmd(uint8_t& targetCmd);
        void throwOnInternalCommsError(internal::CommunicationState commandStatus, internal::CommandType cmdType);

        internal::CommunicationInterface commsInternal;
        std::mutex commsMtx;
};

class USBDevice {
    public:
        USBDevice(const int deviceIndex) : deviceIdentifier(deviceIndex) {}
        template<typename HandleType>
        boost::optional<HandleType> getDeviceHandle() const {
            int deviceHandle;
            HarrierCommsError_t status = HarrierCommsUSBOpenByIndex(&deviceHandle, deviceIdentifier);
            if(status != HarrierCommsOK) {
                throw std::runtime_error("Harrier USB Open failed.");
            }

            return HandleType(deviceHandle);
        }

    private:
        int deviceIdentifier;
};
