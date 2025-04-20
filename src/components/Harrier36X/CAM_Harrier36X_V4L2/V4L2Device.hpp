#include <string>
#include <linux/videodev2.h>
#include <stdexcept>
#include <boost/optional/optional.hpp>

class HarrierDeviceState {
public:
    HarrierDeviceState(const int deviceDescriptor, bool verboseState);
    ~HarrierDeviceState();

    void queryCapabilities();
    void queryFormats();

private:
    int deviceHandle;
    bool verboseQueries;

    struct v4l2_fmtdesc formats;
    struct v4l2_capability caps;
};

class V4L2Device {
    public:
        V4L2Device(const std::string& devicePath);
        boost::optional<HarrierDeviceState> getDeviceHandle() const;
    private:  
        const std::string& path;
};