#ifndef HARRIER_COMMS_USB_H_
#define HARRIER_COMMS_USB_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>

typedef int HarrierUSBHandle;
typedef void *HarrierCommsHandle;

typedef enum {
  HarrierCommsOK = 0,
  HarrierCommsAccessFail,
  HarrierCommsRXOutofBounds,
  HarrierCommsRXBufferOverflow,
  HarrierCommsRXEmpty,
  HarrierCommsOpenFail,
  HarrierCommsBadIndex,
  HarrierCommsBadHandle,
  HarrierCommsBufferTooSmall,
  HarrierCommsBadArgs,
} HarrierCommsError_t;

HarrierCommsError_t HarrierCommsUSBOpenByIndex(int *handle, int index);

HarrierCommsError_t HarrierCommsUSBClose(int handle);

HarrierCommsError_t HarrierCommsUSBTransmit(int handle, const void *msg,
    size_t size);

HarrierCommsError_t HarrierCommsUSBReceive(
        int handle, void *buffer, size_t bufsize, unsigned char *bytes,
        int timeout);
    

#ifdef __cplusplus
}
#endif
#endif // HARRIER_COMMS_USB_H
    


