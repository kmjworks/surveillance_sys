# Surveillance System

## Description

A modular surveillance system with motion detection, ReID, tracking and recording capabilities. 

## TODO / Problems

- DeepSORT (Tracker node - profiled with nsys)
  - Memory transfers: Host-to-device and device-to-host memory transfers are taking a lot of time
     - Potential fix: utilize pinned memory for host buffers
  - Abnormal periods of thread blocking on I/O
     - Synchronization overhead and cuda sync calls

- Frame storage
  - Writing the frames out on a SSD takes a lot of time despite being off-loaded to multiple threads 
     - This bottlenecks the system as a whole
  - Potential fix: cloud storage? TBA

- Comments
  - The code has very little comments complementing the implementation
  - Comments will be added over development over a arbitrary time period
    - This is not the main priority as of this moment

## Performance

The system has been ran and tested on a Jetson Xavier NX 16GB module. 

- End-to-end throughput is yet to be determined.
- Individual components: 
   - DeepSORT (Tracker node) has an average processing time of 16.7ms
   - YOLOv8s (Detector, DeepStream pipeline) has an average processing time of 17.23ms



## System Architecture

The surveillance system is divided into multiple ROS packages:

**Obsolete components**:
- **surveillance_system_pipeline**: Replaced by **surveillance_system_detection_deepstream** combining the video pipeline and detector inference
- **surveillance_system_detection**: Same as above

**Active system components**

- **surveillance_system_msgs**: Message and service definitions
- **surveillance_system_common**: Common utilities and shared components
- **surveillance_system_camera**: Camera interface (not completed and is not currently actively used)
- **surveillance_system_detection_deepstream**: Motion detection using YOLOv8
- **surveillance_system_motion_tracking**: Object tracking using DeepSORT
   -  DeepSORT's components (Tracker and the interface itself are under the common components directory)
- **surveillance_system_capture**: Frame capture and storage
   -  Not utilized actively as storing on the platform itself is computationally expensive in terms of I/O operations
- **surveillance_system_diagnostics**: System diagnostics and monitoring
- **surveillance_system**: Meta-package containing launch files and configurations

## Dependencies

- ROS Noetic
- OpenCV 4.x with CUDA support
- TensorRT 8.x
- CUDA 11.x

## Building

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src

git clone https://github.com/kmjworks/surveillance_system.git .

cd ~/catkin_ws
catkin build
```

## Launch

The system can currently only be launched with a single configuration:

### Production System

```bash
roslaunch surveillance_system system.launch
```

## Configuration

Configuration parameters are stored in YAML files in the `config` directory:

- `system_params.yaml`: Production system parameters

## Models

The DeepSORT CNN and YOLOv8 detection model are stored in the `models` directory:

- `yolov8_finetuned_fp16.engine`: Detection model (TensorRT engine, built on a unique platform i.e. can't be ran)
- `deepsort.engine`: Tracking model (TensorRT engine, built on a unique platform i.e. can't be ran)
- .onnx formats are not currently included 


## License

This project is licensed under the MIT license - see the LICENSE file for details.
