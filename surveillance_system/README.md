# Surveillance System

## System Architecture

The surveillance system is divided into multiple ROS packages:

- **surveillance_system_msgs**: Message and service definitions
- **surveillance_system_common**: Common utilities and shared components
- **surveillance_system_camera**: Camera interface (not completed and is not currently actively used)
- **surveillance_system_pipeline**: Video processing pipeline
    - Replaced by surveillance_system_detection_deepstream combining the pipeline and the detector
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

git clone https://your-git-server/surveillance_system.git .

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

- `yolov8_finetuned_fp16.engine`: Detection model (TensorRT engine)
- `deepsort.engine`: Tracking model (TensorRT engine)
- .onnx formats are not currently included


## License

This project is licensed under the MIT license - see the LICENSE file for details.
