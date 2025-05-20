# Surveillance System

A modular ROS package for a comprehensive surveillance system with motion detection, tracking, and recording capabilities.

## System Architecture

The surveillance system is divided into multiple ROS packages:

- **surveillance_system_msgs**: Message and service definitions
- **surveillance_system_common**: Common utilities and shared components
- **surveillance_system_camera**: Camera interface (not utilized currently)
- **surveillance_system_pipeline**: Video processing pipeline
- **surveillance_system_motion_detection**: Motion detection using YOLOv8s
- **surveillance_system_motion_tracking**: Object tracking using DeepSORT
- **surveillance_system_capture**: Frame capture and storage
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

The system can be launched in a single configurations:

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
- .onnx formats not included (for now) 


## License

This project is licensed under the MIT license - see the LICENSE file for details.
