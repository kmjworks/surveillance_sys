# Surveillance System

A modular ROS package for a comprehensive surveillance system with motion detection, tracking, and recording capabilities.

## System Architecture

The surveillance system is divided into multiple ROS packages:

- **surveillance_system_msgs**: Message and service definitions
- **surveillance_system_common**: Common utilities and shared components
- **surveillance_system_camera**: Camera interface and simulation
- **surveillance_system_pipeline**: Video processing pipeline
- **surveillance_system_motion_detection**: AI-based motion detection using YOLOv8
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
# Create a catkin workspace
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src

# Clone the repository
git clone https://your-git-server/surveillance_system.git .

# Build the packages
cd ~/catkin_ws
catkin build
```

## Launch

The system can be launched in different configurations:

### Production System

```bash
roslaunch surveillance_system system.launch
```

### Simulation Mode

```bash
roslaunch surveillance_system simulation.launch
```

## Configuration

Configuration parameters are stored in YAML files in the `config` directory:

- `system_params.yaml`: Production system parameters
- `simulation_params.yaml`: Simulation parameters

## Models

The AI models are stored in the `models` directory:

- `yolov8_finetuned_fp16.engine`: Detection model (TensorRT engine)
- `deepsort.engine`: Tracking model (TensorRT engine)

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the [insert license name] - see the LICENSE file for details.