#!/usr/bin/env python3
import os
import sys
import subprocess
import platform
import logging
from pathlib import Path
import xml.etree.ElementTree as ET
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("system_setup")

def execute_command(command, description, exit_on_error=True):
    logger.info(f"Executing: {description}")
    try:
        process = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        logger.debug(process.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        if exit_on_error:
            logger.error(f"Exiting due to error in: {description}")
            sys.exit(1)
        return False

def check_environment():
 
    logger.info("[Step 1/5] Checking environment...")
    if platform.system() != "Linux":
        logger.error("This script must be run on a Linux system.")
        sys.exit(1)
    try:
        ubuntu_version = subprocess.check_output(
            "lsb_release -rs", shell=True, universal_newlines=True
        ).strip()
        if ubuntu_version != "20.04":
            logger.warning(f"This script is designed for Ubuntu 20.04, but detected {ubuntu_version}")
            response = input("Continue anyway? [y/N]: ").lower()
            if response != 'y':
                sys.exit(1)
    except:
        logger.warning("Could not determine Ubuntu version.")
    
    ros_distro = os.environ.get("ROS_DISTRO")
    if ros_distro != "noetic":
        logger.error(f"ROS Noetic required, but ROS_DISTRO={ros_distro}")
        logger.error("Please install ROS Noetic or source the appropriate setup.bash file.")
        sys.exit(1)

    package_xml = Path("package.xml")
    if not package_xml.exists():
        logger.error("package.xml not found. Please run this script from the package directory.")
        sys.exit(1)

    try:
        tree = ET.parse(package_xml)
        root = tree.getroot()
        package_name = root.find("name").text
        if package_name != "surveillance_system":
            logger.warning(f"Expected package 'surveillance_system', but found '{package_name}'")
            response = input("Continue anyway? [y/N]: ").lower()
            if response != 'y':
                sys.exit(1)
    except Exception as e:
        logger.error(f"Error parsing package.xml: {e}")
        sys.exit(1)
        
    logger.info("Environment checks passed.")

def install_system_packages():
    logger.info("[Step 2/5] Installing system packages...")
    
    execute_command(
        "sudo apt update",
        "Updating package lists"
    )
    
    system_packages = [
        "python3-pip",
        "git",
        "build-essential",
        "cmake",
        "libopencv-dev",
        "libgstreamer1.0-dev",
        "libgstreamer-plugins-base1.0-dev",
        "gstreamer1.0-plugins-good",
        "gstreamer1.0-plugins-bad",
        "gstreamer1.0-plugins-ugly",
        "pkg-config"
    ]
    
    execute_command(
        f"sudo apt install -y {' '.join(system_packages)}",
        "Installing system packages"
    )
    
    logger.info("System packages installed successfully.")

def setup_rosdep():

    logger.info("[Step 3/5] Setting up rosdep...")
    
    rosdep_initialized = execute_command(
        "rosdep --version",
        "Checking if rosdep is available",
        exit_on_error=False
    )
    
    if not rosdep_initialized:
        logger.warning("Rosdep not available. Installing python3-rosdep...")
        execute_command(
            "sudo apt install -y python3-rosdep",
            "Installing python3-rosdep"
        )
    
    rosdep_list_exists = os.path.exists("/etc/ros/rosdep/sources.list.d/20-default.list")
    if not rosdep_list_exists:
        logger.info("Initializing rosdep...")
        execute_command(
            "sudo rosdep init",
            "Initializing rosdep"
        )
    
    execute_command(
        "rosdep update",
        "Updating rosdep"
    )
    
    logger.info("Rosdep setup completed successfully.")

def install_ros_dependencies():

    logger.info("[Step 4/5] Installing ROS dependencies using rosdep...")

    current_dir = os.getcwd()
    workspace_src = None
    path = Path(current_dir)
    while path != path.parent:
        if (path / "src").exists() and (path / "devel").exists():
            workspace_src = path / "src"
            break
        if path.name == "src" and (path.parent / "devel").exists():
            workspace_src = path
            break
        path = path.parent
    
    if not workspace_src:
        workspace_src = Path(current_dir).parent
        logger.warning(f"Could not determine workspace src directory. Using: {workspace_src}")
    logger.info(f"Using workspace src directory: {workspace_src}")
    
    execute_command(
        f"rosdep install --from-paths {workspace_src} --ignore-src -y",
        "Installing ROS dependencies"
    )
    
    logger.info("ROS dependencies installed successfully.")

def install_python_packages():
    """
    Install Python packages using pip with --user flag.
    """
    logger.info("[Step 5/5] Installing Python packages...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        with open(requirements_file, "w") as f:
            f.write("numpy>=1.19.0\n")  # Compatible with Jetson
            f.write("pillow>=8.0.0\n")
            f.write("matplotlib>=3.3.0\n")
            f.write("pyyaml>=5.3.0\n")

    execute_command(
        f"pip3 install --user -r {requirements_file}",
        "Installing Python packages"
    )
    
    execute_command(
        "pip3 install --user rospkg catkin_pkg",
        "Installing ROS Python packages"
    )
    
    logger.info("Python packages installed successfully.")

def print_final_instructions():
    """
    Print final instructions for building the workspace.
    """
    logger.info("Setup completed successfully!")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("-"*80)
    print("1. Navigate to your workspace root:")
    print("   cd ../../")
    print("\n2. Build the workspace:")
    print("   catkin_make")
    print("   or")
    print("   catkin build")
    print("\n3. Source the setup file:")
    print("   source devel/setup.bash")
    print("\n4. To automatically source the setup file in every new terminal:")
    print("   echo 'source ~/catkin_ws/devel/setup.bash' >> ~/.bashrc")
    print("\n5. Test the installation:")
    print("   roslaunch surveillance_system camera.launch")
    print("="*80 + "\n")

def main():

    start_time = time.time()
    
    print("\n" + "="*80)
    print("SURVEILLANCE SYSTEM - DEPENDENCY SETUP")
    print("="*80)
    print("This script will set up all dependencies for the surveillance_system ROS package.")
    print("Prerequisites:")
    print("  - Ubuntu 20.04 LTS")
    print("  - ROS Noetic installed")
    print("  - Script executed from within the package directory")
    print("-"*80 + "\n")
    
    try:
        check_environment()
        install_system_packages()
        setup_rosdep()
        install_ros_dependencies()
        install_python_packages()
        print_final_instructions()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Setup completed in {elapsed_time:.2f} seconds.")
        
    except KeyboardInterrupt:
        logger.error("Setup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()