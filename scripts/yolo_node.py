#!/usr/bin/env python3

import rospy
import rospkg
import os
import sys
import time
import cv2
from ultralytics import YOLO

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge, CvBridgeError


class YoloNode:
    def __init__(self):
        """Initialize the YOLO model using ultralytics as a ROS node."""
        rospy.loginfo("Initializing YOLO Object Detection Node...")

        # Required parameters for initialization
        self.model_filename = rospy.get_param("~model_filename", "yolov5n.pt") 

       # Optional parameters (can be overridden)
        self.input_topic = rospy.get_param("~input_topic", "pipeline/runtime_potentialMotionEvents")
        self.detection_topic = rospy.get_param("~detection_topic", "yolo/runtime_detections")
        self.visualization_topic = rospy.get_param("~visualization_topic", "yolo/runtime_detectionVisualizationDebug")
        self.confidence_threshold = rospy.get_param("~confidence_threshold", 0.4)
        self.device = rospy.get_param("~device", "auto") 
        self.enable_visualization = rospy.get_param("~enable_visualization", True)
        self.enable_logging = rospy.get_param("~enable_logging", False)
        self.log_dir = rospy.get_param("~log_dir", "/tmp/yolo_objectDetections")

        try:
            rospack_instance = rospkg.RosPack()
            self.package_path = rospack_instance.get_path('surveillance_system')
        except rospkg.ResourceNotFound:
            rospy.logfatal("Unable to locate package 'surveillance_system'. Shutting down.")
            rospy.signal_shutdown("Package not found")
            return

        self.model_abs_path = os.path.join(self.package_path, "models", self.model_filename)
       

        # Device selection (cuda(gpu)/cpu)
        # Ultralytics handles 'auto' well, but a explicit check is still worth it just in case
        # It uses torch backend, so torch check is still valid if torch is installed as dependency
        try:
            import torch # Check torch availability locally for device check
            if self.device == 'auto':
                self.selected_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.selected_device = self.device 

            if "cuda" in self.selected_device and not torch.cuda.is_available():
                rospy.logwarn(f"Device '{self.selected_device}' requested but CUDA is not available, defaulting to CPU.")
                self.selected_device = "cpu"
        except ImportError:
             rospy.logwarn("PyTorch not found, cannot verify CUDA availability. Attempting device='{}'.".format(self.device))
             self.selected_device = self.device if self.device != 'auto' else 'cpu' # Default to CPU if unsure

        rospy.loginfo(f"Using device: '{self.selected_device}' for inference.")

        # Ultralytics model loading
        self.model = None
        try:
            if not os.path.isfile(self.model_abs_path):
                raise FileNotFoundError(f"Model file not found at: {self.model_abs_path}")

            rospy.loginfo(f"Loading Ultralytics YOLO model from: {self.model_abs_path}")
            self.model = YOLO(self.model_abs_path) 
            # self.model.to(self.selected_device) 

            # We can do a dummy inference to warm up the model (if we decide that this is necessary)
            # self.model.predict(np.zeros((640, 640, 3)), verbose=False)

            rospy.loginfo(f"Ultralytics YOLO model loaded successfully from {self.model_abs_path}")
            rospy.loginfo(f"Model class names: {self.model.names}")

        except FileNotFoundError as e:
            rospy.logfatal(f"{e}. Shutting down.")
            rospy.signal_shutdown("Model file not found")
            return
        except Exception as e:
            rospy.logfatal(f"Error loading Ultralytics YOLO model from {self.model_abs_path}: {e}")
            rospy.logfatal("Ensure 'pip install ultralytics' is done and the model path is correct.")
            rospy.signal_shutdown("Model load failed")
            return

        # ROS specific inter-node communication setup 
        self.bridge = CvBridge()
        self.sub_imageData = rospy.Subscriber(self.input_topic, Image, self.callback, queue_size=1, buff_size=2**24)
        self.pub_detectedObjects = rospy.Publisher(self.detection_topic, Detection2DArray, queue_size=10)
        if self.enable_visualization:
            self.pub_detectedObjectsVisualization = rospy.Publisher(self.visualization_topic, Image, queue_size=1)
        else:
            self.pub_detectedObjectsVisualization = None

        # Logging
        if self.enable_logging:
            if not os.path.exists(self.log_dir):
                try:
                    os.makedirs(self.log_dir)
                    rospy.loginfo(f"Created log directory: {self.log_dir}")
                except OSError as e:
                    rospy.logerr(f"Failed to create log directory {self.log_dir}: {e}. Disabling logging.")
                    self.enable_logging = False

        rospy.loginfo("YOLO object detection node initialization OK.")

    def callback(self, msg):
        try:
            cvImage = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        if self.model is None:
            rospy.logwarn("Model not loaded, skipping inference.")
            return

        try:
            start = time.time()
            # Ultralytics predict
            results_list = self.model.predict(
                source=cvImage,
                device=self.selected_device,
                conf=self.confidence_threshold,
                verbose=False 
                )

            # predict returns a list of Results objects, typically one per image
            if not results_list:
                 rospy.logwarn("YOLO prediction returned empty list.")
                 return
            results = results_list[0] 
            end = time.time()
            # rospy.loginfo(f"Inference time: {end - start:.4f} s") 

            detectionArrayMsg = Detection2DArray()
            detectionArrayMsg.header = msg.header
            detected = False

            # Ultralytics results' parser
            boxes = results.boxes # Access the Boxes object containing detections
            if boxes is not None and len(boxes) > 0:
                 detected = True
                 for i in range(len(boxes)):
    
                    box = boxes.xyxy[i].int().tolist() # Coordinates as a list [x1, y1, x2, y2]
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                    conf = boxes.conf[i].item() # Confidence score (float)
                    clsIdentifier = int(boxes.cls[i].item()) # Class ID (int)
                    label = self.model.names[clsIdentifier] # Object classifier (string)

                    # Detection2D message creation for coordinates
                    detection = Detection2D()
                    detection.header = msg.header # Use the original image header
                    # Bounding box
                    detection.bbox.center.x = float((x1 + x2) / 2)
                    detection.bbox.center.y = float((y1 + y2) / 2)
                    detection.bbox.size_x = float(x2 - x1)
                    detection.bbox.size_y = float(y2 - y1)
                    # hypothesis = ObjectHypothesisWithPose() 
                    # hypothesis.id = clsIdentifier 
                    # hypothesis.score = conf 
                    # detection.results.append(hypothesis) 

            
                    hypothesis = ObjectHypothesisWithPose() 
                    hypothesis.id = clsIdentifier 
                    hypothesis.score = conf
                    detection.results.append(hypothesis)


                    detectionArrayMsg.detections.append(detection)


            # Detection publisher
            if detected:
                self.pub_detectedObjects.publish(detectionArrayMsg)

            if detected and self.enable_logging:
                try:
                    timestamp = msg.header.stamp.to_sec()
                    filename = os.path.join(self.log_dir, f"detection_{timestamp:.6f}.png")
                    log_image = results.plot(conf=True, labels=True) 
                    cv2.imwrite(filename, log_image)
                except Exception as e:
                    rospy.logerr(f"Failed to log image to {self.log_dir}: {e}")

            
            if self.enable_visualization and self.pub_detectedObjectsVisualization is not None and self.pub_detectedObjectsVisualization.get_num_connections() > 0:
                try:
                    output_image = results.plot(conf=True, labels=True)
                    viz_msg = self.bridge.cv2_to_imgmsg(output_image, "bgr8")
                    self.pub_detectedObjectsVisualization.publish(viz_msg)
                except CvBridgeError as e:
                    rospy.logerr(f"CvBridge Error for visualization: {e}")
                except Exception as e:
                    rospy.logerr(f"Failed to create/publish visualization image: {e}")

        except Exception as e:
            rospy.logerr(f"YOLO inference or processing error in callback: {e}", exc_info=True) 


if __name__ == "__main__":
    try:
        rospy.init_node("objectDetectionNode", anonymous=True)
        node = YoloNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("YOLO Object detection node shutting down.")
    except Exception as e:
        rospy.logfatal(f"Unhandled exception in YOLO Object detection node __main__: {e}", exc_info=True) 