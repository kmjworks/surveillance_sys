#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import sys

# Add your YOLO repo to sys.path so that we can import it
#sys.path.append('/home/user/ros_ws/src/yolov5')  # Adjust this to your YOLO repo path

# valvesysteem/src/nodes/motionDetectionNode/object_detection

# Load the YOLO model (change the path to your best.pt model)
model = torch.hub.load('ultralytics/yolov5', 'custom', 
	path='yolov11_trained.pt', source='local')
model.eval()

class YoloRosNode:
    def __init__(self):
        # Convert ROS Image messages to OpenCV images
        self.bridge = CvBridge()
        
        # Subscribe to the image topic
        rospy.Subscriber("/camera/image_raw", Image, self.callback)
        
        # Publish the detection results
        self.pub = rospy.Publisher("/yolo/image", Image, queue_size=1)

    def callback(self, msg):
        # Convert the ROS Image message to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Run YOLO detection
        results = model(cv_image)  # Run inference on the image
        output = results.render()[0]  # Get the rendered image with bounding boxes
        
        # Publish the result to a new topic
        self.pub.publish(self.bridge.cv2_to_imgmsg(output, "bgr8"))

if __name__ == "__main__":
    rospy.init_node("yolo_ros_node")
    YoloRosNode()
    rospy.spin()

