# Bounding box of the detected object in pixel coordinates
# uint32 x_offset  # Leftmost pixel of the ROI
# uint32 y_offset  # Topmost pixel of the ROI
# uint32 height    # Height of ROI
# uint32 width     # Width of ROI
# bool do_rectify   # True if ROI is calculated from rectified image
sensor_msgs/RegionOfInterest bounding_box

# Centroid of the detected object in pixel coordinates
geometry_msgs/Point centroid

# Tracking ID assigned to this detection
int32 id

# Confidence score of the detection (Reserved for futre)
# float32 score