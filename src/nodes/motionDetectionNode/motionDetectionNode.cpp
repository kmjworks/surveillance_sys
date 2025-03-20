#include "motionDetectionNode.hpp"

// This file is intentionally kept minimal since the implementation
// is now in the header file using templates.

// Explicit template instantiation for the default motion detection strategy
template class surveillance_system::MotionDetectorNode<
    surveillance_system::FrameDifferenceStrategy<cv::Mat>>;
