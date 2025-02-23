#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>

class CameraNode
{
    private:
        ros::NodeHandle nh_; 
        image_transport::ImageTransport it_; 
        image_transport::Publisher image_pub;

    public: 
        CameraNode() : nh_("~"), it_(nh_) 
        {
            image_pub = it_.advertise("image_raw", 1);
        }

        void run()
        {
            ROS_INFO("Camera Node running.");
            ros::spin();
        }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "camera_node");
    CameraNode node; 
    node.run(); 
    return 0; 
}
