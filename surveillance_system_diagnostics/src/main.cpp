#include "diagnosticsNode.hpp"

int main(int argc, char** argv) {
    ros::init(argc, argv, "diagnostics_node");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    
    DiagnosticsNode diagnostics(nh, private_nh);
        
    ros::spin();
    return 0;
}