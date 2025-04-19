#include <gtest/gtest.h>
#include <ros/ros.h>
#include "../src/nodes/pipelineNode/pipelineNode.hpp"

class PipelineParamsTest : public ::testing::Test {
protected:
    std::unique_ptr<ros::NodeHandle> nh_priv_ptr;

    void SetUp() override {
        nh_priv_ptr = std::make_unique<ros::NodeHandle>("~");
    }

    void TearDown() override {
        nh_priv_ptr.reset();
    }

    ros::NodeHandle& getPrivateNodeHandle() {
        return *nh_priv_ptr;
    }
};

TEST_F(PipelineParamsTest, LoadDefaultParameters) {
    pipeline::ConfigurationParameters params;
    ros::NodeHandle& nh_priv = getPrivateNodeHandle();

    nh_priv.deleteParam("device_path");
    nh_priv.deleteParam("frame_rate");
    nh_priv.deleteParam("night_mode");
    nh_priv.deleteParam("show_debug_frames");
    nh_priv.deleteParam("motion_detector/sampling_rate");
    nh_priv.deleteParam("buffer_size");
    nh_priv.deleteParam("output_path");
    nh_priv.deleteParam("motion_detector/min_area_px");
    nh_priv.deleteParam("motion_detector/downscale");
    nh_priv.deleteParam("motion_detector/history");

    int bufferSize = PipelineNode::loadParameters(nh_priv, params);

    ASSERT_EQ(params.devicePath, "/dev/video0");
    ASSERT_EQ(params.frameRate, 30);
    ASSERT_FALSE(params.nightMode);
    ASSERT_EQ(params.bufferSize, 10);
    ASSERT_EQ(bufferSize, 10);
}

TEST_F(PipelineParamsTest, LoadCustomParameters) {
    pipeline::ConfigurationParameters params;
    ros::NodeHandle& nh_priv = getPrivateNodeHandle();

    nh_priv.setParam("device_path", "/dev/video99");
    nh_priv.setParam("frame_rate", 15);
    nh_priv.setParam("night_mode", true);
    nh_priv.setParam("buffer_size", 25);

    int bufferSize = PipelineNode::loadParameters(nh_priv, params);

    ASSERT_EQ(params.devicePath, "/dev/video99");
    ASSERT_EQ(params.frameRate, 15);
    ASSERT_TRUE(params.nightMode);
    ASSERT_EQ(params.bufferSize, 25);
    ASSERT_EQ(bufferSize, 25);

    nh_priv.deleteParam("device_path");
    nh_priv.deleteParam("frame_rate");
    nh_priv.deleteParam("night_mode");
    nh_priv.deleteParam("buffer_size");
    nh_priv.deleteParam("motion_detector/sampling_rate");
    nh_priv.deleteParam("motion_detector/min_area_px");
}


int main(int argc, char **argv) {
    ros::init(argc, argv, "pipeline_params_test_node");
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}