#include <gtest/gtest.h>
#include "../src/nodes/pipelineNode/components/pipelineInternal.hpp"
#include "../src/nodes/pipelineNode/components/pipelineInitialDetection.hpp"
#include <opencv2/opencv.hpp>

class PipelineTests : public ::testing::Test {
protected:
    void SetUp() override {
        testFrame = cv::Mat(480, 640, CV_8UC3, cv::Scalar(128,128,128));

        cv::rectangle(testFrame, cv::Point(100,100), cv::Point(200,200), cv::Scalar(255,0,0), -1);

        motionFrame = testFrame.clone();

        cv::rectangle(motionFrame, cv::Point(150,150), cv::Point(250,250), cv::Scalar(255,0,0), -1);
    }

    cv::Mat testFrame;
    cv::Mat motionFrame;
};

TEST_F(PipelineTests, PipelineInternalInitialize) {
    PipelineInternal internal(false, false);

    EXPECT_TRUE(internal.initialize());
}

TEST_F(PipelineTests, PipelineInternalFrameProcessing) {
    PipelineInternal internal(false, true);
    internal.initialize();

    cv::Mat processed = internal.processFrame(testFrame);

    EXPECT_FALSE(processed.empty());
    EXPECT_EQ(processed.size(), testFrame.size());
}

TEST_F(PipelineTests, PipelineInternalNightMode) {
    PipelineInternal internal(true, false);
    internal.initialize();

    cv::Mat processed = internal.processFrame(testFrame);
    
    EXPECT_FALSE(processed.empty());
    EXPECT_EQ(processed.channels(), 1); // grayscale frame only has one channel
}

TEST_F(PipelineTests, PipelineInternalInDayMode) {
    PipelineInternal internal(false, false);
    internal.initialize();

    cv::Mat processed = internal.processFrame(testFrame);
    EXPECT_FALSE(processed.empty());
    EXPECT_EQ(processed.channels(), 3);
}

TEST_F(PipelineTests, MotionDetectorInitialize) {
    PipelineInitialDetection detector(1);
    EXPECT_TRUE(detector.initialize());
}

TEST_F(PipelineTests, MotionDetectorWithMotion) {
    PipelineInitialDetection detector(1);
    detector.initialize();

    detector.detectMotion(testFrame);

    bool result = detector.detectMotion(motionFrame);
    EXPECT_TRUE(result);
}


TEST_F(PipelineTests, MotionDetectorThreshold) {
    PipelineInitialDetection detector(1);
    detector.initialize();
    
    detector.setThresholdForDetection(50.0);
    

    detector.detectMotion(testFrame);
    

    bool result = detector.detectMotion(motionFrame);
    EXPECT_FALSE(result);
    
    detector.setThresholdForDetection(1.0);

    result = detector.detectMotion(motionFrame);
    EXPECT_TRUE(result);
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}