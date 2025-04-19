#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "../src/nodes/pipelineNode/components/pipelineInternal.hpp"

class PipelineInternalTests : public ::testing::Test {
    protected:
        cv::Mat bgrFrame;
        cv::Mat grayFrame;
    
        void SetUp() override {
            /* 64x64 BGR frame (blue)*/
            bgrFrame = cv::Mat(64, 64, CV_8UC3, cv::Scalar(255, 0, 0));
            /* 64x64 Grayscale frame (mid-gray) */
            grayFrame = cv::Mat(64, 64, CV_8UC1, cv::Scalar(128));
        }
};

TEST_F(PipelineInternalTests, ConvertFormatNightMode) {
    pipeline::PipelineInternal processor(30, true); // Night mode = true (expects GRAY)

    /* Receives a BGR frame as input, which it should convert to grayscale as a output */
    cv::Mat outputGrayscale = processor.convertFormat(bgrFrame);
    ASSERT_FALSE(outputGrayscale.empty());
    ASSERT_EQ(outputGrayscale.channels(), 1);
    ASSERT_EQ(outputGrayscale.type(), CV_8UC1);
    ASSERT_EQ(outputGrayscale.rows, bgrFrame.rows);
    ASSERT_EQ(outputGrayscale.cols, bgrFrame.cols);

    /* Grayscale should not be reconverted */
    cv::Mat outputGrayscale_2 = processor.convertFormat(grayFrame);
    ASSERT_FALSE(outputGrayscale_2.empty());
    ASSERT_EQ(outputGrayscale_2.channels(), 1);
    ASSERT_EQ(outputGrayscale_2.type(), CV_8UC1);
    ASSERT_LT(cv::norm(grayFrame, outputGrayscale_2, cv::NORM_L1), 1e-5 * grayFrame.total());
}

TEST_F(PipelineInternalTests, ConvertFormatDayMode) {
    pipeline::PipelineInternal processor(30, false); // Night mode = false (expects BGR)

    cv::Mat outputBgr = processor.convertFormat(grayFrame);
    ASSERT_FALSE(outputBgr.empty());
    ASSERT_EQ(outputBgr .channels(), 3);
    ASSERT_EQ(outputBgr .type(), CV_8UC3);
    ASSERT_EQ(outputBgr .rows, grayFrame.rows);
    ASSERT_EQ(outputBgr .cols, grayFrame.cols);

    cv::Mat outputBgr_2 = processor.convertFormat(bgrFrame);
    ASSERT_FALSE(outputBgr_2.empty());
    ASSERT_EQ(outputBgr_2.channels(), 3);
    ASSERT_EQ(outputBgr_2.type(), CV_8UC3);
    ASSERT_LT(cv::norm(bgrFrame, outputBgr_2, cv::NORM_L1), 1e-5 * bgrFrame.total());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}