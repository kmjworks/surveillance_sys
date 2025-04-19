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

TEST_F(PipelineInternalTests, PreprocessFrameGray) {
    pipeline::PipelineInternal processor(30, true); 

    cv::Mat nonUniformGrayscale = cv::Mat(64, 64, CV_8UC1);
    cv::randu(nonUniformGrayscale, cv::Scalar(50), cv::Scalar(150)); 

    cv::Mat processed = processor.preprocessFrame(nonUniformGrayscale);
    ASSERT_FALSE(processed.empty());
    ASSERT_EQ(processed.channels(), 1);
    ASSERT_EQ(processed.type(), CV_8UC1);

    /*
        An extremely trivial check - the mean might change but not drastically
        A better check would be to involve comparing the histograms before and after
    */
    // cv::Scalar meanBefore = cv::mean(nonUniformGrayscale);
    cv::Scalar meanAfter = cv::mean(processed);
    ASSERT_GT(meanAfter[0], 0);
}

TEST_F(PipelineInternalTests, PreprocessFrameBGR) {
    pipeline::PipelineInternal processor(30, false);

    cv::Mat nonUniformBgr = cv::Mat(64, 64, CV_8UC3);
    cv::randu(nonUniformBgr, cv::Scalar(0, 50, 100), cv::Scalar(100, 150, 200));

    cv::Mat processed = processor.preprocessFrame(nonUniformBgr);
    ASSERT_FALSE(processed.empty());
    ASSERT_EQ(processed.channels(), 3);
    ASSERT_EQ(processed.type(), CV_8UC3);

    cv::Mat grayBefore, grayAfter;
    cv::cvtColor(nonUniformBgr, grayBefore, cv::COLOR_BGR2GRAY);
    cv::cvtColor(processed, grayAfter, cv::COLOR_BGR2GRAY);
   // cv::Scalar meanBefore = cv::mean(grayBefore);
    cv::Scalar meanAfter = cv::mean(grayAfter);
    ASSERT_GT(meanAfter[0], 0);
}

TEST_F(PipelineInternalTests, ProcessFrameIntegration) {
    pipeline::PipelineInternal processor(30, false); 

    /* Process a gray frame - it should be converted to BGR and preprocessed */
    cv::Mat result = processor.processFrame(grayFrame);
    ASSERT_FALSE(result.empty());
    ASSERT_EQ(result.channels(), 3); 
    ASSERT_EQ(result.type(), CV_8UC3);

    cv::Mat emptyFrameAfterProcesing = processor.processFrame(cv::Mat());
    ASSERT_TRUE(emptyFrameAfterProcesing.empty());
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}