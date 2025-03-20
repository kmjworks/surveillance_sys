#ifndef MOCK_FRAMES_HPP
#define MOCK_FRAMES_HPP

#include <opencv2/opencv.hpp>
#include <random>

class MockFrames {
private:
    int width_;
    int height_;
    cv::Mat background_;
    std::mt19937 rng_;

    cv::Point2f object_position_;
    cv::Point2f object_velocity_;
    float object_size_;

public:
    MockFrames(int width = 1920, int height = 1080) : width_(width), height_(height) {
        background_ = cv::Mat::zeros(height_, width_, CV_8UC3);

        std::random_device rd;
        rng_ = std::mt19937(rd());

        resetObject();
    }

    void resetObject() {
        std::uniform_real_distribution<float> pos_dist_x(0, width_);
        std::uniform_real_distribution<float> pos_dist_y(0, height_);

        object_position_ = cv::Point2f(pos_dist_x(rng_), pos_dist_y(rng_));

        std::uniform_real_distribution<float> vel_dist(-5.0f, 5.0f);
        object_velocity_ = cv::Point2f(vel_dist(rng_), vel_dist(rng_));

        std::uniform_real_distribution<float> size_dist(30.0f, 100.0f);
        object_size_ = size_dist(rng_);
    }

    cv::Mat generateFrame() {
        cv::Mat frame = background_.clone();

        object_position_ += object_velocity_;

        if (object_position_.x < 0 || object_position_.x > width_) {
            object_velocity_.x *= -1;
        }
        if (object_position_.y < 0 || object_position_.y > height_) {
            object_velocity_.y *= -1;
        }

        cv::circle(frame, object_position_, object_size_, cv::Scalar(0, 0, 2555), -1);

        return frame;
    }
};

#endif  // MOCK_FRAMES_HPP