#include <opencv2/opencv.hpp>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <functional>

struct DetectionBox;

class VisualizationHandler {
public:
    virtual void processTrackingVisualization(const cv::Mat& image, const std::vector<DetectionBox>& boundingBoxes, const ros::Time& timestamp) = 0;
    virtual ~VisualizationHandler() = default;
};

class AsyncProcessingPipeline {
public:
    AsyncProcessingPipeline(VisualizationHandler* handler) : handlerInternal(handler) {
        if (!handler) {
            throw std::runtime_error("Visualization handler cannot be null");
        }
        
        processingThread = std::thread([this]() {
            while (running) {
                std::tuple<cv::Mat, std::vector<DetectionBox>, ros::Time> work;
                bool hasWork = false;
                {
                    std::unique_lock<std::mutex> lock(queueMutex);
                    queueCondition.wait(lock, [this]() { 
                        return !workQueue.empty() || !running; 
                    });
                    
                    if (!running) break;
                    work = workQueue.front();
                    workQueue.pop();
                    hasWork = true;
                }
                
                if (hasWork) {
                    handlerInternal->processTrackingVisualization(
                        std::get<0>(work), 
                        std::get<1>(work), 
                        std::get<2>(work)
                    );
                }
            }
        });
    }
    
    void enqueueWork(const cv::Mat& image, const std::vector<DetectionBox>& boxes, const ros::Time& timestamp) {
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            if (workQueue.size() > 2) {
                std::queue<std::tuple<cv::Mat, std::vector<DetectionBox>, ros::Time>> emptyQueue;
                std::swap(workQueue, emptyQueue);
            }
            workQueue.push(std::make_tuple(image.clone(), boxes, timestamp));
        }
        queueCondition.notify_one();
    }
    
    ~AsyncProcessingPipeline() {
        running = false;
        queueCondition.notify_all();
        if (processingThread.joinable()) {
            processingThread.join();
        }
    }
private:
    std::queue<std::tuple<cv::Mat, std::vector<DetectionBox>, ros::Time>> workQueue;
    std::mutex queueMutex;
    std::condition_variable queueCondition;
    std::atomic<bool> running{true};
    std::thread processingThread;
    VisualizationHandler* handlerInternal;
};