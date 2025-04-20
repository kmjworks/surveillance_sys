#include "motionDetectionNode_trt.hpp"


class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            if(severity == Severity::kINFO) return;
            ROS_INFO("[MotionDetectionNode - TensorRT] %s", msg);
        }
};

static Logger gLoggerInstance;
nvinfer1::ILogger& gLogger = gLoggerInstance;

MotionDetectionNode::MotionDetectionNode(ros::NodeHandle& nh, ros::NodeHandle& pnh) : nh(nh), private_nh(pnh), imageTransport(nh) {

    pnh.param("engine_file", runtimeConfiguration.enginePath, std::string("models/yolov11_fp16.engine"));
    pnh.param("confidence_threshold", runtimeConfiguration.confidenceThreshold, 0.6F);
    pnh.param("enable_viz", runtimeDebugConfiguration.enableViz, true);
    pnh.param("input_w", runtimeConfiguration.inputWidth, 640);
    pnh.param("input_h", runtimeConfiguration.inputHeight, 640);

    initEngine();

    rosInterface.sub_imageSrc = imageTransport.subscribe("pipeline/runtime_potentialMotionEvents", 1, &MotionDetectionNode::imageCb, this, image_transport::TransportHints("raw"));
    rosInterface.pub_detectedMotion = nh.advertise<vision_msgs::Detection2DArray>("yolo/runtime_detections", 10);
    if(runtimeDebugConfiguration.enableViz) rosInterface.pub_vizDebug = nh.advertise<sensor_msgs::Image>("yolo/runtime_detectionVisualizationDebug", 1);

    ROS_INFO("[MotionDetectionNode- TensorRT] ready - engine: %s", runtimeConfiguration.enginePath.c_str());
}

MotionDetectionNode::~MotionDetectionNode() {
    if(engineInterface.stream) {
        cudaStreamSynchronize(engineInterface.stream);
        cudaStreamDestroy(engineInterface.stream); engineInterface.stream = nullptr;
    }

    if(runtimeConfiguration.gpuBuffers[0]) {
        cudaFree(runtimeConfiguration.gpuBuffers[0]);
        runtimeConfiguration.gpuBuffers[0] = nullptr;
    }

    if(runtimeConfiguration.gpuBuffers[1]) {
        cudaFree(runtimeConfiguration.gpuBuffers[1]);
        runtimeConfiguration.gpuBuffers[1] =  nullptr;
    }
}

void MotionDetectionNode::initEngine() {
    engineInterface.runtime.reset(nvinfer1::createInferRuntime(gLogger));
    if(not engineInterface.runtime) throw std::runtime_error("[MotionDetectionNode- TensorRT] Runtime creation failed.");

    std::ifstream f(runtimeConfiguration.enginePath, std::ios::binary | std::ios::ate);
    if(not f) throw std::runtime_error("[MotionDetectionNode- TensorRT] Failed to open engine file: " + runtimeConfiguration.enginePath);

    size_t engineSize = f.tellg(); f.seekg(0, std::ios::beg);
    std::vector<char> engineData(engineSize);
    f.read(engineData.data(), engineSize); f.close();

    engineInterface.engine.reset(engineInterface.runtime->deserializeCudaEngine(engineData.data(), engineSize, nullptr));
    if(not engineInterface.engine) throw std::runtime_error("[MotionDetectionNode- TensorRT] Engine deserialisation failed.");
    if (engineInterface.engine->getNbBindings() != 2) {
        ROS_WARN("[MotionDetectionNode - TensorRT] Expected 2 bindings (input 'images', output 'output0'), but found %d.", engineInterface.engine->getNbBindings());
   }

    engineInterface.ctx.reset(engineInterface.engine->createExecutionContext());
    if(not engineInterface.ctx) throw std::runtime_error("[MotionDetectionNode- TensorRT] Execution context creation failed.");

    const int inIdx = engineInterface.engine->getBindingIndex("images");
    const int outIdx = engineInterface.engine->getBindingIndex("output0");

    if (inIdx < 0) throw std::runtime_error("[MotionDetectionNode - TensorRT] Input binding 'images' not found.");
    if (outIdx < 0) throw std::runtime_error("[MotionDetectionNode - TensorRT] Output binding 'output0' not found.");

    const size_t inBytes = 1 * 3 * runtimeConfiguration.inputHeight * runtimeConfiguration.inputWidth * sizeof(float);

    nvinfer1::Dims outDims = engineInterface.engine->getBindingDimensions(outIdx);
    runtimeConfiguration.outputSize = 1; 
    for (int i = 0; i < outDims.nbDims; ++i) {
        runtimeConfiguration.outputSize *= outDims.d[i];
    }
    runtimeConfiguration.outputSize *= sizeof(float);

    cudaError_t cudaErr;

    cudaErr = cudaMalloc(&runtimeConfiguration.gpuBuffers[inIdx], inBytes);
    if(cudaErr != cudaSuccess) throw std::runtime_error("[MotionDetectionNode - TensorRT] CUDA Malloc failed for input buffer: " + std::string(cudaGetErrorString(cudaErr)));
    
    cudaErr = cudaMalloc(&runtimeConfiguration.gpuBuffers[outIdx], runtimeConfiguration.outputSize);
    if (cudaErr != cudaSuccess) throw std::runtime_error("[MotionDetectionNode - TensorRT] CUDA Malloc failed for output buffer: " + std::string(cudaGetErrorString(cudaErr)));
    
    cudaErr = cudaStreamCreate(&engineInterface.stream);
    if (cudaErr != cudaSuccess) throw std::runtime_error("[MotionDetectionNode - TensorRT] CUDA Stream creation failed: " + std::string(cudaGetErrorString(cudaErr)));
    ROS_INFO("[MotionDetectionNode - TensorRT] Engine initialization complete.");
}

void MotionDetectionNode::imageCb(const sensor_msgs::ImageConstPtr& msg) {
    if (!engineInterface.ctx) {
        ROS_WARN_THROTTLE(5.0, "[MotionDetectionNode] Engine not ready, skipping inference.");
        return;
    }

    cv_bridge::CvImageConstPtr cvPtr;

    try {
        cvPtr = cv_bridge::toCvShare(msg, msg->encoding);
    } catch (const cv_bridge::Exception& e) {
        ROS_ERROR("%s", e.what());
        return;
    }

    int originalWidth = cvPtr->image.cols;
    int originalHeight = cvPtr->image.rows;
    runtimeDebugConfiguration.scaleX = static_cast<float>(originalWidth) / runtimeConfiguration.inputWidth;
    runtimeDebugConfiguration.scaleY = static_cast<float>(originalHeight) / runtimeConfiguration.inputHeight;

    cv::Mat hostInput = preProcess(cvPtr->image);
    const int inIdx = engineInterface.engine->getBindingIndex("images");
    const int outIdx = engineInterface.engine->getBindingIndex("output0");
    cudaError_t cudaErr = cudaMemcpyAsync(runtimeConfiguration.gpuBuffers[inIdx], hostInput.data, hostInput.total() * hostInput.elemSize(), cudaMemcpyHostToDevice, engineInterface.stream);
    if(cudaErr != cudaSuccess) {
        ROS_ERROR("[MotionDetectionNode] CUDA Memcpy H2D failed: %s", cudaGetErrorString(cudaErr));
        return; 
    }

    bool status = engineInterface.ctx->enqueueV2(runtimeConfiguration.gpuBuffers, engineInterface.stream, nullptr);
    if (!status) {
        ROS_ERROR("[MotionDetectionNode] TensorRT inference enqueue failed.");
        cudaStreamSynchronize(engineInterface.stream);
        return;
    }
    std::vector<float> out(runtimeConfiguration.outputSize / sizeof(float));
    cudaErr = cudaMemcpyAsync(out.data(), runtimeConfiguration.gpuBuffers[outIdx], runtimeConfiguration.outputSize, cudaMemcpyDeviceToHost, engineInterface.stream);
    if(cudaErr != cudaSuccess) {
        ROS_ERROR("[MotionDetectionNode] CUDA Memcpy D2H failed: %s", cudaGetErrorString(cudaErr));
         cudaStreamSynchronize(engineInterface.stream);
        return;
    }
    cudaStreamSynchronize(engineInterface.stream);

    auto detections = postProcess(out.data(), msg->header.stamp, msg->header.frame_id);
    if(!detections.empty()) {
        vision_msgs::Detection2DArray detectionArray;
        detectionArray.header = msg->header;
        detectionArray.detections = detections;
        rosInterface.pub_detectedMotion.publish(detectionArray);
    }

    if(runtimeDebugConfiguration.enableViz) publishForVisualization(detections, cvPtr->image.clone(), msg);

}

cv::Mat MotionDetectionNode::preProcess(const cv::Mat& img) {
    cv::Mat bgr;
    if(img.channels() == 1) {
        cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
    } else {
        bgr = img;
    }

    cv::Mat resized_image;
    cv::resize(bgr, resized_image, cv::Size(runtimeConfiguration.inputWidth, runtimeConfiguration.inputHeight),0, 0, cv::INTER_LINEAR); 

    cv::Mat float_image;
    resized_image.convertTo(float_image, CV_32F, 1.0 / 255.0);

    cv::Mat blob = cv::dnn::blobFromImage(float_image, 1.0, cv::Size(), cv::Scalar(), false, false);
    return blob;
}

std::vector<vision_msgs::Detection2D> MotionDetectionNode::postProcess(
    const float* outputData, const ros::Time& timestamp, const std::string& frame_id) {
    std::vector<vision_msgs::Detection2D> finalDetections;
    nvinfer1::Dims outDims = engineInterface.engine->getBindingDimensions(
        engineInterface.engine->getBindingIndex("output0"));

    if (outDims.nbDims != 3 || outDims.d[0] != 1) {
        ROS_ERROR(
            "[MotionDetectionNode::postProcess] Unexpected output dimensions. Expected [1, "
            "num_boxes, 5+num_classes], got %d dims.",
            outDims.nbDims);
        return finalDetections;
    }

    const int numBoxes = outDims.d[1];
    const int numElementsPerBox = outDims.d[2];
    const int numClasses = numElementsPerBox - 5;

    if (numElementsPerBox < 5) {
        ROS_ERROR(
            "[MotionDetectionNode::postProcess] Unexpected number of elements per box: %d. "
            "Expected >= 5.",
            numElementsPerBox);
        return finalDetections;
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classIDs;

    boxes.reserve(numBoxes);
    scores.reserve(numBoxes);
    classIDs.reserve(numBoxes);

    for (int i = 0; i < numBoxes; ++i) {
        const float* boxData = outputData + i * numElementsPerBox;

        float cx = boxData[0];
        float cy = boxData[1];
        float w = boxData[2];
        float h = boxData[3];
        float conf = boxData[4];

        if (conf < runtimeConfiguration.confidenceThreshold) {
            continue;
        }

        int bestClassIdx = 0;
        float bestScore = conf;

        if (numClasses >= 1) {
            float maxClsProbability = 0.0f;
            int maxClsIdx = 0;

            for (int j = 0; j < numClasses; ++j) {
                float classProbability = boxData[5 + j];
                if (classProbability > maxClsProbability) {
                    maxClsProbability = classProbability;
                    maxClsIdx = j;
                }
            }
            float finalConfidenceScore = conf * maxClsProbability;
            if (finalConfidenceScore < runtimeConfiguration.confidenceThreshold) {
                continue;
            }
            bestScore = finalConfidenceScore;
            bestClassIdx = maxClsIdx;
        }

        float cx_pix = cx * runtimeConfiguration.inputWidth;
        float cy_pix = cy * runtimeConfiguration.inputHeight;
        float w_pix = w * runtimeConfiguration.inputWidth;
        float h_pix = h* runtimeConfiguration.inputHeight;

        float x = cx_pix - w_pix / 2.0f;
        float y = cy_pix - h_pix / 2.0f;

        boxes.emplace_back(static_cast<int>(x), static_cast<int>(y), static_cast<int>(w_pix),
                           static_cast<int>(h_pix));
        scores.push_back(bestScore);
        classIDs.push_back(bestClassIdx);
    }

    if (boxes.empty()) {
        return finalDetections;
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, runtimeConfiguration.confidenceThreshold,
                      runtimeConfiguration.nmsThreshold, keep);

    finalDetections.reserve(keep.size());
    for (int idx : keep) {
        vision_msgs::Detection2D det;
        det.header.stamp = timestamp;
        det.header.frame_id = frame_id;

        float x = static_cast<float>(boxes[idx].x);
        float y = static_cast<float>(boxes[idx].y);
        float w = static_cast<float>(boxes[idx].width);
        float h = static_cast<float>(boxes[idx].height);

        det.bbox.center.x = x + w / 2.0f;
        det.bbox.center.y = y + h / 2.0f;
        det.bbox.size_x = w;
        det.bbox.size_y = h;

        vision_msgs::ObjectHypothesisWithPose hyp;
        hyp.id = classIDs[idx];
        hyp.score = scores[idx];
        det.results.push_back(hyp);

        finalDetections.push_back(det);
    }
    return finalDetections;
}

void MotionDetectionNode::publishForVisualization(
    std::vector<vision_msgs::Detection2D>& detectionPoints, cv::Mat viz,
    const sensor_msgs::ImageConstPtr& msg) {
    if (viz.empty()) {
        ROS_WARN("[MotionDetectionNode::publishForVisualization] Cannot visualize on empty frame.");
        return;
    }

    for (const auto& d : detectionPoints) {
        float box_cx = d.bbox.center.x;
        float box_cy = d.bbox.center.y;
        float box_w = d.bbox.size_x;
        float box_h = d.bbox.size_y;

        int xmin = static_cast<int>((box_cx - box_w / 2.0f) * runtimeDebugConfiguration.scaleX);
        int ymin = static_cast<int>((box_cy - box_h / 2.0f) * runtimeDebugConfiguration.scaleY);
        int xmax = static_cast<int>((box_cx + box_w / 2.0f) * runtimeDebugConfiguration.scaleX);
        int ymax = static_cast<int>((box_cy + box_h / 2.0f) * runtimeDebugConfiguration.scaleY);

        xmin = std::max(0, std::min(xmin, viz.cols - 1));
        ymin = std::max(0, std::min(ymin, viz.rows - 1));
        xmax = std::max(0, std::min(xmax, viz.cols - 1));
        ymax = std::max(0, std::min(ymax, viz.rows - 1));

        cv::rectangle(viz, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(0, 255, 0), 2);
        if (!d.results.empty()) {
            float score = d.results[0].score;
            std::ostringstream labelStream;
            labelStream << "Human: " << std::fixed << std::setprecision(2) << score;
            std::string label = labelStream.str();

            int baseline = 0;
            cv::Size labelSize =
                cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
            baseline += 1;

            int top = std::max(ymin - labelSize.height - baseline, 0);
            int left = std::max(xmin, 0);
            int right = std::min(left + labelSize.width, viz.cols);
            int bottom = std::min(top + labelSize.height + baseline, viz.rows);

            cv::rectangle(viz, cv::Point(left, top), cv::Point(right, bottom),
                          cv::Scalar(0, 255, 0), cv::FILLED);

            cv::putText(viz, label, cv::Point(left, bottom - baseline), cv::FONT_HERSHEY_SIMPLEX,
                        0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }
    }
    try {
        runtimeDebugConfiguration.vizImg.image = viz;
        runtimeDebugConfiguration.vizImg.encoding = msg->encoding;
        runtimeDebugConfiguration.vizImg.header = msg->header;
        rosInterface.pub_vizDebug.publish(runtimeDebugConfiguration.vizImg.toImageMsg());
    } catch (const cv_bridge::Exception& e) {
        ROS_ERROR("[MotionDetectionNode::publishForVisualization] cv_bridge exception: %s",
                  e.what());
    }
}
