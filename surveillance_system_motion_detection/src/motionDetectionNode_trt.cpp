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

    cv::Mat blob = cv::dnn::blobFromImage(float_image, 1.0, cv::Size(), cv::Scalar(), true, false);
    return blob;
}

std::vector<vision_msgs::Detection2D> MotionDetectionNode::postProcess(
    const float* outputData, 
    const ros::Time& timestamp, 
    const std::string& frame_id
) {
    std::vector<vision_msgs::Detection2D> finalDetections;
    const int outIdx = engineInterface.engine->getBindingIndex("output0");
    nvinfer1::Dims outDims = engineInterface.engine->getBindingDimensions(outIdx);

    if (outDims.nbDims != 3 || outDims.d[0] != 1) {
        ROS_ERROR("[postProcess] Unexpected output dims. Expected [1,7,8400], got [%d,%d,%d].",
                  outDims.d[0], outDims.d[1], outDims.d[2]);
        return finalDetections;
    }
    const int numElementsPerBox = outDims.d[1];  // 7
    const int numBoxes          = outDims.d[2];  // 8400

    if (numElementsPerBox != (4 + runtimeConfiguration.numClasses)) {
        ROS_ERROR("[postProcess] Mismatch: Model's second dimension = %d, but we expect %d = (4 + %d classes).",
                  numElementsPerBox, 4 + runtimeConfiguration.numClasses, runtimeConfiguration.numClasses);
        return finalDetections;
    }

    std::vector<cv::Rect> boxesNet;
    std::vector<float>    scores;
    std::vector<int>      classIDs;
    boxesNet.reserve(numBoxes);
    scores.reserve(numBoxes);
    classIDs.reserve(numBoxes);

    const float* cxPtr = outputData + 0 * numBoxes;
    const float* cyPtr = outputData + 1 * numBoxes;
    const float* wPtr  = outputData + 2 * numBoxes;
    const float* hPtr  = outputData + 3 * numBoxes;

    const float* classPtr = outputData + 4 * numBoxes;
    const int numClasses   = runtimeConfiguration.numClasses;

    const float confThresh   = runtimeConfiguration.confidenceThreshold;
    const float netInputWf   = static_cast<float>(runtimeConfiguration.inputWidth);
    const float netInputHf   = static_cast<float>(runtimeConfiguration.inputHeight);

    for (int i = 0; i < numBoxes; ++i) {
        float maxClsScore = -1.0f;
        int   bestClsIdx  = -1;
        for (int c = 0; c < numClasses; ++c) {
            float clsScore = (classPtr + c * numBoxes)[i];
            if (clsScore > maxClsScore) {
                maxClsScore = clsScore;
                bestClsIdx  = c;
            }
        }

        if (maxClsScore < confThresh) {
            continue;
        }

        float cx = cxPtr[i];
        float cy = cyPtr[i];
        float w  = wPtr[i];
        float h  = hPtr[i];

        float x_net = (cx - w * 0.5f);
        float y_net = (cy - h * 0.5f);
        float w_net = w;
        float h_net = h;

        x_net = std::max(0.0f, std::min(x_net, netInputWf - 1.0f));
        y_net = std::max(0.0f, std::min(y_net, netInputHf - 1.0f));
        w_net = std::max(1.0f, std::min(w_net, netInputWf - x_net));
        h_net = std::max(1.0f, std::min(h_net, netInputHf - y_net));

        boxesNet.emplace_back(
            static_cast<int>(x_net),
            static_cast<int>(y_net),
            static_cast<int>(w_net),
            static_cast<int>(h_net)
        );
        scores.push_back(maxClsScore);
        classIDs.push_back(bestClsIdx);
    }

    if (boxesNet.empty()) {
        return finalDetections;
    }

    float nmsThreshold = 0.5f;  
    std::vector<int> keepIndices;
    cv::dnn::NMSBoxes(boxesNet, scores, confThresh, nmsThreshold, keepIndices);

    finalDetections.reserve(keepIndices.size());
    for (int idx : keepIndices) {
        vision_msgs::Detection2D detection;
        detection.header.stamp    = timestamp;
        detection.header.frame_id = frame_id;

        const cv::Rect& box = boxesNet[idx];
        detection.bbox.center.x = box.x + box.width  * 0.5;
        detection.bbox.center.y = box.y + box.height * 0.5;
        detection.bbox.size_x   = box.width;
        detection.bbox.size_y   = box.height;

        vision_msgs::ObjectHypothesisWithPose hyp;
        hyp.id    = classIDs[idx];
        hyp.score = scores[idx];
        detection.results.push_back(hyp);

        finalDetections.push_back(detection);
    }

    return finalDetections;
}

void MotionDetectionNode::publishForVisualization(
    std::vector<vision_msgs::Detection2D> &detectionPoints, cv::Mat viz,
    const sensor_msgs::ImageConstPtr& msg) {
    if (viz.empty()) {
        ROS_WARN("[publishForVisualization] Empty frame, skipping.");
        return;
    }

    const std::vector<std::string> classNames = {"person", "dog", "cat"};

    for (const auto& d : detectionPoints) {

        double boxCx = d.bbox.center.x;
        double boxCy = d.bbox.center.y;
        double boxWidth = d.bbox.size_x;
        double boxHeight = d.bbox.size_y;

        int xmin = static_cast<int>((boxCx - boxWidth / 2.0) * runtimeDebugConfiguration.scaleX);
        int ymin = static_cast<int>((boxCy - boxHeight / 2.0) * runtimeDebugConfiguration.scaleY);
        int xmax = static_cast<int>((boxCx + boxWidth / 2.0) * runtimeDebugConfiguration.scaleX);
        int ymax = static_cast<int>((boxCy + boxHeight / 2.0) * runtimeDebugConfiguration.scaleY);

        xmin = std::max(0, std::min(xmin, viz.cols - 1));
        ymin = std::max(0, std::min(ymin, viz.rows - 1));
        xmax = std::max(0, std::min(xmax, viz.cols - 1));
        ymax = std::max(0, std::min(ymax, viz.rows - 1));
        if (xmax <= xmin || ymax <= ymin) {
            continue;
        }

        cv::rectangle(viz, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(0, 255, 0), 2);

        if (!d.results.empty()) {
            const auto& hyp = d.results[0];
            int classID = hyp.id;
            float score = hyp.score;
            std::string label = "unknown";

            if (classID >= 0 && classID < static_cast<int>(classNames.size())) {
                label = classNames[classID];
            }

            std::ostringstream ss;
            ss << label << ": " << std::fixed << std::setprecision(2) << score;
            std::string labelStr = ss.str();

            int baseline = 0;
            cv::Size labelSize =
                cv::getTextSize(labelStr, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
            baseline += 1;

            int textBgTop = std::max(ymin - labelSize.height - baseline - 5, 0);
            int textBgLeft = xmin;
            int textBgRight = textBgLeft + labelSize.width;
            int textBgBottom = textBgTop + labelSize.height + baseline;

            cv::rectangle(
                viz, cv::Point(textBgLeft, textBgTop),
                cv::Point(std::min(textBgRight, viz.cols), std::min(textBgBottom, viz.rows)),
                cv::Scalar(0, 255, 0), cv::FILLED);

            int textOrgX = textBgLeft;
            int textOrgY = textBgBottom - baseline / 2;
            textOrgY = std::max(labelSize.height, textOrgY);

            cv::putText(viz, labelStr, cv::Point(textOrgX, textOrgY), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }
    }

    try {
        cv_bridge::CvImage outputCvImage;
        outputCvImage.encoding = sensor_msgs::image_encodings::BGR8;
        outputCvImage.header = msg->header;
        outputCvImage.image = viz;
        rosInterface.pub_vizDebug.publish(outputCvImage.toImageMsg());
    } catch (const cv_bridge::Exception& e) {
        ROS_WARN("[publishForVisualization] cv_bridge publishing error: %s", e.what());
    }
}