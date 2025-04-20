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
    cudaStreamSynchronize(engineInterface.stream);
    cudaStreamDestroy(engineInterface.stream);
    cudaFree(runtimeConfiguration.gpuBuffers[0]);
    cudaFree(runtimeConfiguration.gpuBuffers[1]);
    
    if(engineInterface.ctx) engineInterface.ctx->destroy();
    if(engineInterface.engine) engineInterface.engine->destroy();
    if(engineInterface.runtime) engineInterface.runtime->destroy();
}

void MotionDetectionNode::initEngine() {
    engineInterface.runtime.reset(nvinfer1::createInferRuntime(gLogger));
    if(not engineInterface.runtime) throw std::runtime_error("[MotionDetectionNode- TensorRT] Runtime creation failed.");

    std::ifstream f(runtimeConfiguration.enginePath, std::ios::binary | std::ios::ate);
    if(not f) throw std::runtime_error("[MotionDetectionNode- TensorRT] Failed to open engine file: " + runtimeConfiguration.enginePath);

    size_t engineSize = f.tellg(); f.seekg(0);
    std::vector<char> engineData(engineSize);
    f.read(engineData.data(), engineSize);

    engineInterface.engine.reset(engineInterface.runtime->deserializeCudaEngine(engineData.data(), engineSize, nullptr));
    if(not engineInterface.engine) throw std::runtime_error("[MotionDetectionNode- TensorRT] Engine deserialisation failed.");

    engineInterface.ctx.reset(engineInterface.engine->createExecutionContext());
    if(not engineInterface.ctx) throw std::runtime_error("[MotionDetectionNode- TensorRT] Execution context creation failed.");

    const int inIdx = engineInterface.engine->getBindingIndex("images");
    const int outIdx = engineInterface.engine->getBindingIndex("output0");

    auto outDims = engineInterface.engine->getBindingDimensions(outIdx);
    const size_t inBytes = 1 * 3 * runtimeConfiguration.inputHeight * runtimeConfiguration.inputWidth * sizeof(float);

    runtimeConfiguration.outputSize = 1;
    for (int i = 0; i < outDims.nbDims; ++i) {
        runtimeConfiguration.outputSize *= outDims.d[i];
    }
    runtimeConfiguration.outputSize *= sizeof(float);

    cudaMalloc(&runtimeConfiguration.gpuBuffers[inIdx], inBytes);
    cudaMalloc(&runtimeConfiguration.gpuBuffers[outIdx], runtimeConfiguration.outputSize);
    cudaStreamCreate(&engineInterface.stream);
}

void MotionDetectionNode::imageCb(const sensor_msgs::ImageConstPtr& msg) {
    cv_bridge::CvImageConstPtr cvPtr;
    try {
        cvPtr = cv_bridge::toCvShare(msg, msg->encoding);
    } catch (const cv_bridge::Exception& e) {
        ROS_ERROR("%s", e.what());
        return;
    }

    cv::Mat hostInput = preProcess(cvPtr->image);
    const int inIdx = engineInterface.engine->getBindingIndex("images");
    const int outIdx = engineInterface.engine->getBindingIndex("output0");
    cudaMemcpyAsync(runtimeConfiguration.gpuBuffers[inIdx], hostInput.data, hostInput.total() * sizeof(float), cudaMemcpyHostToDevice, engineInterface.stream);

    engineInterface.ctx->enqueue(1, runtimeConfiguration.gpuBuffers, engineInterface.stream, nullptr);

    std::vector<float> out(runtimeConfiguration.outputSize / sizeof(float));
    cudaMemcpyAsync(out.data(), runtimeConfiguration.gpuBuffers[outIdx], runtimeConfiguration.outputSize, cudaMemcpyDeviceToHost, engineInterface.stream);
    cudaStreamSynchronize(engineInterface.stream);

    auto detections = postProcess(out.data());
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

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(runtimeConfiguration.inputWidth, runtimeConfiguration.inputHeight));
    resized.convertTo(resized, CV_32FC3, 1.f/255.f);

    std::vector<cv::Mat> splitCh;
    cv::split(resized, splitCh);
    cv::Mat blob(1, 3*runtimeConfiguration.inputHeight*runtimeConfiguration.inputWidth, CV_32F);
    float *dst = blob.ptr<float>();

    const size_t chArea = runtimeConfiguration.inputHeight * runtimeConfiguration.inputWidth;
    for (int c = 0; c < 3; ++c) {
        memcpy(dst + c * chArea, splitCh[c].data, chArea * sizeof(float));
    }
    return blob;
}


std::vector<vision_msgs::Detection2D> MotionDetectionNode::postProcess(const float* out) {
    std::vector<vision_msgs::Detection2D> dets;
    const int stride = 5;
    const int nBoxes = static_cast<int>(runtimeConfiguration.outputSize / (stride*sizeof(float)));
    const float nmsTh = 0.5F;

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    for(int i = 0; i < nBoxes; ++i) {
        float conf = out[i*stride+4];
        if(conf < runtimeConfiguration.confidenceThreshold) continue;

        float cx = out[i * stride + 0] * runtimeConfiguration.inputWidth;
        float cy = out[i * stride + 1] * runtimeConfiguration.inputHeight;
        float w  = out[i * stride + 2] * runtimeConfiguration.inputWidth;
        float h  = out[i * stride + 3] * runtimeConfiguration.inputHeight;

        boxes.emplace_back(
            static_cast<int>(cx - w * 0.5f),
            static_cast<int>(cy - h * 0.5f),
            static_cast<int>(w),
            static_cast<int>(h)
        );
        scores.push_back(conf);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, runtimeConfiguration.confidenceThreshold, nmsTh, keep);

    for(int idx : keep) {
        vision_msgs::Detection2D det;
        det.bbox.center.x = boxes[idx].x + boxes[idx].width / 2.0F;
        det.bbox.center.y = boxes[idx].y + boxes[idx].height / 2.0F;
        det.bbox.size_x = boxes[idx].width;
        det.bbox.size_y = boxes[idx].height;

        vision_msgs::ObjectHypothesisWithPose hyp;
        hyp.id = 0;
        hyp.score = scores[idx];
        det.results.push_back(hyp);
        dets.push_back(det);
    }

    return dets;
}

void MotionDetectionNode::publishForVisualization(std::vector<vision_msgs::Detection2D> &detectionPoints, cv::Mat viz, const sensor_msgs::ImageConstPtr& msg) {
    for(const auto& d : detectionPoints) {
        float x = d.bbox.center.x;
        float y = d.bbox.center.y;
        float w = d.bbox.size_x;
        float h = d.bbox.size_y;
        cv::rectangle(viz,
            cv::Point(int(x - w / 2), int(y - h / 2)),
            cv::Point(int(x + w / 2), int(y + h / 2)),
            cv::Scalar(0, 255, 0), 2
        );
    }

    runtimeDebugConfiguration.vizImg.image = viz;
    runtimeDebugConfiguration.vizImg.encoding = msg->encoding;
    runtimeDebugConfiguration.vizImg.header = msg->header;
    rosInterface.pub_vizDebug.publish(runtimeDebugConfiguration.vizImg.toImageMsg());
}
