#include "motionDetectionNode_trt.hpp"

MotionDetectionNode::MotionDetectionNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
    : imageTransport(nh) {

    pnh.param("engine_file", enginePath, std::string("models/yolov11_fp16.engine"));
    pnh.param("confidence_threshold", confidenceThreshold, 0.4F);
    pnh.param("enable_viz", enableViz, true);
    pnh.param("input_w", inputWidth, 640);
    pnh.param("input_h", inputHeight, 640);

    runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime)
        throw std::runtime_error("Failed to create TensorRT runtime");

    std::ifstream f(enginePath, std::ios::binary | std::ios::ate);
    if (!f.good())
        throw std::runtime_error("Cannot open engine file: " + enginePath);
    size_t engSize = f.tellg();
    f.seekg(0);
    std::vector<char> engData(engSize);
    f.read(engData.data(), engSize);

    engine = runtime->deserializeCudaEngine(engData.data(), engSize, nullptr);
    if (!engine)
        throw std::runtime_error("Engine deserialisation failed");
    ctx = engine->createExecutionContext();
    if (!ctx)
        throw std::runtime_error("Failed to create execution context");

    const int inIdx = engine->getBindingIndex("images");
    const int outIdx = engine->getBindingIndex("output0");
    auto outDims = engine->getBindingDimensions(outIdx);

    const size_t inBytes = 1 * 3 * inputHeight * inputWidth * sizeof(float);
    outputSize = 1;
    for (int i = 0; i < outDims.nbDims; ++i)
        outputSize *= outDims.d[i];
    outputSize *= sizeof(float);

    cudaMalloc(&gpuBuffers[inIdx], inBytes);
    cudaMalloc(&gpuBuffers[outIdx], outputSize);
    cudaStreamCreate(&stream);

    sub_imageSrc = imageTransport.subscribe("pipeline/runtime_potentialMotionEvents", 1,
                                            &MotionDetectionNode::imageCb, this,
                                            image_transport::TransportHints("raw"));

    pub_detectedMotion = nh.advertise<vision_msgs::Detection2DArray>("yolo/runtime_detections", 10);

    if (enableViz)
        pub_vizDebug =
            nh.advertise<sensor_msgs::Image>("yolo/runtime_detectionVisualizationDebug", 1);

    ROS_INFO("[MotionDetectionNode-TRT] ready - engine: %s", enginePath.c_str());
}

MotionDetectionNode::~MotionDetectionNode() {
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(gpuBuffers[0]);
    cudaFree(gpuBuffers[1]);

    if (ctx)
        ctx->destroy();
    if (engine)
        engine->destroy();
    if (runtime)
        runtime->destroy();
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

    cudaMemcpyAsync(gpuBuffers[0], hostInput.data, hostInput.total() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    ctx->enqueue(1, gpuBuffers, stream, nullptr);
    std::vector<float> out(outputSize / sizeof(float));
    cudaMemcpyAsync(out.data(), gpuBuffers[1], outputSize, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    vision_msgs::Detection2DArray arr;
    arr.header = msg->header;
    arr.detections = postProcess(out.data());
    if (!arr.detections.empty())
        pub_detectedMotion.publish(arr);

    if (enableViz && pub_vizDebug.getNumSubscribers() > 0) {
        cv::Mat viz = cvPtr->image.clone();
        for (const auto& d : arr.detections) {
            float x = d.bbox.center.x, y = d.bbox.center.y;
            float w = d.bbox.size_x, h = d.bbox.size_y;
            cv::rectangle(viz, {int(x - w / 2), int(y - h / 2)}, {int(x + w / 2), int(y + h / 2)},
                          cv::Scalar(0, 255, 0), 2);
        }
        vizImg.image = viz;
        vizImg.encoding = msg->encoding;
        vizImg.header = msg->header;
        pub_vizDebug.publish(vizImg.toImageMsg());
    }
}

cv::Mat MotionDetectionNode::preProcess(const cv::Mat& img) {
    cv::Mat bgr;
    if (img.channels() == 1)
        cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
    else
        bgr = img;

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(inputWidth, inputHeight));
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> splitCh;
    cv::split(resized, splitCh);  

    cv::Mat blob(1, 3 * inputHeight * inputWidth, CV_32F);
    float* dst = blob.ptr<float>();
    size_t chArea = inputHeight * inputWidth;
    for (int c = 0; c < 3; ++c) {
        memcpy(dst + c * chArea, splitCh[c].data, chArea * sizeof(float));
    }
    return blob; 
}


std::vector<vision_msgs::Detection2D> MotionDetectionNode::postProcess(const float* out) {
    std::vector<vision_msgs::Detection2D> dets;

    const int stride = 5; 
    const int nBoxes = static_cast<int>(outputSize / (stride * sizeof(float)));
    const float nmsTh = 0.45F;

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    for (int i = 0; i < nBoxes; ++i) {
        float conf = out[i * stride + 4];
        if (conf < confidenceThreshold)
            continue;

        float cx = out[i * stride + 0] * inputWidth;
        float cy = out[i * stride + 1] * inputHeight;
        float w = out[i * stride + 2] * inputWidth;
        float h = out[i * stride + 3] * inputHeight;
        boxes.emplace_back(static_cast<int>(cx-w*0.5F), static_cast<int>(cy-h*0.5F), static_cast<int>(w), static_cast<int>(h));
        scores.push_back(conf);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, confidenceThreshold, nmsTh, keep);

    for (int idx : keep) {
        vision_msgs::Detection2D det;
        det.bbox.center.x = boxes[idx].x + boxes[idx].width / 2.0F;
        det.bbox.center.y = boxes[idx].y + boxes[idx].height / 2.0F;
        det.bbox.size_x = boxes[idx].width;
        det.bbox.size_y = boxes[idx].height;

        vision_msgs::ObjectHypothesisWithPose hyp;
        hyp.id = 0;
        hyp.score = scores[idx];
        det.results.push_back(hyp);

        dets.emplace_back(std::move(det));
    }
    return dets;
}
