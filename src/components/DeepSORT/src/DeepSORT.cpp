#include <utility>

#include "../incl/DeepSORT.hpp"

DeepSort::DeepSort(std::string modelPath, int batchSize, int featureDim, int gpuID, nvinfer1::ILogger* gLogger) : 
enginePath(modelPath), batchSize(batchSize), featureDim(featureDim), gLogger(gLogger), gpuID(gpuID) {

    imgShape = cv::Size(64, 128);
    maxBudget = 100;
    maxCosineDist = 0.2;
    init();
}

DeepSort::~DeepSort() {}

void DeepSort::init() {
    try {
        objTracker = std::make_unique<Tracker>(maxCosineDist, maxBudget);
        featureExtractor = std::make_unique<FeatureTensor>(batchSize, imgShape, featureDim, gpuID, gLogger);
        featureExtractor->loadEngine(enginePath);

    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("[Tracker] DeepSort initialization failed: ") + e.what());
    }
}


void DeepSort::sort(cv::Mat& frame, std::vector<DetectionBox>& dets) {
    // preprocess Mat -> DETECTION
    model_internal::DETECTIONS detections;
    std::vector<CLSCONF> clsConf;
    
    for (DetectionBox i : dets) {
        tracking::DETECTBOX box(i.x1, i.y1, i.x2-i.x1, i.y2-i.y1);
        DETECTION_ROW d;
        d.tlwh = box;
        d.confidence = i.confidence;
        detections.push_back(d);
        clsConf.push_back(CLSCONF((int)i.classIdentifier, i.confidence));
    }
    result.clear();
    results.clear();

    if (!detections.empty()) {
        model_internal::DETECTIONSV2 detectionsv2 = make_pair(clsConf, detections);
        sort(frame, detectionsv2);
    }
    // postprocess DETECTION -> Mat
    dets.clear();
    for (auto r : result) {
        tracking::DETECTBOX i = r.second;
        DetectionBox b(i(0), i(1), i(2)+i(0), i(3)+i(1), 1.);
        b.trackIdentifier = (float)r.first;
        dets.push_back(b);
    }

    size_t min_size = std::min(results.size(), dets.size());
    
    for (size_t i = 0; i < min_size; ++i) {
        CLSCONF c = results[i].first;
        dets[i].classIdentifier = c.cls;
        dets[i].confidence = c.conf;
    }
}


void DeepSort::sort(cv::Mat& frame, model_internal::DETECTIONS& detections) {
    bool flag = featureExtractor->getRectsFeature(frame, detections);
    if (flag) {
        objTracker->predict();
        objTracker->update(detections);
        //result.clear();
        for (Track& track : objTracker->tracks) {
            if (!track.isConfirmed() || track.time_since_update > 1)
                continue;
            result.push_back(std::make_pair(track.track_id, track.toTlwh()));
        }
    }
}

void DeepSort::sort(cv::Mat& frame, model_internal::DETECTIONSV2& detectionsv2) {
    std::vector<CLSCONF>& clsConf = detectionsv2.first;
    model_internal::DETECTIONS& detections = detectionsv2.second;
    bool flag = featureExtractor->getRectsFeature(frame, detections);
    if (flag) {
        objTracker->predict();
        objTracker->update(detectionsv2);
        result.clear();
        results.clear();
        for (Track& track : objTracker->tracks) {
            if (!track.isConfirmed() || track.time_since_update > 1)
                continue;
            result.push_back(std::make_pair(track.track_id, track.toTlwh()));
            results.push_back(std::make_pair(CLSCONF(track.cls, track.conf) ,track.toTlwh()));
        }
    }
}

void DeepSort::sort(std::vector<DetectionBox>& dets) {
    model_internal::DETECTIONS detections;
    for (DetectionBox i : dets) {
        tracking::DETECTBOX box(i.x1, i.y1, i.x2-i.x1, i.y2-i.y1);
        DETECTION_ROW d;
        d.tlwh = box;
        d.confidence = i.confidence;
        detections.push_back(d);
    }
    if (detections.size() > 0)
        sort(detections);
    dets.clear();
    for (const auto& r : result) {
        tracking::DETECTBOX i = r.second;
        DetectionBox b(i(0), i(1), i(2), i(3), 1.);
        b.trackIdentifier = r.first;
        dets.push_back(b);
    }
}

void DeepSort::sort(model_internal::DETECTIONS& detections) {
    bool flag = featureExtractor->getRectsFeature(detections);
    if (flag) {
        objTracker->predict();
        objTracker->update(detections);
        result.clear();
        for (Track& track : objTracker->tracks) {
            if (!track.isConfirmed() || track.time_since_update > 1)
                continue;
            result.push_back(std::make_pair(track.track_id, track.toTlwh()));
        }
    }
}
