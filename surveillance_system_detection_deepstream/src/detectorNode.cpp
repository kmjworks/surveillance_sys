#include "detectorNode.hpp"
#include <exception>
#include <gstreamer-1.0/gst/gstelement.h>
#include <gstreamer-1.0/gst/gstobject.h>


DetectorNode::DetectorNode(ros::NodeHandle& nh, ros::NodeHandle &pnh) : nh(nh), private_nh(pnh) {

    pnh.param<std::string>("config_file", runtimeConfiguration.configurationFile, "config/deepstream_primary.config");
    pnh.param<float>("confidence_threshold", runtimeConfiguration.confidenceThreshold, 0.25f); 
    pnh.param<bool>("enable_visualization", runtimeConfiguration.enableViz, true);
    pnh.param<std::string>("frame_id", runtimeConfiguration.frameConfiguration.frameIdentifier, "camera_link");
    pnh.param<int>("input_width",  runtimeConfiguration.frameConfiguration.inputWidthAndHeight.first, 1920);
    pnh.param<int>("input_height", runtimeConfiguration.frameConfiguration.inputWidthAndHeight.second, 1080);
    pnh.param<int>("model_width", runtimeConfiguration.frameConfiguration.detectorWidthAndHeight.first , 640);
    pnh.param<int>("model_height", runtimeConfiguration.frameConfiguration.detectorWidthAndHeight.second, 640);

    components.pubQueue.publishQueue.initialize(30);
    components.pubQueue.publisherRunning = true;
    components.pubQueue.publishingThread = std::thread(&DetectorNode::publishingHandler, this);

    components.pgieQueue.pgieQueue.initialize(10);
    components.pgieQueue.pgieQRunning = true;
    components.pgieQueue.pgieProcessingThread = std::thread(&DetectorNode::pgiePreprocess, this);

    if(initROSIO(nh)) {
        initPipeline();
    }
}

DetectorNode::~DetectorNode() {

    components.pgieQueue.pgieQRunning = false;
    components.pgieQueue.pgieQueue.stopWaitingThreads();
    if(components.pgieQueue.pgieProcessingThread.joinable()) components.pgieQueue.pgieProcessingThread.join();

    components.pubQueue.publisherRunning = false;
    components.pubQueue.publishQueue.stopWaitingThreads();
    if(components.pubQueue.publishingThread.joinable()) components.pubQueue.publishingThread.join();


    stopPipeline();
}

bool DetectorNode::initROSIO(ros::NodeHandle& nodeHandle) {
    try {
        rosInterface.pub_detectedMotion = nodeHandle.advertise<vision_msgs::Detection2DArray>("yolo/runtime_detections", 10);
        rosInterface.pub_vizDebug = nodeHandle.advertise<sensor_msgs::Image>("yolo/runtime_detectionVisualizationDebug", 1);
        rosInterface.pub_imageSrc = nodeHandle.advertise<sensor_msgs::Image>("pipeline/runtime_potentialMotionEvents", 30);
    } catch (const std::runtime_error &e) {
        ROS_ERROR_THROTTLE(5.0, "[DetectorNode] Error caught during ROS IO initialization : %s", e.what());
        return false;
    }

    return true;
}

vision_msgs::Detection2DArray DetectorNode::getDetectionArray(ros::Time timestamp, const std::string& frameId) {
    vision_msgs::Detection2DArray detectionArr;
    detectionArr.header.stamp = ros::Time::now();
    detectionArr.header.frame_id = frameId;

    return detectionArr;
}

bool DetectorNode::initPipeline(void) {
    gst_init(NULL, NULL);
    dsInterface.appCtx = g_new0(AppCtx, 1);
    dsInterface.parsedConfig = &dsInterface.appCtx->config; 

    std::string& configFilePath = runtimeConfiguration.configurationFile;
    gchar* cfg_file_path_cstr = g_strdup(configFilePath.c_str());

    if(not parse_config_file(dsInterface.parsedConfig, cfg_file_path_cstr)) {
        ROS_ERROR_THROTTLE(5.0, "[DetectorNode] Failed to parse DeepStream config file: %s", runtimeConfiguration.configurationFile.c_str());
        g_free(cfg_file_path_cstr);
        g_free(dsInterface.appCtx);
        dsInterface.appCtx = nullptr;
        return false;
    }
    g_free(cfg_file_path_cstr);

    if (not create_pipeline(dsInterface.appCtx, NULL, NULL, NULL, NULL)) {
        ROS_ERROR_THROTTLE(5.0, "[DetectorNode] Failed to create pipeline.");
        g_free(dsInterface.appCtx);
        dsInterface.appCtx = nullptr;
        return false;
    }

    GstElement* ext_pipeline = dsInterface.appCtx->pipeline.pipeline;
    if (!ext_pipeline) {
        ROS_ERROR("[DetectorNode] Failed to get GStreamer pipeline element from AppCtx.");
        destroy_pipeline(dsInterface.appCtx); 
        g_free(dsInterface.appCtx);
        dsInterface.appCtx = nullptr;
        return false;
    }

    dsInterface.loop = g_main_loop_new(NULL, FALSE);
    if (not dsInterface.loop) {
        ROS_ERROR("[DetectorNode] Failed to create GMainLoop.");
        destroy_pipeline(dsInterface.appCtx);
        g_free(dsInterface.appCtx);
        dsInterface.appCtx = nullptr;
        return false;
    }

    GstBus *bus = gst_element_get_bus(ext_pipeline);
    if (not bus) {
        ROS_ERROR("[DetectorNode] Failed to get pipeline bus.");
        g_main_loop_unref(dsInterface.loop);
        dsInterface.loop = nullptr;
        destroy_pipeline(dsInterface.appCtx);
        g_free(dsInterface.appCtx);
        dsInterface.appCtx = nullptr;
        return false;
    }

    gst_bus_add_watch(bus, (GstBusFunc)DetectorNode::busCb, dsInterface.loop);
    gst_object_unref(bus);
    
    GstElement *osd_element = gst_bin_get_by_name(GST_BIN(ext_pipeline), "nvosd0");
    if (not osd_element) {
        ROS_ERROR_THROTTLE(5.0, "[DetectorNode] Failed to find OSD element in pipeline.");
        g_main_loop_unref(dsInterface.loop); dsInterface.loop = nullptr;
        destroy_pipeline(dsInterface.appCtx); g_free(dsInterface.appCtx);
        return false;
    }

       
    GstPad *osd_src_pad = gst_element_get_static_pad(osd_element, "src");
    if (not osd_src_pad) {
        ROS_ERROR_THROTTLE(5.0, "[DetectorNode] Failed to get OSD sink pad.");
        gst_object_unref(osd_element);
        g_main_loop_unref(dsInterface.loop); dsInterface.loop = nullptr;
        destroy_pipeline(dsInterface.appCtx); g_free(dsInterface.appCtx);
        return false;
    }

    dsInterface.osd_probeIdentifier = gst_pad_add_probe(osd_src_pad, GST_PAD_PROBE_TYPE_BUFFER, osdPadBufferProbe, this, NULL);
    gst_object_unref(osd_src_pad);
    gst_object_unref(osd_element);


    
    GstElement *pgie = gst_bin_get_by_name(GST_BIN(ext_pipeline), "primary_gie");  

    if(pgie) {
        ROS_INFO("[DetectorNode] Found primary inference engine, setting up coordinate scaling probe");
        g_object_set(G_OBJECT(pgie), "confidence-threshold", runtimeConfiguration.confidenceThreshold, NULL);
        ROS_INFO("[DetectorNode] Set PGIE confidence threshold to %.2f", runtimeConfiguration.confidenceThreshold);
        GstPad *pgie_src_pad = gst_element_get_static_pad(pgie, "src");
        
        if(pgie_src_pad) {
            dsInterface.pgie_probeIdentifier = gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER, pgieSrcPadBufferProbe, this, NULL);
            gst_object_unref(pgie_src_pad);
        } else {
            ROS_WARN("[DetectorNode] Could not get source pad from primary inference engine");
        }
        gst_object_unref(pgie);
    } else {
        ROS_WARN("[DetectorNode] Could not find primary inference engine element");
    }

    ROS_INFO("[DetectorNode] Setting pipeline to PLAYING state.");
    if (gst_element_set_state(ext_pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        ROS_ERROR_THROTTLE(5.0, "[DetectorNode] Failed to set pipeline to PLAYING state.");
        g_main_loop_unref(dsInterface.loop); dsInterface.loop = nullptr;
        destroy_pipeline(dsInterface.appCtx); g_free(dsInterface.appCtx);
        return false;
    }

    state.running = true;
    state.pipelineThread = std::thread([this]() {
        ROS_INFO("[DetectorNode] Starting GMainLoop in a new thread.");
        g_main_loop_run(this->dsInterface.loop);
        ROS_INFO("[DetectorNode] GMainLoop finished.");
    });

    ROS_INFO("[DetectorNode] Detector pipeline started.");
    return true;
} 

void DetectorNode::stopPipeline() {
    if(not state.running) return;
    
    ROS_INFO("[DetectorNode] Stopping detector pipeline...");
    state.running = false;
    
    if (dsInterface.loop) {
        ROS_INFO("[DetectorNode] Quitting GMainLoop.");
        g_main_loop_quit(dsInterface.loop);
    }

    if(state.pipelineThread.joinable()) state.pipelineThread.join();

    if (dsInterface.loop) {
        g_main_loop_unref(dsInterface.loop);
        dsInterface.loop = nullptr;
    }

    if (dsInterface.appCtx) {
        GstElement* ext_pipeline = dsInterface.appCtx->pipeline.pipeline;
        if (ext_pipeline) {
            ROS_INFO("[DetectorNode] Setting pipeline to NULL state.");
            gst_element_set_state(ext_pipeline, GST_STATE_NULL);
        } else {
            ROS_WARN("[DetectorNode] Pipeline element in AppCtx was null during stop sequence.");
        }

        ROS_INFO("[DetectorNode] Calling destroy_pipeline to clean up AppCtx resources.");
        destroy_pipeline(dsInterface.appCtx);

        ROS_INFO("[DetectorNode] Freeing AppCtx.");
        g_free(dsInterface.appCtx);
        dsInterface.appCtx = nullptr;
    }


    ROS_INFO("[DetectorNode] Detector pipeline stopped.");
}

GstPadProbeReturn DetectorNode::osdPadBufferProbe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    static int frame_counter = 0;
    DetectorNode *self = static_cast<DetectorNode*>(user_data);

    GstBuffer *buf = (GstBuffer *)info->data;
    if(not buf) return GST_PAD_PROBE_OK;

    self->processBuffer(buf);

    return GST_PAD_PROBE_OK;
}

GstPadProbeReturn DetectorNode::pgieSrcPadBufferProbe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    static int frameCount = 0;
    frameCount++;

    DetectorNode *self = static_cast<DetectorNode*>(user_data);
    GstBuffer *buf = (GstBuffer *)info->data;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) {
        if (frameCount % 100 == 0) {
            ROS_WARN("[DetectorNode] No batch metadata in PGIE on frame %d", frameCount);
        }
        return GST_PAD_PROBE_OK;
    }

    int objCount = 0;

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            ++objCount;
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
        }
    }

    if (objCount > 0) {
        detector_internal::PGIEData pgieData;
        pgieData.batchMeta = batch_meta;
        pgieData.buffer = buf;
        pgieData.timestamp = ros::Time::now();

        for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
            NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
            
            for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) { 
                NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
                if(obj_meta) pgieData.objects.push_back(obj_meta);
            }
            
        }

        if(!self->components.pgieQueue.pgieQueue.try_push(std::move(pgieData))) {
            ROS_WARN_THROTTLE(5.0, "[DetectorNode] PGIE queue full, dropping frame with %d objects", objCount);
        }
    }

    //for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        //NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);

        //float model_width = self->runtimeConfiguration.frameConfiguration.detectorWidthAndHeight.first;
        //float model_height = self->runtimeConfiguration.frameConfiguration.detectorWidthAndHeight.second;
        //float frame_width = frame_meta->source_frame_width;
        //float frame_height = frame_meta->source_frame_height;

        //float scale_x = frame_width / model_width;
        //float scale_y = frame_height / model_height;

        /*for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
            
            obj_meta->rect_params.left *= scale_x;
            obj_meta->rect_params.top *= scale_y;
            obj_meta->rect_params.width *= scale_x;
            obj_meta->rect_params.height *= scale_y;
        }*/
   // }


    return GST_PAD_PROBE_OK;
}

void DetectorNode::publishingHandler() {
    while(components.pubQueue.publisherRunning) {
        auto item = components.pubQueue.publishQueue.pop();

        if(!item.has_value()) {
            if(!components.pubQueue.publisherRunning) break;
            continue;
        }

        const auto& val = item.value();

        if(val.srcImage) {
            rosInterface.pub_imageSrc.publish(val.srcImage);
        }

        if(val.hasDetections) {
            rosInterface.pub_detectedMotion.publish(val.detections);
            if(runtimeConfiguration.enableViz && !val.vizImg.empty()) {
                sensor_msgs::ImagePtr vizMsg = cv_bridge::CvImage(val.detections.header, "bgr8", val.vizImg).toImageMsg();
                rosInterface.pub_vizDebug.publish(vizMsg);

                ROS_DEBUG_THROTTLE(5.0,"[DetectorNode] Published detection visualization with %zu boxes",
                    val.detections.detections.size());
            }
        }
    }
    ROS_DEBUG_THROTTLE(5.0, "[DetectorNode] Publisher thread stopped");
}

void DetectorNode::processBuffer(GstBuffer *buf) {
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if(not batch_meta) {
        ROS_WARN_THROTTLE(5.0, "[DetectorNode] No batch metadata found");
        return;
    }

    NvBufSurface *surface = nullptr;
    gpointer state = nullptr;
    GstMapInfo map_info;

    if (!gst_buffer_map(buf, &map_info, GST_MAP_READ)) {
        ROS_WARN_THROTTLE(5.0, "[DetectorNode] Failed to map buffer");
        return;
    }

    surface = (NvBufSurface *)map_info.data;

    gst_buffer_unmap(buf, &map_info);

    if (!surface) {
        ROS_WARN_THROTTLE(5.0, "[DetectorNode] Failed to get NvBufSurface from buffer");
        return;
    } else {
      //  ROS_INFO("[DetectorNode] Surface info - batchSize: %d, gpuId: %d", 
        //    surface->batchSize, surface->gpuId);
    }

    vision_msgs::Detection2DArray detectionArr = getDetectionArray(ros::Time::now(), runtimeConfiguration.frameConfiguration.frameIdentifier);
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frameMeta = (NvDsFrameMeta *)(l_frame->data);
        if(!frameMeta) continue;

        if(NvBufSurfaceMap(surface, frameMeta->batch_id, -1, NVBUF_MAP_READ) != 0) {
            ROS_WARN_THROTTLE(5.0, "[DetectorNode] Failed to map surface for reading");
            continue;
        }

        std_msgs::Header msg_header;
        msg_header.stamp = ros::Time::now();
        msg_header.frame_id = runtimeConfiguration.frameConfiguration.frameIdentifier;
        NvBufSurfaceParams *params = &surface->surfaceList[frameMeta->batch_id];
        cv::Mat rgbaFrame(params->height, params->width, CV_8UC4, params->mappedAddr.addr[0], params->pitch);
        sensor_msgs::ImagePtr frameMsg = cv_bridge::CvImage(msg_header, "rgba8", rgbaFrame).toImageMsg();

        vision_msgs::Detection2DArray detectionArr;
        detectionArr.header = msg_header;

        for(NvDsMetaList *l_obj = frameMeta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta*)(l_obj->data);
            if(!obj_meta) continue;

            vision_msgs::Detection2D detection;
            detection.header = msg_header;
            
            float x = obj_meta->rect_params.left;
            float y = obj_meta->rect_params.top;
            float width = obj_meta->rect_params.width;
            float height = obj_meta->rect_params.height;

            detection.bbox.center.x = x + width / 2.0;
            detection.bbox.center.y = y + height / 2.0;
            detection.bbox.size_x = width;
            detection.bbox.size_y = height;

            vision_msgs::ObjectHypothesisWithPose hyp;
            hyp.id = obj_meta->class_id;
            hyp.score = obj_meta->confidence;
            detection.results.push_back(hyp);

            detectionArr.detections.push_back(detection);
        }

        NvBufSurfaceUnMap(surface, frameMeta->batch_id, -1);

        detector_internal::PublishedFrameData pubf;
        pubf.timestamp = msg_header.stamp;
        pubf.frameIdentifier = msg_header.frame_id;
        pubf.srcImage = frameMsg;
        pubf.detections = detectionArr;
        pubf.hasDetections = !detectionArr.detections.empty();

        if (!components.pubQueue.publishQueue.try_push(std::move(pubf))) {
            ROS_WARN_THROTTLE(1.0, "[DetectorNode] Publisher queue full, dropping frame");
        }
    }
}

void DetectorNode::pgiePreprocess() {
    while(components.pgieQueue.pgieQRunning) {
        auto objMeta = components.pgieQueue.pgieQueue.pop();
        if(!objMeta.has_value()) {
            if(!components.pgieQueue.pgieQRunning) break;
            continue; 
        }

        const auto& pgieData = objMeta.value();

        if(!pgieData.objects.empty()) {
            vision_msgs::Detection2DArray detectionArr;
            detectionArr.header.stamp = pgieData.timestamp;
            detectionArr.header.frame_id = runtimeConfiguration.frameConfiguration.frameIdentifier;

            for(const auto& obj_meta : pgieData.objects) {
                vision_msgs::Detection2D detection;
                detection.header = detectionArr.header;

                float x = obj_meta->rect_params.left;
                float y = obj_meta->rect_params.top;
                float width = obj_meta->rect_params.width;
                float height = obj_meta->rect_params.height;

                detection.bbox.center.x = x + width / 2.0;
                detection.bbox.center.y = y + height / 2.0;
                detection.bbox.size_x = width;
                detection.bbox.size_y = height;

                vision_msgs::ObjectHypothesisWithPose hyp;
                hyp.id = obj_meta->class_id;
                hyp.score = obj_meta->confidence;
                detection.results.push_back(hyp);

                detectionArr.detections.push_back(detection);
            }

            detector_internal::PublishedFrameData pubf;
            pubf.timestamp = pgieData.timestamp;
            pubf.frameIdentifier = runtimeConfiguration.frameConfiguration.frameIdentifier;
            pubf.detections = detectionArr;
            pubf.hasDetections = true;

            if (!components.pubQueue.publishQueue.try_push(std::move(pubf))) {
                ROS_WARN_THROTTLE(1.0, "[DetectorNode] Publisher queue full, dropping detections");
            }
        }
    }

    ROS_INFO("[DetectorNode] PGIE processing thread stopped");
}

gboolean DetectorNode::busCb(GstBus *bus, GstMessage *message, gpointer loop_to_quit_ptr) {
    GMainLoop *loop = (GMainLoop*)loop_to_quit_ptr;

    switch (GST_MESSAGE_TYPE(message)) {
        case GST_MESSAGE_EOS:
            ROS_INFO("[DetectorNode::busCb] EOS received from element %s. Quitting GMainLoop.",
                      GST_OBJECT_NAME(GST_MESSAGE_SRC(message)));
            g_main_loop_quit(loop);
            break;
        case GST_MESSAGE_ERROR: {
            GError *err = nullptr;
            gchar *debug_info = nullptr;
            gst_message_parse_error(message, &err, &debug_info);
            ROS_ERROR("[DetectorNode::busCb] GStreamer Error from element %s: %s. Debug: %s. Quitting GMainLoop.",
                      GST_OBJECT_NAME(GST_MESSAGE_SRC(message)),
                      err ? err->message : "Unknown error",
                      debug_info ? debug_info : "None");
            if (err) g_error_free(err);
            if (debug_info) g_free(debug_info);
            g_main_loop_quit(loop);
            break;
        }
        default:
            break;
    }
    return TRUE;
}