#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vp_sdk {

    struct vp_pose_keypoint_result {
        int type = -1;
        int x = -1;
        int y = -1;
        float score = 0.0f;
    };

    struct vp_pose_result {
        int type = -1;
        std::vector<vp_pose_keypoint_result> keypoints;
    };

    struct vp_target_result {
        int x = 0;
        int y = 0;
        int width = 0;
        int height = 0;
        int class_id = -1;
        float score = 0.0f;
        std::string label;
        int track_id = -1;
    };

    struct vp_pipeline_result {
        int channel_index = -1;
        int frame_index = -1;
        int width = 0;
        int height = 0;
        int fps = 0;
        int64_t latency_ms = 0;
        std::vector<vp_target_result> targets;
        std::vector<vp_pose_result> poses;
    };

    struct vp_jpeg_frame {
        int channel_index = -1;
        int frame_index = -1;
        int width = 0;
        int height = 0;
        std::vector<std::uint8_t> jpeg;
    };

    struct vp_pipeline_config {
        std::string yolo_engine;
        std::string pose_engine;
        std::string labels;
        int yolo_width = 640;
        int yolo_height = 640;
        int pose_width = 192;
        int pose_height = 256;
        float score_threshold = 0.35f;
        float nms_threshold = 0.45f;
        int max_detections = 20;
        int device_id = 0;
        int result_queue_size = 100;
        int frame_queue_size = 3;
        bool enable_frame_output = true;
        bool enable_osd = true;
        int jpeg_quality = 82;
        int jpeg_max_width = 960;
        std::string decoder = "avdec_h264";
        int skip_interval = 0;
    };

}
