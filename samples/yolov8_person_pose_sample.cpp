#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_yolov8_onnx_detector_node.h"
#include "../nodes/infers/vp_rtmpose_onnx_secondary_node.h"
#include "../nodes/osd/vp_pose_osd_node.h"
#include "../nodes/vp_fake_des_node.h"

/*
* ## yolov8 person pose sample ##
* detect person boxes using a YOLOv8 ONNX model, then run RTMPose on each
* detected person crop. headless version for GPU/SSH benchmarking.
*/

int main(int argc, char** argv) {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    std::string video_path = argc > 1 ? argv[1] : "./vp_data/test_video/pose.mp4";
    std::string detector_model_path = argc > 2 ? argv[2] : "/home/yuan/venture/vision_fullstack/vision_service/model_assets/yolo/yolov8n-person/model.onnx";
    std::string pose_model_path = argc > 3 ? argv[3] : "/home/yuan/venture/vision_fullstack/vision_service/model_assets/rtmpose/small/model.onnx";
    std::string trt_cache_path = argc > 4 ? argv[4] : "./vp_data/trt_engine_cache/rtmpose";

    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, video_path);
    auto person_detector = std::make_shared<vp_nodes::vp_yolov8_onnx_detector_node>(
        "person_detector",
        detector_model_path,
        "",
        640,
        640,
        1,
        0,
        0.35f,
        0.45f,
        20,
        std::vector<int>{0});
    auto person_pose = std::make_shared<vp_nodes::vp_rtmpose_onnx_secondary_node>(
        "person_pose",
        pose_model_path,
        192,
        256,
        2.0f,
        0.0f,
        1,
        std::vector<int>{0},
        0,
        0,
        10,
        0,
        "tensorrt",
        0,
        true,
        trt_cache_path);
    auto pose_osd_0 = std::make_shared<vp_nodes::vp_pose_osd_node>("pose_osd_0");
    auto fake_des_0 = std::make_shared<vp_nodes::vp_fake_des_node>("fake_des_0", 0);

    person_detector->attach_to({file_src_0});
    person_pose->attach_to({person_detector});
    pose_osd_0->attach_to({person_pose});
    fake_des_0->attach_to({pose_osd_0});

    file_src_0->start();

    std::string wait;
    std::getline(std::cin, wait);
    file_src_0->detach_recursively();
}
