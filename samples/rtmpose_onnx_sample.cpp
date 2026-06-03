#ifdef VP_WITH_ONNXRUNTIME

#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_rtmpose_onnx_node.h"
#include "../nodes/osd/vp_pose_osd_node.h"
#include "../nodes/vp_screen_des_node.h"

#include "../utils/analysis_board/vp_analysis_board.h"

#include <string>

/*
* ## rtmpose onnx sample ##
* RTMPose inference using ONNXRuntime C++ SDK.
*/

int main(int argc, char** argv) {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    std::string model_path = argc > 1
        ? argv[1]
        : "/home/yuan/venture/vision_fullstack/service_backend/vision_service/models/rtmpose/small/model.onnx";
    std::string video_path = argc > 2
        ? argv[2]
        : "./vp_data/test_video/pose.mp4";
    std::string execution_provider = argc > 3
        ? argv[3]
        : "tensorrt";
    std::string trt_engine_cache_path = argc > 4
        ? argv[4]
        : "./vp_data/trt_engine_cache/rtmpose";

    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>(
        "file_src_0",
        0,
        video_path);
    auto rtmpose = std::make_shared<vp_nodes::vp_rtmpose_onnx_node>(
        "rtmpose_onnx",
        model_path,
        192,
        256,
        2.0f,
        0.0f,
        1,
        0,
        execution_provider,
        0,
        true,
        trt_engine_cache_path);
    auto pose_osd_0 = std::make_shared<vp_nodes::vp_pose_osd_node>("pose_osd_0");
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);

    rtmpose->attach_to({file_src_0});
    pose_osd_0->attach_to({rtmpose});
    screen_des_0->attach_to({pose_osd_0});

    file_src_0->start();

    vp_utils::vp_analysis_board board({file_src_0});
    board.display();
}

#endif
