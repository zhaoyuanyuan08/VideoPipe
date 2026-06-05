#ifdef VP_WITH_TENSORRT_RUNTIME

#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_yolov8_trtexec_detector_node.h"
#include "../nodes/infers/vp_rtmpose_trtexec_secondary_node.h"
#include "../nodes/osd/vp_pose_osd_node.h"
#include "../nodes/broker/vp_hospital_pose_json_udp_broker_node.h"
#include "../nodes/broker/vp_hospital_frame_jpeg_tcp_broker_node.h"
#include "../nodes/vp_fake_des_node.h"

#include <cstdlib>

/*
* ## yolov8 person pose trtexec sample ##
* detect person boxes using a TensorRT engine generated from YOLOv8 ONNX by
* trtexec, then run RTMPose on each detected person crop.
*/

int main(int argc, char** argv) {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    std::string video_path = argc > 1 ? argv[1] : "./vp_data/test_video/pose.mp4";
    std::string detector_engine_path = argc > 2 ? argv[2] : "./vp_data/models/trt/yolo/yolov8n-person_fp16.engine";
    std::string pose_engine_path = argc > 3 ? argv[3] : "./vp_data/models/trt/rtmpose/rtmpose-small.engine";
    std::string labels_path = argc > 4 ? argv[4] : "./vp_data/models/coco_80classes.txt";
    std::string broker_host = argc > 5 ? argv[5] : "127.0.0.1";
    int broker_port = argc > 6 ? std::atoi(argv[6]) : 9977;
    int frame_port = argc > 7 ? std::atoi(argv[7]) : 9978;

    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, video_path);
    auto person_detector = std::make_shared<vp_nodes::vp_yolov8_trtexec_detector_node>(
        "person_detector",
        detector_engine_path,
        labels_path,
        640,
        640,
        1,
        0,
        0.25f,
        0.45f,
        20,
        std::vector<int>{0},
        0);
    auto person_pose = std::make_shared<vp_nodes::vp_rtmpose_trtexec_secondary_node>(
        "person_pose",
        pose_engine_path,
        192,
        256,
        2.0f,
        0.0f,
        1,
        std::vector<int>{0},
        0,
        0,
        10,
        0);
    auto pose_broker_0 = std::make_shared<vp_nodes::vp_hospital_pose_json_udp_broker_node>("pose_broker_0", broker_host, broker_port);
    auto pose_osd_0 = std::make_shared<vp_nodes::vp_pose_osd_node>("pose_osd_0");
    auto frame_broker_0 = std::make_shared<vp_nodes::vp_hospital_frame_jpeg_tcp_broker_node>("frame_broker_0", broker_host, frame_port);
    auto fake_des_0 = std::make_shared<vp_nodes::vp_fake_des_node>("fake_des_0", 0);

    person_detector->attach_to({file_src_0});
    person_pose->attach_to({person_detector});
    pose_broker_0->attach_to({person_pose});
    pose_osd_0->attach_to({pose_broker_0});
    frame_broker_0->attach_to({pose_osd_0});
    fake_des_0->attach_to({frame_broker_0});

    file_src_0->start();

    std::string wait;
    std::getline(std::cin, wait);
    file_src_0->detach_recursively();
}

#endif
