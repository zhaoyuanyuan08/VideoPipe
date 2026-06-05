#include "../cpp/vp_pose_pipeline.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

int main(int argc, char** argv) {
#ifndef VP_WITH_TENSORRT_RUNTIME
    std::cerr << "videopipe_sdk_file_sample requires VP_WITH_TENSORRT_RUNTIME" << std::endl;
    return 1;
#else
    if (argc < 5) {
        std::cerr
            << "usage: " << argv[0]
            << " <video> <yolo.engine> <rtmpose.engine> <labels> [frames_to_read] [score_threshold] [decoder]"
            << std::endl;
        return 1;
    }

    vp_sdk::vp_pipeline_config config;
    config.yolo_engine = argv[2];
    config.pose_engine = argv[3];
    config.labels = argv[4];
    config.enable_frame_output = true;
    config.enable_osd = true;

    int frames_to_read = argc > 5 ? std::max(1, std::atoi(argv[5])) : 25;
    if (argc > 6) {
        config.score_threshold = static_cast<float>(std::atof(argv[6]));
    }
    if (argc > 7) {
        config.decoder = argv[7];
    }

    try {
        vp_sdk::vp_pose_pipeline pipeline(config);
        pipeline.start_file(argv[1], true);

        auto start = std::chrono::steady_clock::now();
        int received = 0;
        while (received < frames_to_read) {
            auto result = pipeline.read_result(3000);
            if (!result) {
                std::cerr << "timeout waiting for result" << std::endl;
                break;
            }
            ++received;
            std::cout
                << "frame=" << result->frame_index
                << " targets=" << result->targets.size()
                << " poses=" << result->poses.size()
                << " latency_ms=" << result->latency_ms
                << std::endl;
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        pipeline.stop();

        if (elapsed > 0) {
            std::cout << "sdk_sample_fps=" << received * 1000.0 / elapsed << std::endl;
        }
        std::cout << "dropped_results=" << pipeline.dropped_results()
                  << " dropped_frames=" << pipeline.dropped_frames()
                  << std::endl;
    }
    catch (const std::exception& exc) {
        std::cerr << "videopipe sdk sample failed: " << exc.what() << std::endl;
        return 2;
    }
    catch (const char* exc) {
        std::cerr << "videopipe sdk sample failed: " << exc << std::endl;
        return 2;
    }
    catch (...) {
        std::cerr << "videopipe sdk sample failed: unknown non-standard exception" << std::endl;
        return 2;
    }

    return 0;
#endif
}
