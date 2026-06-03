#ifdef VP_WITH_ONNXRUNTIME

#include "../nodes/vp_fake_des_node.h"
#include "../nodes/vp_file_src_node.h"
#include "../nodes/infers/vp_rtmpose_onnx_node.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>

/*
* ## rtmpose onnx benchmark sample ##
* Headless RTMPose benchmark using VideoPipe stream status.
*/

int main(int argc, char** argv) {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::WARN);
    VP_LOGGER_INIT();

    std::string model_path = argc > 1
        ? argv[1]
        : "/home/yuan/venture/vision_fullstack/vision_service/model_assets/rtmpose/small/model.onnx";
    std::string video_path = argc > 2
        ? argv[2]
        : "./vp_data/test_video/pose.mp4";
    std::string execution_provider = argc > 3
        ? argv[3]
        : "tensorrt";
    std::string trt_engine_cache_path = argc > 4
        ? argv[4]
        : "./vp_data/trt_engine_cache/rtmpose";
    int max_frames = argc > 5
        ? std::max(1, std::stoi(argv[5]))
        : 120;

    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>(
        "file_src_0",
        0,
        video_path,
        1.0f,
        false);
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
    auto fake_des_0 = std::make_shared<vp_nodes::vp_fake_des_node>("fake_des_0", 0);

    rtmpose->attach_to({file_src_0});
    fake_des_0->attach_to({rtmpose});

    std::mutex done_mutex;
    std::condition_variable done_cv;
    std::atomic<int> frames {0};
    std::atomic<long long> latency_sum_ms {0};
    std::atomic<int> latency_min_ms {0};
    std::atomic<int> latency_max_ms {0};
    std::atomic<float> latest_status_fps {0.0f};
    std::chrono::steady_clock::time_point first_frame_time;
    std::chrono::steady_clock::time_point last_frame_time;

    fake_des_0->set_stream_status_hooker([&](
        std::string node_name,
        vp_nodes::vp_stream_status status) {
        auto frame_count = ++frames;
        auto now = std::chrono::steady_clock::now();
        if (frame_count == 1) {
            first_frame_time = now;
            latency_min_ms = status.latency;
            latency_max_ms = status.latency;
        }
        last_frame_time = now;
        latency_sum_ms += status.latency;
        latency_min_ms = std::min(latency_min_ms.load(), status.latency);
        latency_max_ms = std::max(latency_max_ms.load(), status.latency);
        latest_status_fps = status.fps;

        if (frame_count % 30 == 0 || frame_count >= max_frames) {
            std::cout << "progress frame=" << frame_count
                      << " latency_ms=" << status.latency
                      << " status_fps=" << std::fixed << std::setprecision(2) << status.fps
                      << std::endl;
        }

        if (frame_count >= max_frames) {
            file_src_0->stop();
            done_cv.notify_one();
        }
    });

    auto start_time = std::chrono::steady_clock::now();
    file_src_0->start();

    {
        std::unique_lock<std::mutex> lock(done_mutex);
        done_cv.wait_for(lock, std::chrono::seconds(180), [&] {
            return frames.load() >= max_frames;
        });
    }

    auto measured_frames = frames.load();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        last_frame_time - first_frame_time).count();
    auto wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time).count();
    auto avg_latency = measured_frames > 0
        ? static_cast<double>(latency_sum_ms.load()) / measured_frames
        : 0.0;
    auto throughput_fps = total_ms > 0 && measured_frames > 1
        ? static_cast<double>(measured_frames - 1) * 1000.0 / total_ms
        : 0.0;

    std::cout << "summary"
              << " provider=" << execution_provider
              << " frames=" << measured_frames
              << " throughput_fps=" << std::fixed << std::setprecision(2) << throughput_fps
              << " avg_latency_ms=" << std::fixed << std::setprecision(2) << avg_latency
              << " min_latency_ms=" << latency_min_ms.load()
              << " max_latency_ms=" << latency_max_ms.load()
              << " latest_status_fps=" << std::fixed << std::setprecision(2) << latest_status_fps.load()
              << " wall_ms=" << wall_ms
              << std::endl;

    file_src_0->detach_recursively();
    return measured_frames >= max_frames ? 0 : 1;
}

#endif
