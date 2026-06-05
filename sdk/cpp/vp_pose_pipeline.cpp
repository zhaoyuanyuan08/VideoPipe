#ifdef VP_WITH_TENSORRT_RUNTIME

#include "vp_pose_pipeline.h"
#include "vp_sdk_frame_sink_node.h"
#include "vp_sdk_result_sink_node.h"

#include "../../nodes/infers/vp_rtmpose_trtexec_secondary_node.h"
#include "../../nodes/infers/vp_yolov8_trtexec_detector_node.h"
#include "../../nodes/osd/vp_pose_osd_node.h"
#include "../../nodes/vp_file_src_node.h"
#include "../../nodes/vp_rtsp_src_node.h"
#include "../../nodes/vp_src_node.h"
#include "../../utils/logger/vp_logger.h"

#include <cuda_runtime_api.h>
#include <mutex>
#include <stdexcept>

namespace vp_sdk {
    namespace {
        void ensure_logger_initialized() {
            static std::once_flag logger_once;
            std::call_once(logger_once, []() {
                VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
                VP_LOGGER_INIT();
            });
        }

        void ensure_cuda_device_available(int device_id) {
            int device_count = 0;
            auto status = cudaGetDeviceCount(&device_count);
            if (status != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA device check failed: ") + cudaGetErrorString(status));
            }
            if (device_id < 0 || device_id >= device_count) {
                throw std::runtime_error("CUDA device_id is out of range");
            }
        }
    }

    vp_pose_pipeline::vp_pose_pipeline(vp_pipeline_config config):
        config(std::move(config)),
        results(std::make_shared<vp_sdk_blocking_queue<vp_pipeline_result>>(this->config.result_queue_size)),
        frames(std::make_shared<vp_sdk_blocking_queue<vp_jpeg_frame>>(this->config.frame_queue_size)) {
    }

    vp_pose_pipeline::~vp_pose_pipeline() {
        stop();
        results->close();
        frames->close();
    }

    void vp_pose_pipeline::start_common(std::shared_ptr<vp_nodes::vp_node> source_node) {
        if (config.yolo_engine.empty() || config.pose_engine.empty()) {
            throw std::runtime_error("yolo_engine and pose_engine are required");
        }
        if (running) {
            throw std::runtime_error("pipeline is already running");
        }
        ensure_cuda_device_available(config.device_id);

        results->reopen();
        frames->reopen();
        nodes.clear();
        source = std::move(source_node);

        auto person_detector = std::make_shared<vp_nodes::vp_yolov8_trtexec_detector_node>(
            "sdk_person_detector",
            config.yolo_engine,
            config.labels,
            config.yolo_width,
            config.yolo_height,
            1,
            0,
            config.score_threshold,
            config.nms_threshold,
            config.max_detections,
            std::vector<int>{0},
            config.device_id);
        auto person_pose = std::make_shared<vp_nodes::vp_rtmpose_trtexec_secondary_node>(
            "sdk_person_pose",
            config.pose_engine,
            config.pose_width,
            config.pose_height,
            2.0f,
            0.0f,
            1,
            std::vector<int>{0},
            0,
            0,
            10,
            config.device_id);
        auto result_sink = std::make_shared<vp_sdk_result_sink_node>("sdk_result_sink", 0, results);

        person_detector->attach_to({source});
        person_pose->attach_to({person_detector});
        result_sink->attach_to({person_pose});

        nodes.push_back(source);
        nodes.push_back(person_detector);
        nodes.push_back(person_pose);
        nodes.push_back(result_sink);

        if (config.enable_frame_output) {
            std::shared_ptr<vp_nodes::vp_node> frame_parent = person_pose;
            if (config.enable_osd) {
                auto pose_osd = std::make_shared<vp_nodes::vp_pose_osd_node>("sdk_pose_osd");
                pose_osd->attach_to({person_pose});
                nodes.push_back(pose_osd);
                frame_parent = pose_osd;
            }
            auto frame_sink = std::make_shared<vp_sdk_frame_sink_node>(
                "sdk_frame_sink",
                0,
                frames,
                config.jpeg_quality,
                config.jpeg_max_width);
            frame_sink->attach_to({frame_parent});
            nodes.push_back(frame_sink);
        }

        running = true;
        last_error.clear();
        std::dynamic_pointer_cast<vp_nodes::vp_src_node>(source)->start();
    }

    void vp_pose_pipeline::start_file(const std::string& path, bool cycle) {
        std::lock_guard<std::mutex> guard(lifecycle_lock);
        try {
            ensure_logger_initialized();
            auto source_node = std::make_shared<vp_nodes::vp_file_src_node>(
                "sdk_file_src",
                0,
                path,
                1.0,
                cycle,
                config.decoder,
                config.skip_interval);
            start_common(source_node);
        }
        catch (const std::exception& exc) {
            last_error = exc.what();
            if (source) {
                source->detach_recursively();
            }
            nodes.clear();
            source.reset();
            results->clear();
            frames->clear();
            running = false;
            throw;
        }
    }

    void vp_pose_pipeline::start_rtsp(const std::string& url) {
        std::lock_guard<std::mutex> guard(lifecycle_lock);
        try {
            ensure_logger_initialized();
            auto source_node = std::make_shared<vp_nodes::vp_rtsp_src_node>(
                "sdk_rtsp_src",
                0,
                url,
                1.0,
                config.decoder,
                config.skip_interval);
            start_common(source_node);
        }
        catch (const std::exception& exc) {
            last_error = exc.what();
            if (source) {
                source->detach_recursively();
            }
            nodes.clear();
            source.reset();
            results->clear();
            frames->clear();
            running = false;
            throw;
        }
    }

    void vp_pose_pipeline::stop() {
        std::lock_guard<std::mutex> guard(lifecycle_lock);
        if (!running && nodes.empty()) {
            return;
        }
        try {
            if (source) {
                source->detach_recursively();
            }
        }
        catch (const std::exception& exc) {
            last_error = exc.what();
        }
        nodes.clear();
        source.reset();
        results->clear();
        frames->clear();
        running = false;
    }

    std::optional<vp_pipeline_result> vp_pose_pipeline::read_result(int timeout_ms) {
        return results->pop(timeout_ms);
    }

    std::optional<vp_jpeg_frame> vp_pose_pipeline::read_jpeg_frame(int timeout_ms) {
        return frames->pop(timeout_ms);
    }

    bool vp_pose_pipeline::is_running() const {
        std::lock_guard<std::mutex> guard(lifecycle_lock);
        return running;
    }

    std::string vp_pose_pipeline::error() const {
        std::lock_guard<std::mutex> guard(lifecycle_lock);
        return last_error;
    }

    std::size_t vp_pose_pipeline::dropped_results() const {
        return results->dropped_count();
    }

    std::size_t vp_pose_pipeline::dropped_frames() const {
        return frames->dropped_count();
    }

}

#endif
