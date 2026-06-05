#pragma once

#ifdef VP_WITH_TENSORRT_RUNTIME

#include "vp_pose_pipeline_types.h"
#include "vp_sdk_blocking_queue.h"

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace vp_nodes {
    class vp_node;
}

namespace vp_sdk {

    class vp_pose_pipeline {
    private:
        vp_pipeline_config config;
        std::shared_ptr<vp_sdk_blocking_queue<vp_pipeline_result>> results;
        std::shared_ptr<vp_sdk_blocking_queue<vp_jpeg_frame>> frames;
        std::vector<std::shared_ptr<vp_nodes::vp_node>> nodes;
        std::shared_ptr<vp_nodes::vp_node> source;
        mutable std::mutex lifecycle_lock;
        bool running = false;
        std::string last_error;

        void start_common(std::shared_ptr<vp_nodes::vp_node> source_node);

    public:
        explicit vp_pose_pipeline(vp_pipeline_config config);
        ~vp_pose_pipeline();

        void start_file(const std::string& path, bool cycle = true);
        void start_rtsp(const std::string& url);
        void stop();

        std::optional<vp_pipeline_result> read_result(int timeout_ms = 0);
        std::optional<vp_jpeg_frame> read_jpeg_frame(int timeout_ms = 0);

        bool is_running() const;
        std::string error() const;
        std::size_t dropped_results() const;
        std::size_t dropped_frames() const;
    };

}

#endif
