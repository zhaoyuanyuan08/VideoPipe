#include "vp_sdk_frame_sink_node.h"

#include <algorithm>
#include <cmath>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace vp_sdk {

    vp_sdk_frame_sink_node::vp_sdk_frame_sink_node(
        std::string node_name,
        int channel_index,
        std::shared_ptr<vp_sdk_blocking_queue<vp_jpeg_frame>> frames,
        int jpeg_quality,
        int max_width):
        vp_nodes::vp_des_node(node_name, channel_index),
        frames(std::move(frames)),
        jpeg_quality(std::max(1, std::min(100, jpeg_quality))),
        max_width(max_width) {
        this->initialized();
    }

    vp_sdk_frame_sink_node::~vp_sdk_frame_sink_node() {
        deinitialized();
    }

    std::shared_ptr<vp_objects::vp_meta> vp_sdk_frame_sink_node::handle_frame_meta(
        std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        const cv::Mat& source = !meta->osd_frame.empty() ? meta->osd_frame : meta->frame;
        if (!source.empty()) {
            cv::Mat output;
            if (max_width > 0 && source.cols > max_width) {
                auto scale = static_cast<double>(max_width) / static_cast<double>(source.cols);
                cv::resize(source, output, cv::Size(max_width, std::max(1, static_cast<int>(std::round(source.rows * scale)))), 0, 0, cv::INTER_AREA);
            }
            else {
                output = source;
            }

            std::vector<std::uint8_t> jpeg;
            std::vector<int> params {cv::IMWRITE_JPEG_QUALITY, jpeg_quality};
            if (cv::imencode(".jpg", output, jpeg, params) && !jpeg.empty()) {
                vp_jpeg_frame frame;
                frame.channel_index = meta->channel_index;
                frame.frame_index = meta->frame_index;
                frame.width = output.cols;
                frame.height = output.rows;
                frame.jpeg = std::move(jpeg);
                frames->push_latest(std::move(frame));
            }
        }
        return vp_nodes::vp_des_node::handle_frame_meta(meta);
    }

    std::shared_ptr<vp_objects::vp_meta> vp_sdk_frame_sink_node::handle_control_meta(
        std::shared_ptr<vp_objects::vp_control_meta> meta) {
        return vp_nodes::vp_des_node::handle_control_meta(meta);
    }

}
