#include "vp_sdk_result_sink_node.h"

namespace vp_sdk {

    vp_sdk_result_sink_node::vp_sdk_result_sink_node(
        std::string node_name,
        int channel_index,
        std::shared_ptr<vp_sdk_blocking_queue<vp_pipeline_result>> results):
        vp_nodes::vp_des_node(node_name, channel_index),
        results(std::move(results)) {
        this->initialized();
    }

    vp_sdk_result_sink_node::~vp_sdk_result_sink_node() {
        deinitialized();
    }

    std::shared_ptr<vp_objects::vp_meta> vp_sdk_result_sink_node::handle_frame_meta(
        std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        vp_pipeline_result result;
        result.channel_index = meta->channel_index;
        result.frame_index = meta->frame_index;
        result.width = meta->frame.cols;
        result.height = meta->frame.rows;
        result.fps = meta->fps;
        result.latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now() - meta->create_time).count();

        result.targets.reserve(meta->targets.size());
        for (const auto& target: meta->targets) {
            vp_target_result item;
            item.x = target->x;
            item.y = target->y;
            item.width = target->width;
            item.height = target->height;
            item.class_id = target->primary_class_id;
            item.score = target->primary_score;
            item.label = target->primary_label;
            item.track_id = target->track_id;
            result.targets.push_back(std::move(item));
        }

        result.poses.reserve(meta->pose_targets.size());
        for (const auto& pose: meta->pose_targets) {
            vp_pose_result item;
            item.type = static_cast<int>(pose->type);
            item.keypoints.reserve(pose->key_points.size());
            for (const auto& keypoint: pose->key_points) {
                item.keypoints.push_back(vp_pose_keypoint_result {
                    keypoint.point_type,
                    keypoint.x,
                    keypoint.y,
                    keypoint.score
                });
            }
            result.poses.push_back(std::move(item));
        }

        results->push_latest(std::move(result));
        return vp_nodes::vp_des_node::handle_frame_meta(meta);
    }

    std::shared_ptr<vp_objects::vp_meta> vp_sdk_result_sink_node::handle_control_meta(
        std::shared_ptr<vp_objects::vp_control_meta> meta) {
        return vp_nodes::vp_des_node::handle_control_meta(meta);
    }

}
