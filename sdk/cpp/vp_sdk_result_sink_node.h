#pragma once

#include "../../nodes/vp_des_node.h"
#include "vp_pose_pipeline_types.h"
#include "vp_sdk_blocking_queue.h"

#include <memory>
#include <string>

namespace vp_sdk {

    class vp_sdk_result_sink_node: public vp_nodes::vp_des_node {
    private:
        std::shared_ptr<vp_sdk_blocking_queue<vp_pipeline_result>> results;

    protected:
        std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
        std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override;

    public:
        vp_sdk_result_sink_node(
            std::string node_name,
            int channel_index,
            std::shared_ptr<vp_sdk_blocking_queue<vp_pipeline_result>> results);
        ~vp_sdk_result_sink_node();
    };

}
