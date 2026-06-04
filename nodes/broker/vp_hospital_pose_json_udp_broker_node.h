#pragma once

#include "vp_msg_broker_node.h"

#include <netinet/in.h>

namespace vp_nodes {
    // Sends person boxes and pose keypoints as newline-free JSON over UDP.
    class vp_hospital_pose_json_udp_broker_node: public vp_msg_broker_node
    {
    private:
        std::string des_ip = "";
        int des_port = 0;
        int socket_fd = -1;
        sockaddr_in des_addr {};

    protected:
        virtual void format_msg(const std::shared_ptr<vp_objects::vp_frame_meta>& meta, std::string& msg) override;
        virtual void broke_msg(const std::string& msg) override;

    public:
        vp_hospital_pose_json_udp_broker_node(
            std::string node_name,
            std::string des_ip = "127.0.0.1",
            int des_port = 9977,
            int broking_cache_warn_threshold = 50,
            int broking_cache_ignore_threshold = 200);
        ~vp_hospital_pose_json_udp_broker_node();
    };
}
