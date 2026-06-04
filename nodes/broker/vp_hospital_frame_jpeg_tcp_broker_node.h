#pragma once

#include "../vp_node.h"

#include <netinet/in.h>

namespace vp_nodes {
    // Sends the latest OSD frame as length-prefixed JPEG over a local TCP socket.
    class vp_hospital_frame_jpeg_tcp_broker_node: public vp_node
    {
    private:
        std::string des_ip;
        int des_port;
        int jpeg_quality;
        int max_width;
        int socket_fd = -1;
        sockaddr_in des_addr {};
        std::chrono::steady_clock::time_point last_connect_attempt {};

        bool connect_if_needed();
        void close_socket();
        bool send_all(const void* data, size_t size);

    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override;

    public:
        vp_hospital_frame_jpeg_tcp_broker_node(
            std::string node_name,
            std::string des_ip = "127.0.0.1",
            int des_port = 9978,
            int jpeg_quality = 82,
            int max_width = 960);
        ~vp_hospital_frame_jpeg_tcp_broker_node();
    };
}
