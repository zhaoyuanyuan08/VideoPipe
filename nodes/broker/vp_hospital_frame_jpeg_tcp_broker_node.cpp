#include "vp_hospital_frame_jpeg_tcp_broker_node.h"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {
    constexpr uint32_t JPEG_MAGIC = 0x56504a46; // VPJF
}

namespace vp_nodes {
    vp_hospital_frame_jpeg_tcp_broker_node::vp_hospital_frame_jpeg_tcp_broker_node(
        std::string node_name,
        std::string des_ip,
        int des_port,
        int jpeg_quality,
        int max_width):
        vp_node(node_name),
        des_ip(des_ip),
        des_port(des_port),
        jpeg_quality(std::max(1, std::min(100, jpeg_quality))),
        max_width(max_width) {
        des_addr.sin_family = AF_INET;
        des_addr.sin_port = htons(static_cast<uint16_t>(des_port));
        if (inet_pton(AF_INET, des_ip.c_str(), &des_addr.sin_addr) != 1) {
            VP_WARN(vp_utils::string_format("[%s] invalid TCP host `%s`, frame jpeg broker disabled", node_name.c_str(), des_ip.c_str()));
            socket_fd = -2;
        }
        VP_INFO(vp_utils::string_format("[%s] hospital frame jpeg tcp broker -> %s:%d", node_name.c_str(), des_ip.c_str(), des_port));
        this->initialized();
    }

    vp_hospital_frame_jpeg_tcp_broker_node::~vp_hospital_frame_jpeg_tcp_broker_node() {
        deinitialized();
        close_socket();
    }

    bool vp_hospital_frame_jpeg_tcp_broker_node::connect_if_needed() {
        if (socket_fd >= 0) {
            return true;
        }
        if (socket_fd == -2) {
            return false;
        }

        auto now = std::chrono::steady_clock::now();
        if (last_connect_attempt.time_since_epoch().count() != 0 &&
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_connect_attempt).count() < 1000) {
            return false;
        }
        last_connect_attempt = now;

        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) {
            return false;
        }
        if (connect(fd, reinterpret_cast<sockaddr*>(&des_addr), sizeof(des_addr)) != 0) {
            close(fd);
            return false;
        }
        socket_fd = fd;
        VP_INFO(vp_utils::string_format("[%s] connected frame jpeg tcp broker -> %s:%d", node_name.c_str(), des_ip.c_str(), des_port));
        return true;
    }

    void vp_hospital_frame_jpeg_tcp_broker_node::close_socket() {
        if (socket_fd >= 0) {
            close(socket_fd);
        }
        socket_fd = -1;
    }

    bool vp_hospital_frame_jpeg_tcp_broker_node::send_all(const void* data, size_t size) {
        const char* ptr = static_cast<const char*>(data);
        size_t sent = 0;
        while (sent < size) {
            auto n = send(socket_fd, ptr + sent, size - sent, MSG_NOSIGNAL);
            if (n <= 0) {
                close_socket();
                return false;
            }
            sent += static_cast<size_t>(n);
        }
        return true;
    }

    std::shared_ptr<vp_objects::vp_meta> vp_hospital_frame_jpeg_tcp_broker_node::handle_frame_meta(
        std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        if (!connect_if_needed()) {
            return meta;
        }

        cv::Mat frame = !meta->osd_frame.empty() ? meta->osd_frame : meta->frame;
        if (frame.empty()) {
            return meta;
        }

        cv::Mat stream_frame;
        if (max_width > 0 && frame.cols > max_width) {
            auto scale = static_cast<double>(max_width) / static_cast<double>(frame.cols);
            cv::resize(frame, stream_frame, cv::Size(max_width, std::max(1, static_cast<int>(std::round(frame.rows * scale)))), 0, 0, cv::INTER_AREA);
        }
        else {
            stream_frame = frame;
        }

        std::vector<unsigned char> jpeg;
        std::vector<int> params {cv::IMWRITE_JPEG_QUALITY, jpeg_quality};
        if (!cv::imencode(".jpg", stream_frame, jpeg, params) || jpeg.empty()) {
            return meta;
        }

        std::array<uint32_t, 3> header {
            htonl(JPEG_MAGIC),
            htonl(static_cast<uint32_t>(std::max(0, meta->frame_index))),
            htonl(static_cast<uint32_t>(jpeg.size()))
        };
        if (!send_all(header.data(), header.size() * sizeof(uint32_t))) {
            return meta;
        }
        send_all(jpeg.data(), jpeg.size());
        return meta;
    }

    std::shared_ptr<vp_objects::vp_meta> vp_hospital_frame_jpeg_tcp_broker_node::handle_control_meta(
        std::shared_ptr<vp_objects::vp_control_meta> meta) {
        return meta;
    }
}
