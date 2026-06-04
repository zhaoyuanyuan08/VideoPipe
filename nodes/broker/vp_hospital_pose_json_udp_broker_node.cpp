#include "vp_hospital_pose_json_udp_broker_node.h"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <sstream>

namespace {
    std::string json_escape(const std::string& value) {
        std::ostringstream out;
        for (char ch: value) {
            switch (ch) {
                case '\\': out << "\\\\"; break;
                case '"': out << "\\\""; break;
                case '\b': out << "\\b"; break;
                case '\f': out << "\\f"; break;
                case '\n': out << "\\n"; break;
                case '\r': out << "\\r"; break;
                case '\t': out << "\\t"; break;
                default: out << ch; break;
            }
        }
        return out.str();
    }
}

namespace vp_nodes {
    vp_hospital_pose_json_udp_broker_node::vp_hospital_pose_json_udp_broker_node(
        std::string node_name,
        std::string des_ip,
        int des_port,
        int broking_cache_warn_threshold,
        int broking_cache_ignore_threshold):
        vp_msg_broker_node(
            node_name,
            vp_broke_for::POSE,
            broking_cache_warn_threshold,
            broking_cache_ignore_threshold),
        des_ip(des_ip),
        des_port(des_port) {
        socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
        if (socket_fd >= 0) {
            des_addr.sin_family = AF_INET;
            des_addr.sin_port = htons(static_cast<uint16_t>(des_port));
            if (inet_pton(AF_INET, des_ip.c_str(), &des_addr.sin_addr) != 1) {
                VP_WARN(vp_utils::string_format("[%s] invalid UDP host `%s`, hospital pose broker disabled", node_name.c_str(), des_ip.c_str()));
                close(socket_fd);
                socket_fd = -1;
            }
        }
        else {
            VP_WARN(vp_utils::string_format("[%s] unable to create UDP socket, hospital pose broker disabled", node_name.c_str()));
        }
        VP_INFO(vp_utils::string_format("[%s] hospital pose json udp broker -> %s:%d", node_name.c_str(), des_ip.c_str(), des_port));
        this->initialized();
    }

    vp_hospital_pose_json_udp_broker_node::~vp_hospital_pose_json_udp_broker_node() {
        deinitialized();
        stop_broking();
        if (socket_fd >= 0) {
            close(socket_fd);
            socket_fd = -1;
        }
    }

    void vp_hospital_pose_json_udp_broker_node::format_msg(
        const std::shared_ptr<vp_objects::vp_frame_meta>& meta,
        std::string& msg) {
        std::ostringstream out;
        out << "{";
        out << "\"channel_index\":" << meta->channel_index << ",";
        out << "\"frame_index\":" << meta->frame_index << ",";
        out << "\"width\":" << meta->frame.cols << ",";
        out << "\"height\":" << meta->frame.rows << ",";
        out << "\"fps\":" << meta->fps << ",";

        out << "\"targets\":[";
        for (size_t i = 0; i < meta->targets.size(); ++i) {
            const auto& target = meta->targets[i];
            if (i > 0) {
                out << ",";
            }
            out << "{";
            out << "\"x\":" << target->x << ",";
            out << "\"y\":" << target->y << ",";
            out << "\"width\":" << target->width << ",";
            out << "\"height\":" << target->height << ",";
            out << "\"class_id\":" << target->primary_class_id << ",";
            out << "\"score\":" << target->primary_score << ",";
            out << "\"label\":\"" << json_escape(target->primary_label) << "\"";
            out << "}";
        }
        out << "],";

        out << "\"pose_targets\":[";
        for (size_t i = 0; i < meta->pose_targets.size(); ++i) {
            const auto& pose = meta->pose_targets[i];
            if (i > 0) {
                out << ",";
            }
            out << "{";
            out << "\"type\":" << static_cast<int>(pose->type) << ",";
            out << "\"keypoints\":[";
            for (size_t j = 0; j < pose->key_points.size(); ++j) {
                const auto& point = pose->key_points[j];
                if (j > 0) {
                    out << ",";
                }
                out << "{";
                out << "\"type\":" << point.point_type << ",";
                out << "\"x\":" << point.x << ",";
                out << "\"y\":" << point.y << ",";
                out << "\"score\":" << point.score;
                out << "}";
            }
            out << "]";
            out << "}";
        }
        out << "]";
        out << "}";
        msg = out.str();
    }

    void vp_hospital_pose_json_udp_broker_node::broke_msg(const std::string& msg) {
        if (socket_fd < 0) {
            return;
        }
        sendto(
            socket_fd,
            msg.data(),
            msg.size(),
            MSG_DONTWAIT,
            reinterpret_cast<sockaddr*>(&des_addr),
            sizeof(des_addr));
    }
}
