#pragma once

#ifdef VP_WITH_ONNXRUNTIME

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>

#include "../vp_primary_infer_node.h"
#include "../../objects/vp_frame_pose_target.h"

namespace vp_nodes {
    // RTMPose ONNXRuntime node.
    // It writes COCO-17 keypoints into vp_frame_meta::pose_targets, so existing
    // vp_pose_osd_node can render the result without changes.
    class vp_rtmpose_onnx_node: public vp_primary_infer_node
    {
    private:
        struct BBox {
            int x1;
            int y1;
            int x2;
            int y2;
        };

        Ort::Env env;
        Ort::SessionOptions session_options;
        std::unique_ptr<Ort::Session> session;
        std::string input_name;
        std::vector<std::string> output_names;
        std::vector<const char*> output_name_ptrs;

        int input_width;
        int input_height;
        float simcc_split_ratio;
        float score_threshold;
        int infer_every_n_frames;
        int device_id;
        bool enable_fp16;
        std::string execution_provider;
        std::string trt_engine_cache_path;

        void configure_execution_provider();
        bool append_tensorrt_provider();
        bool append_cuda_provider();
        std::vector<float> preprocess(const cv::Mat& image, const BBox& bbox);
        std::vector<vp_objects::vp_pose_keypoint> decode_outputs(
            std::vector<Ort::Value>& outputs,
            const BBox& bbox);
        std::vector<vp_objects::vp_pose_keypoint> decode_simcc(
            Ort::Value& simcc_x,
            Ort::Value& simcc_y,
            const BBox& bbox);
        std::vector<vp_objects::vp_pose_keypoint> decode_heatmap(
            Ort::Value& heatmap,
            const BBox& bbox);
        BBox full_frame_bbox(const cv::Mat& image) const;

    protected:
        virtual void run_infer_combinations(
            const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;

        virtual void postprocess(
            const std::vector<cv::Mat>& raw_outputs,
            const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;

    public:
        vp_rtmpose_onnx_node(
            std::string node_name,
            std::string model_path,
            int input_width = 192,
            int input_height = 256,
            float simcc_split_ratio = 2.0f,
            float score_threshold = 0.0f,
            int infer_every_n_frames = 1,
            int intra_op_num_threads = 0,
            std::string execution_provider = "cpu",
            int device_id = 0,
            bool enable_fp16 = true,
            std::string trt_engine_cache_path = "");
        ~vp_rtmpose_onnx_node();
    };
}

#endif
