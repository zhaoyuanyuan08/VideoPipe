#pragma once

#ifdef VP_WITH_TENSORRT_RUNTIME

#include "../vp_secondary_infer_node.h"
#include "../../objects/vp_frame_target.h"
#include "../../objects/vp_frame_pose_target.h"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <opencv2/imgproc.hpp>

#include <memory>
#include <string>
#include <vector>

namespace vp_nodes {
    // RTMPose secondary node for TensorRT engines generated from ONNX by trtexec.
    class vp_rtmpose_trtexec_secondary_node: public vp_secondary_infer_node
    {
    private:
        class TrtLogger: public nvinfer1::ILogger {
        public:
            void log(Severity severity, const char* msg) noexcept override;
        };

        struct BBox {
            int x1;
            int y1;
            int x2;
            int y2;
        };

        struct TensorOutput {
            std::string name;
            nvinfer1::Dims dims {};
            std::vector<float> data;
        };

        TrtLogger logger;
        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::ICudaEngine* engine = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;
        cudaStream_t stream = nullptr;
        void* input_device = nullptr;
        unsigned char* frame_device = nullptr;
        std::vector<void*> output_devices;
        size_t input_device_bytes = 0;
        size_t frame_device_bytes = 0;
        std::vector<size_t> output_device_bytes;

        std::string input_tensor_name;
        std::vector<std::string> output_tensor_names;
        nvinfer1::Dims input_dims {};

        int input_width;
        int input_height;
        float simcc_split_ratio;
        float score_threshold;
        int infer_every_n_frames;
        int device_id;

        void load_engine(const std::string& engine_path);
        void ensure_device_buffer(void** ptr, size_t& current_bytes, size_t required_bytes, const char* name);
        void upload_frame_cuda(const cv::Mat& image);
        void preprocess_cuda(const cv::Mat& image, const BBox& bbox);
        std::vector<TensorOutput> infer_engine();
        std::vector<vp_objects::vp_pose_keypoint> decode_outputs(
            const std::vector<TensorOutput>& outputs,
            const BBox& bbox);
        std::vector<vp_objects::vp_pose_keypoint> decode_simcc(
            const TensorOutput& simcc_x,
            const TensorOutput& simcc_y,
            const BBox& bbox);
        std::vector<vp_objects::vp_pose_keypoint> decode_heatmap(
            const TensorOutput& heatmap,
            const BBox& bbox);
        BBox target_bbox(
            const vp_objects::vp_frame_target& target,
            const cv::Mat& image) const;
        static size_t dims_volume(const nvinfer1::Dims& dims);
        static bool has_dynamic_dim(const nvinfer1::Dims& dims);

    protected:
        virtual void run_infer_combinations(
            const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;

        virtual void postprocess(
            const std::vector<cv::Mat>& raw_outputs,
            const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;

    public:
        vp_rtmpose_trtexec_secondary_node(
            std::string node_name,
            std::string engine_path,
            int input_width = 192,
            int input_height = 256,
            float simcc_split_ratio = 2.0f,
            float score_threshold = 0.0f,
            int infer_every_n_frames = 1,
            std::vector<int> p_class_ids_applied_to = std::vector<int>{0},
            int min_width_applied_to = 0,
            int min_height_applied_to = 0,
            int crop_padding = 10,
            int device_id = 0);
        ~vp_rtmpose_trtexec_secondary_node();
    };
}

#endif
