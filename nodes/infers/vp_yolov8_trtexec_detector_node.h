#pragma once

#ifdef VP_WITH_TENSORRT_RUNTIME

#include "../vp_primary_infer_node.h"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

namespace vp_nodes {
    // YOLOv8 detector for TensorRT engines generated from ONNX by trtexec.
    class vp_yolov8_trtexec_detector_node: public vp_primary_infer_node
    {
    private:
        class TrtLogger: public nvinfer1::ILogger {
        public:
            void log(Severity severity, const char* msg) noexcept override;
        };

        struct LetterboxMeta {
            float scale;
            float pad_x;
            float pad_y;
            int image_width;
            int image_height;
        };

        TrtLogger logger;
        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::ICudaEngine* engine = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;
        cudaStream_t stream = nullptr;
        void* input_device = nullptr;
        void* output_device = nullptr;
        unsigned char* frame_device = nullptr;
        size_t input_device_bytes = 0;
        size_t output_device_bytes = 0;
        size_t frame_device_bytes = 0;
        std::vector<float> output_host;

        std::string input_tensor_name;
        std::string output_tensor_name;
        nvinfer1::Dims input_dims {};

        float score_threshold;
        float nms_threshold;
        int max_detections;
        int device_id;
        std::vector<int> class_ids_applied_to;

        void load_engine(const std::string& engine_path);
        void ensure_device_buffer(void** ptr, size_t& current_bytes, size_t required_bytes, const char* name);
        LetterboxMeta preprocess_letterbox_cuda(const cv::Mat& image);
        cv::Mat infer_engine();
        std::vector<std::pair<cv::Rect, std::pair<int, float>>> decode_predictions(
            const cv::Mat& output,
            const LetterboxMeta& meta);
        static size_t dims_volume(const nvinfer1::Dims& dims);
        static bool has_dynamic_dim(const nvinfer1::Dims& dims);

    protected:
        virtual void run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
        virtual void postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;

    public:
        vp_yolov8_trtexec_detector_node(
            std::string node_name,
            std::string engine_path,
            std::string labels_path = "",
            int input_width = 640,
            int input_height = 640,
            int batch_size = 1,
            int class_id_offset = 0,
            float score_threshold = 0.35f,
            float nms_threshold = 0.45f,
            int max_detections = 20,
            std::vector<int> class_ids_applied_to = std::vector<int>{0},
            int device_id = 0);
        ~vp_yolov8_trtexec_detector_node();
    };
}

#endif
