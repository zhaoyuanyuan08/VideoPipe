#pragma once

#include "../vp_primary_infer_node.h"

namespace vp_nodes {
    // YOLOv8 ONNX detector with letterbox preprocessing and person-class filtering.
    class vp_yolov8_onnx_detector_node: public vp_primary_infer_node
    {
    private:
        struct LetterboxMeta {
            float scale;
            float pad_x;
            float pad_y;
            int image_width;
            int image_height;
        };

        float score_threshold;
        float nms_threshold;
        int max_detections;
        std::vector<int> class_ids_applied_to;

        LetterboxMeta preprocess_letterbox(const cv::Mat& image, cv::Mat& blob_to_infer);
        std::vector<std::pair<cv::Rect, std::pair<int, float>>> decode_predictions(
            const cv::Mat& output,
            const LetterboxMeta& meta);

    protected:
        virtual void run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;
        virtual void postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;

    public:
        vp_yolov8_onnx_detector_node(
            std::string node_name,
            std::string model_path,
            std::string labels_path = "",
            int input_width = 640,
            int input_height = 640,
            int batch_size = 1,
            int class_id_offset = 0,
            float score_threshold = 0.35f,
            float nms_threshold = 0.45f,
            int max_detections = 20,
            std::vector<int> class_ids_applied_to = std::vector<int>{0});
        ~vp_yolov8_onnx_detector_node();
    };
}
