#include "vp_yolov8_onnx_detector_node.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>

namespace vp_nodes {
    vp_yolov8_onnx_detector_node::vp_yolov8_onnx_detector_node(
        std::string node_name,
        std::string model_path,
        std::string labels_path,
        int input_width,
        int input_height,
        int batch_size,
        int class_id_offset,
        float score_threshold,
        float nms_threshold,
        int max_detections,
        std::vector<int> class_ids_applied_to):
        vp_primary_infer_node(
            node_name,
            model_path,
            "",
            labels_path,
            input_width,
            input_height,
            batch_size,
            class_id_offset,
            1.0,
            cv::Scalar(0, 0, 0),
            cv::Scalar(1, 1, 1),
            true,
            false),
        score_threshold(score_threshold),
        nms_threshold(nms_threshold),
        max_detections(max_detections),
        class_ids_applied_to(class_ids_applied_to) {
        this->initialized();
    }

    vp_yolov8_onnx_detector_node::~vp_yolov8_onnx_detector_node() {
        deinitialized();
    }

    vp_yolov8_onnx_detector_node::LetterboxMeta vp_yolov8_onnx_detector_node::preprocess_letterbox(
        const cv::Mat& image,
        cv::Mat& blob_to_infer) {
        const int input_w = input_width;
        const int input_h = input_height;
        const int image_w = image.cols;
        const int image_h = image.rows;
        const float scale = std::min(static_cast<float>(input_w) / image_w, static_cast<float>(input_h) / image_h);
        const int resized_w = static_cast<int>(std::round(image_w * scale));
        const int resized_h = static_cast<int>(std::round(image_h * scale));

        cv::Mat resized;
        cv::resize(image, resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_LINEAR);
        cv::Mat canvas(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
        const float pad_x = (input_w - resized_w) / 2.0f;
        const float pad_y = (input_h - resized_h) / 2.0f;
        const int left = static_cast<int>(std::round(pad_x - 0.1f));
        const int top = static_cast<int>(std::round(pad_y - 0.1f));
        resized.copyTo(canvas(cv::Rect(left, top, resized_w, resized_h)));

        cv::Mat rgb;
        cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
        cv::dnn::blobFromImage(rgb, blob_to_infer, 1.0 / 255.0, cv::Size(input_w, input_h), cv::Scalar(), false, false);
        return LetterboxMeta {scale, static_cast<float>(left), static_cast<float>(top), image_w, image_h};
    }

    std::vector<std::pair<cv::Rect, std::pair<int, float>>> vp_yolov8_onnx_detector_node::decode_predictions(
        const cv::Mat& output,
        const LetterboxMeta& meta) {
        cv::Mat predictions;
        if (output.dims == 3) {
            assert(output.size[0] == 1);
            const int rows = output.size[1];
            const int cols = output.size[2];
            cv::Mat raw(rows, cols, CV_32F, const_cast<float*>(output.ptr<float>()));
            if (rows < cols) {
                cv::transpose(raw, predictions);
            }
            else {
                predictions = raw.clone();
            }
        }
        else if (output.dims == 2) {
            const int rows = output.rows;
            const int cols = output.cols;
            cv::Mat raw(rows, cols, CV_32F, const_cast<float*>(output.ptr<float>()));
            if (rows < cols && rows <= 128) {
                cv::transpose(raw, predictions);
            }
            else {
                predictions = raw.clone();
            }
        }
        else {
            throw std::runtime_error("unsupported YOLOv8 output dimensions");
        }

        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;

        for (int i = 0; i < predictions.rows; ++i) {
            const float* row = predictions.ptr<float>(i);
            const int cols = predictions.cols;
            if (cols < 6) {
                continue;
            }

            int class_id = -1;
            float score = 0.0f;
            float x1 = 0.0f;
            float y1 = 0.0f;
            float x2 = 0.0f;
            float y2 = 0.0f;

            if (cols == 6) {
                score = row[4];
                class_id = static_cast<int>(std::round(row[5]));
                x1 = row[0];
                y1 = row[1];
                x2 = row[2];
                y2 = row[3];
            }
            else {
                const float* class_scores = row + 4;
                auto max_iter = std::max_element(class_scores, class_scores + (cols - 4));
                class_id = static_cast<int>(max_iter - class_scores);
                score = *max_iter;
                const float cx = row[0];
                const float cy = row[1];
                const float width = row[2];
                const float height = row[3];
                x1 = cx - width / 2.0f;
                y1 = cy - height / 2.0f;
                x2 = cx + width / 2.0f;
                y2 = cy + height / 2.0f;
            }

            if (!class_ids_applied_to.empty() &&
                std::find(class_ids_applied_to.begin(), class_ids_applied_to.end(), class_id) == class_ids_applied_to.end()) {
                continue;
            }
            if (score < score_threshold) {
                continue;
            }

            x1 = (x1 - meta.pad_x) / meta.scale;
            y1 = (y1 - meta.pad_y) / meta.scale;
            x2 = (x2 - meta.pad_x) / meta.scale;
            y2 = (y2 - meta.pad_y) / meta.scale;

            x1 = std::max(0.0f, std::min(static_cast<float>(meta.image_width), x1));
            y1 = std::max(0.0f, std::min(static_cast<float>(meta.image_height), y1));
            x2 = std::max(0.0f, std::min(static_cast<float>(meta.image_width), x2));
            y2 = std::max(0.0f, std::min(static_cast<float>(meta.image_height), y2));

            cv::Rect box(
                static_cast<int>(std::round(x1)),
                static_cast<int>(std::round(y1)),
                static_cast<int>(std::round(x2 - x1)),
                static_cast<int>(std::round(y2 - y1)));
            if (box.width <= 0 || box.height <= 0) {
                continue;
            }

            boxes.push_back(box);
            scores.push_back(score);
            class_ids.push_back(class_id);
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, score_threshold, nms_threshold, indices);

        std::vector<std::pair<cv::Rect, std::pair<int, float>>> detections;
        detections.reserve(std::min<int>(static_cast<int>(indices.size()), max_detections));
        for (int i = 0; i < indices.size(); ++i) {
            if (i >= max_detections) {
                break;
            }
            const auto idx = indices[i];
            auto box = boxes[idx];
            box.x = std::max(box.x, 0);
            box.y = std::max(box.y, 0);
            box.width = std::min(box.width, meta.image_width - box.x);
            box.height = std::min(box.height, meta.image_height - box.y);
            if (box.width <= 0 || box.height <= 0) {
                continue;
            }
            detections.push_back({box, {class_ids[idx], scores[idx]}});
        }

        return detections;
    }

    void vp_yolov8_onnx_detector_node::run_infer_combinations(
        const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        assert(frame_meta_with_batch.size() == 1);
        auto& frame_meta = frame_meta_with_batch[0];

        std::vector<cv::Mat> mats_to_infer;
        cv::Mat blob_to_infer;
        std::vector<cv::Mat> raw_outputs;

        auto start_time = std::chrono::system_clock::now();

        mats_to_infer.push_back(frame_meta->frame);
        auto prepare_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        start_time = std::chrono::system_clock::now();
        auto meta = preprocess_letterbox(mats_to_infer[0], blob_to_infer);
        auto preprocess_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        start_time = std::chrono::system_clock::now();
        net.setInput(blob_to_infer);
        net.forward(raw_outputs, net.getUnconnectedOutLayersNames());
        auto infer_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        start_time = std::chrono::system_clock::now();
        assert(!raw_outputs.empty());
        auto detections = decode_predictions(raw_outputs[0], meta);
        for (auto& detection: detections) {
            auto& box = detection.first;
            auto& cls = detection.second.first;
            auto& score = detection.second.second;
            auto label = (labels.size() < cls + 1) ? "" : labels[cls];
            auto target = std::make_shared<vp_objects::vp_frame_target>(
                box.x,
                box.y,
                box.width,
                box.height,
                cls + class_id_offset,
                score,
                frame_meta->frame_index,
                frame_meta->channel_index,
                label);
            frame_meta->targets.push_back(target);
        }
        auto postprocess_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        infer_combinations_time_cost(
            mats_to_infer.size(),
            prepare_time.count(),
            preprocess_time.count(),
            infer_time.count(),
            postprocess_time.count());
    }

    void vp_yolov8_onnx_detector_node::postprocess(
        const std::vector<cv::Mat>& raw_outputs,
        const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
    }
}
