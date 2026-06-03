#include "vp_rtmpose_onnx_node.h"

#ifdef VP_WITH_ONNXRUNTIME

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cmath>
#include <exception>
#include <unordered_map>

namespace vp_nodes {
    namespace {
        std::string lower_string(std::string value) {
            std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
                return static_cast<char>(std::tolower(ch));
            });
            return value;
        }
    }

    vp_rtmpose_onnx_node::vp_rtmpose_onnx_node(
        std::string node_name,
        std::string model_path,
        int input_width,
        int input_height,
        float simcc_split_ratio,
        float score_threshold,
        int infer_every_n_frames,
        int intra_op_num_threads,
        std::string execution_provider,
        int device_id,
        bool enable_fp16,
        std::string trt_engine_cache_path):
        vp_primary_infer_node(node_name, ""),
        env(ORT_LOGGING_LEVEL_WARNING, node_name.c_str()),
        input_width(input_width),
        input_height(input_height),
        simcc_split_ratio(simcc_split_ratio),
        score_threshold(score_threshold),
        infer_every_n_frames(std::max(1, infer_every_n_frames)),
        device_id(device_id),
        enable_fp16(enable_fp16),
        execution_provider(lower_string(execution_provider)),
        trt_engine_cache_path(trt_engine_cache_path) {
        if (intra_op_num_threads > 0) {
            session_options.SetIntraOpNumThreads(intra_op_num_threads);
        }
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        configure_execution_provider();
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name_alloc = session->GetInputNameAllocated(0, allocator);
        input_name = input_name_alloc.get();

        auto output_count = session->GetOutputCount();
        output_names.reserve(output_count);
        output_name_ptrs.reserve(output_count);
        for (size_t i = 0; i < output_count; ++i) {
            auto output_name_alloc = session->GetOutputNameAllocated(i, allocator);
            output_names.emplace_back(output_name_alloc.get());
        }
        for (auto& name: output_names) {
            output_name_ptrs.push_back(name.c_str());
        }

        this->initialized();
    }

    vp_rtmpose_onnx_node::~vp_rtmpose_onnx_node() {
        deinitialized();
    }

    void vp_rtmpose_onnx_node::configure_execution_provider() {
        if (execution_provider == "cpu" || execution_provider.empty()) {
            VP_INFO(vp_utils::string_format("[%s] use ONNXRuntime CPU execution provider.", node_name.c_str()));
            return;
        }

        if (execution_provider == "tensorrt" || execution_provider == "trt") {
            auto trt_added = append_tensorrt_provider();
            auto cuda_added = append_cuda_provider();
            if (!trt_added && !cuda_added) {
                VP_WARN(vp_utils::string_format(
                    "[%s] TensorRT/CUDA providers are unavailable, fallback to ONNXRuntime CPU execution provider.",
                    node_name.c_str()));
            }
            return;
        }

        if (execution_provider == "cuda" || execution_provider == "gpu") {
            if (!append_cuda_provider()) {
                VP_WARN(vp_utils::string_format(
                    "[%s] CUDA provider is unavailable, fallback to ONNXRuntime CPU execution provider.",
                    node_name.c_str()));
            }
            return;
        }

        VP_WARN(vp_utils::string_format(
            "[%s] unknown ONNXRuntime execution provider [%s], fallback to CPU.",
            node_name.c_str(),
            execution_provider.c_str()));
    }

    bool vp_rtmpose_onnx_node::append_tensorrt_provider() {
        try {
            Ort::TensorRTProviderOptions trt_options;
            std::unordered_map<std::string, std::string> options {
                {"device_id", std::to_string(device_id)},
                {"trt_fp16_enable", enable_fp16 ? "1" : "0"}
            };
            if (!trt_engine_cache_path.empty()) {
                options["trt_engine_cache_enable"] = "1";
                options["trt_engine_cache_path"] = trt_engine_cache_path;
            }
            trt_options.Update(options);
            session_options.AppendExecutionProvider_TensorRT_V2(*trt_options);
            VP_INFO(vp_utils::string_format(
                "[%s] use ONNXRuntime TensorRT execution provider. device_id=%d, fp16=%d",
                node_name.c_str(),
                device_id,
                enable_fp16 ? 1 : 0));
            return true;
        }
        catch (const Ort::Exception& ex) {
            VP_WARN(vp_utils::string_format(
                "[%s] append TensorRT provider failed: %s",
                node_name.c_str(),
                ex.what()));
        }
        catch (const std::exception& ex) {
            VP_WARN(vp_utils::string_format(
                "[%s] append TensorRT provider failed: %s",
                node_name.c_str(),
                ex.what()));
        }
        return false;
    }

    bool vp_rtmpose_onnx_node::append_cuda_provider() {
        try {
            Ort::CUDAProviderOptions cuda_options;
            std::unordered_map<std::string, std::string> options {
                {"device_id", std::to_string(device_id)}
            };
            cuda_options.Update(options);
            session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);
            VP_INFO(vp_utils::string_format(
                "[%s] use ONNXRuntime CUDA execution provider. device_id=%d",
                node_name.c_str(),
                device_id));
            return true;
        }
        catch (const Ort::Exception& ex) {
            VP_WARN(vp_utils::string_format(
                "[%s] append CUDA provider failed: %s",
                node_name.c_str(),
                ex.what()));
        }
        catch (const std::exception& ex) {
            VP_WARN(vp_utils::string_format(
                "[%s] append CUDA provider failed: %s",
                node_name.c_str(),
                ex.what()));
        }
        return false;
    }

    vp_rtmpose_onnx_node::BBox vp_rtmpose_onnx_node::full_frame_bbox(const cv::Mat& image) const {
        return BBox {0, 0, image.cols, image.rows};
    }

    std::vector<float> vp_rtmpose_onnx_node::preprocess(const cv::Mat& image, const BBox& bbox) {
        auto x1 = std::max(0, std::min(image.cols - 1, bbox.x1));
        auto y1 = std::max(0, std::min(image.rows - 1, bbox.y1));
        auto x2 = std::max(0, std::min(image.cols, bbox.x2));
        auto y2 = std::max(0, std::min(image.rows, bbox.y2));
        assert(x2 > x1 && y2 > y1);

        cv::Mat crop = image(cv::Rect(x1, y1, x2 - x1, y2 - y1));
        cv::Mat resized;
        cv::resize(crop, resized, cv::Size(input_width, input_height), 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

        const std::array<float, 3> mean = {0.485f, 0.456f, 0.406f};
        const std::array<float, 3> stddev = {0.229f, 0.224f, 0.225f};
        std::vector<float> tensor(3 * input_height * input_width);
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < input_height; ++y) {
                for (int x = 0; x < input_width; ++x) {
                    float value = resized.at<cv::Vec3b>(y, x)[c] / 255.0f;
                    tensor[c * input_height * input_width + y * input_width + x] = (value - mean[c]) / stddev[c];
                }
            }
        }
        return tensor;
    }

    std::vector<vp_objects::vp_pose_keypoint> vp_rtmpose_onnx_node::decode_simcc(
        Ort::Value& simcc_x,
        Ort::Value& simcc_y,
        const BBox& bbox) {
        auto x_info = simcc_x.GetTensorTypeAndShapeInfo();
        auto y_info = simcc_y.GetTensorTypeAndShapeInfo();
        auto x_shape = x_info.GetShape();
        auto y_shape = y_info.GetShape();
        assert(x_shape.size() == 2 || x_shape.size() == 3);
        assert(y_shape.size() == 2 || y_shape.size() == 3);

        int keypoint_count = static_cast<int>(x_shape.size() == 3 ? x_shape[1] : x_shape[0]);
        int x_length = static_cast<int>(x_shape.size() == 3 ? x_shape[2] : x_shape[1]);
        int y_length = static_cast<int>(y_shape.size() == 3 ? y_shape[2] : y_shape[1]);
        const float* x_data = simcc_x.GetTensorData<float>();
        const float* y_data = simcc_y.GetTensorData<float>();

        float crop_width = std::max(1.0f, static_cast<float>(bbox.x2 - bbox.x1));
        float crop_height = std::max(1.0f, static_cast<float>(bbox.y2 - bbox.y1));
        std::vector<vp_objects::vp_pose_keypoint> keypoints;
        keypoints.reserve(keypoint_count);

        for (int k = 0; k < keypoint_count; ++k) {
            const float* x_row = x_data + k * x_length;
            const float* y_row = y_data + k * y_length;
            auto x_iter = std::max_element(x_row, x_row + x_length);
            auto y_iter = std::max_element(y_row, y_row + y_length);
            int x_index = static_cast<int>(x_iter - x_row);
            int y_index = static_cast<int>(y_iter - y_row);
            float score = (*x_iter + *y_iter) * 0.5f;
            int x = -1;
            int y = -1;
            if (score >= score_threshold) {
                x = static_cast<int>(std::round(bbox.x1 + ((x_index / simcc_split_ratio) / input_width) * crop_width));
                y = static_cast<int>(std::round(bbox.y1 + ((y_index / simcc_split_ratio) / input_height) * crop_height));
            }
            keypoints.push_back(vp_objects::vp_pose_keypoint {k, x, y, score});
        }
        return keypoints;
    }

    std::vector<vp_objects::vp_pose_keypoint> vp_rtmpose_onnx_node::decode_heatmap(
        Ort::Value& heatmap,
        const BBox& bbox) {
        auto info = heatmap.GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        assert(shape.size() == 3 || shape.size() == 4);

        int keypoint_count = static_cast<int>(shape.size() == 4 ? shape[1] : shape[0]);
        int heatmap_height = static_cast<int>(shape.size() == 4 ? shape[2] : shape[1]);
        int heatmap_width = static_cast<int>(shape.size() == 4 ? shape[3] : shape[2]);
        const float* data = heatmap.GetTensorData<float>();
        float crop_width = std::max(1.0f, static_cast<float>(bbox.x2 - bbox.x1));
        float crop_height = std::max(1.0f, static_cast<float>(bbox.y2 - bbox.y1));
        int area = heatmap_width * heatmap_height;
        std::vector<vp_objects::vp_pose_keypoint> keypoints;
        keypoints.reserve(keypoint_count);

        for (int k = 0; k < keypoint_count; ++k) {
            const float* plane = data + k * area;
            auto max_iter = std::max_element(plane, plane + area);
            int index = static_cast<int>(max_iter - plane);
            float score = *max_iter;
            int x = -1;
            int y = -1;
            if (score >= score_threshold) {
                float hx = static_cast<float>(index % heatmap_width);
                float hy = static_cast<float>(index / heatmap_width);
                x = static_cast<int>(std::round(bbox.x1 + (hx / std::max(1.0f, static_cast<float>(heatmap_width - 1))) * crop_width));
                y = static_cast<int>(std::round(bbox.y1 + (hy / std::max(1.0f, static_cast<float>(heatmap_height - 1))) * crop_height));
            }
            keypoints.push_back(vp_objects::vp_pose_keypoint {k, x, y, score});
        }
        return keypoints;
    }

    std::vector<vp_objects::vp_pose_keypoint> vp_rtmpose_onnx_node::decode_outputs(
        std::vector<Ort::Value>& outputs,
        const BBox& bbox) {
        if (outputs.size() >= 2) {
            return decode_simcc(outputs[0], outputs[1], bbox);
        }
        if (outputs.size() == 1) {
            return decode_heatmap(outputs[0], bbox);
        }
        return {};
    }

    void vp_rtmpose_onnx_node::run_infer_combinations(
        const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        for (auto& frame_meta: frame_meta_with_batch) {
            if (infer_every_n_frames > 1 && frame_meta->frame_index >= 0 &&
                frame_meta->frame_index % infer_every_n_frames != 0) {
                continue;
            }
            auto bbox = full_frame_bbox(frame_meta->frame);
            auto tensor = preprocess(frame_meta->frame, bbox);
            std::array<int64_t, 4> input_shape = {1, 3, input_height, input_width};
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            auto input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                tensor.data(),
                tensor.size(),
                input_shape.data(),
                input_shape.size());
            const char* input_names[] = {input_name.c_str()};
            auto outputs = session->Run(
                Ort::RunOptions{nullptr},
                input_names,
                &input_tensor,
                1,
                output_name_ptrs.data(),
                output_name_ptrs.size());
            auto keypoints = decode_outputs(outputs, bbox);
            if (!keypoints.empty()) {
                frame_meta->pose_targets.push_back(
                    std::make_shared<vp_objects::vp_frame_pose_target>(
                        vp_objects::vp_pose_type::yolov8_pose_17,
                        keypoints));
            }
        }
    }

    void vp_rtmpose_onnx_node::postprocess(
        const std::vector<cv::Mat>& raw_outputs,
        const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
    }
}

#endif
