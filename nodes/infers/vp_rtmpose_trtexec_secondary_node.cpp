#ifdef VP_WITH_TENSORRT_RUNTIME

#include "vp_rtmpose_trtexec_secondary_node.h"
#include "vp_trtexec_preprocess_cuda.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdexcept>

namespace {
    void check_cuda(cudaError_t status, const char* what) {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
        }
    }

    nvinfer1::Dims make_static_input_dims(const nvinfer1::Dims& dims, int h, int w) {
        nvinfer1::Dims result = dims;
        if (result.nbDims == 4) {
            result.d[0] = result.d[0] < 0 ? 1 : result.d[0];
            result.d[1] = result.d[1] < 0 ? 3 : result.d[1];
            result.d[2] = result.d[2] < 0 ? h : result.d[2];
            result.d[3] = result.d[3] < 0 ? w : result.d[3];
        }
        return result;
    }

    int pose_dim_at(const nvinfer1::Dims& dims, int index_without_batch, int index_with_batch) {
        return dims.nbDims == 3 ? dims.d[index_with_batch] : dims.d[index_without_batch];
    }
}

namespace vp_nodes {
    void vp_rtmpose_trtexec_secondary_node::TrtLogger::log(Severity severity, const char* msg) noexcept {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }

    vp_rtmpose_trtexec_secondary_node::vp_rtmpose_trtexec_secondary_node(
        std::string node_name,
        std::string engine_path,
        int input_width,
        int input_height,
        float simcc_split_ratio,
        float score_threshold,
        int infer_every_n_frames,
        std::vector<int> p_class_ids_applied_to,
        int min_width_applied_to,
        int min_height_applied_to,
        int crop_padding,
        int device_id):
        vp_secondary_infer_node(
            node_name,
            "",
            "",
            "",
            input_width,
            input_height,
            1,
            p_class_ids_applied_to,
            min_width_applied_to,
            min_height_applied_to,
            crop_padding),
        input_width(input_width),
        input_height(input_height),
        simcc_split_ratio(simcc_split_ratio),
        score_threshold(score_threshold),
        infer_every_n_frames(std::max(1, infer_every_n_frames)),
        device_id(device_id) {
        load_engine(engine_path);
        check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
        check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
        this->initialized();
    }

    vp_rtmpose_trtexec_secondary_node::~vp_rtmpose_trtexec_secondary_node() {
        deinitialized();
        if (frame_device != nullptr) {
            cudaFree(frame_device);
            frame_device = nullptr;
        }
        if (input_device != nullptr) {
            cudaFree(input_device);
            input_device = nullptr;
        }
        for (auto* ptr: output_devices) {
            if (ptr != nullptr) {
                cudaFree(ptr);
            }
        }
        output_devices.clear();
        output_device_bytes.clear();
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
        delete context;
        delete engine;
        delete runtime;
    }

    void vp_rtmpose_trtexec_secondary_node::load_engine(const std::string& engine_path) {
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good()) {
            throw std::runtime_error("failed to open TensorRT engine: " + engine_path);
        }
        std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        runtime = nvinfer1::createInferRuntime(logger);
        if (runtime == nullptr) {
            throw std::runtime_error("failed to create TensorRT runtime");
        }
        engine = runtime->deserializeCudaEngine(data.data(), data.size());
        if (engine == nullptr) {
            throw std::runtime_error("failed to deserialize TensorRT engine: " + engine_path);
        }
        context = engine->createExecutionContext();
        if (context == nullptr) {
            throw std::runtime_error("failed to create TensorRT execution context");
        }

        for (int i = 0; i < engine->getNbIOTensors(); ++i) {
            const char* name = engine->getIOTensorName(i);
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                input_tensor_name = name;
            }
            else if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
                output_tensor_names.emplace_back(name);
            }
        }
        if (input_tensor_name.empty() || output_tensor_names.empty()) {
            throw std::runtime_error("TensorRT RTMPose engine must have one input and at least one output tensor");
        }
        if (engine->getTensorDataType(input_tensor_name.c_str()) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("vp_rtmpose_trtexec_secondary_node supports FP32 input tensors only");
        }
        for (auto& name: output_tensor_names) {
            if (engine->getTensorDataType(name.c_str()) != nvinfer1::DataType::kFLOAT) {
                throw std::runtime_error("vp_rtmpose_trtexec_secondary_node supports FP32 output tensors only");
            }
        }

        input_dims = make_static_input_dims(engine->getTensorShape(input_tensor_name.c_str()), input_height, input_width);
        if (input_dims.nbDims != 4 || input_dims.d[0] != 1 || input_dims.d[1] != 3) {
            throw std::runtime_error("expected RTMPose TensorRT input shape [1,3,H,W]");
        }
        this->input_height = input_dims.d[2];
        this->input_width = input_dims.d[3];
        input_height = input_dims.d[2];
        input_width = input_dims.d[3];
    }

    vp_rtmpose_trtexec_secondary_node::BBox vp_rtmpose_trtexec_secondary_node::target_bbox(
        const vp_objects::vp_frame_target& target,
        const cv::Mat& image) const {
        auto x1 = target.x;
        auto y1 = target.y;
        auto x2 = target.x + target.width;
        auto y2 = target.y + target.height;
        if (crop_padding != 0) {
            x1 -= crop_padding;
            y1 -= crop_padding;
            x2 += crop_padding;
            y2 += crop_padding;
        }
        x1 = std::max(0, std::min(image.cols - 1, x1));
        y1 = std::max(0, std::min(image.rows - 1, y1));
        x2 = std::max(0, std::min(image.cols, x2));
        y2 = std::max(0, std::min(image.rows, y2));
        return BBox {x1, y1, x2, y2};
    }

    void vp_rtmpose_trtexec_secondary_node::ensure_device_buffer(
        void** ptr,
        size_t& current_bytes,
        size_t required_bytes,
        const char* name) {
        if (current_bytes >= required_bytes && *ptr != nullptr) {
            return;
        }
        if (*ptr != nullptr) {
            check_cuda(cudaFree(*ptr), name);
            *ptr = nullptr;
            current_bytes = 0;
        }
        check_cuda(cudaMalloc(ptr, required_bytes), name);
        current_bytes = required_bytes;
    }

    void vp_rtmpose_trtexec_secondary_node::upload_frame_cuda(const cv::Mat& image) {
        if (image.empty()) {
            throw std::runtime_error("RTMPose TensorRT input frame is empty");
        }
        if (image.type() != CV_8UC3) {
            throw std::runtime_error("RTMPose TensorRT CUDA preprocess expects CV_8UC3 BGR frames");
        }
        const size_t frame_bytes = static_cast<size_t>(image.step[0]) * image.rows;
        ensure_device_buffer(reinterpret_cast<void**>(&frame_device), frame_device_bytes, frame_bytes, "cudaMalloc RTMPose frame");
        check_cuda(cudaMemcpyAsync(frame_device, image.data, frame_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync RTMPose frame");
    }

    void vp_rtmpose_trtexec_secondary_node::preprocess_cuda(const cv::Mat& image, const BBox& bbox) {
        auto x1 = std::max(0, std::min(image.cols - 1, bbox.x1));
        auto y1 = std::max(0, std::min(image.rows - 1, bbox.y1));
        auto x2 = std::max(0, std::min(image.cols, bbox.x2));
        auto y2 = std::max(0, std::min(image.rows, bbox.y2));
        assert(x2 > x1 && y2 > y1);

        const size_t input_bytes = dims_volume(input_dims) * sizeof(float);
        ensure_device_buffer(&input_device, input_device_bytes, input_bytes, "cudaMalloc RTMPose input");
        vp_rtmpose_crop_preprocess_cuda(
            frame_device,
            image.cols,
            image.rows,
            static_cast<int>(image.step[0]),
            static_cast<float*>(input_device),
            input_width,
            input_height,
            x1,
            y1,
            x2,
            y2,
            stream);
        check_cuda(cudaGetLastError(), "RTMPose CUDA preprocess launch");
    }

    std::vector<vp_rtmpose_trtexec_secondary_node::TensorOutput> vp_rtmpose_trtexec_secondary_node::infer_engine() {
        if (has_dynamic_dim(engine->getTensorShape(input_tensor_name.c_str()))) {
            if (!context->setInputShape(input_tensor_name.c_str(), input_dims)) {
                throw std::runtime_error("failed to set TensorRT RTMPose input shape");
            }
        }

        std::vector<TensorOutput> outputs;
        outputs.reserve(output_tensor_names.size());
        for (auto& name: output_tensor_names) {
            auto dims = context->getTensorShape(name.c_str());
            if (has_dynamic_dim(dims)) {
                throw std::runtime_error("TensorRT RTMPose output shape is still dynamic after setting input shape");
            }
            auto count = dims_volume(dims);
            TensorOutput output;
            output.name = name;
            output.dims = dims;
            output.data.resize(count);
            outputs.push_back(std::move(output));
        }

        if (output_devices.size() < outputs.size()) {
            output_devices.resize(outputs.size(), nullptr);
            output_device_bytes.resize(outputs.size(), 0);
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
            ensure_device_buffer(
                &output_devices[i],
                output_device_bytes[i],
                outputs[i].data.size() * sizeof(float),
                "cudaMalloc RTMPose output");
        }

        if (!context->setTensorAddress(input_tensor_name.c_str(), input_device)) {
            throw std::runtime_error("failed to set RTMPose input tensor address");
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (!context->setTensorAddress(outputs[i].name.c_str(), output_devices[i])) {
                throw std::runtime_error("failed to set RTMPose output tensor address");
            }
        }
        if (!context->enqueueV3(stream)) {
            throw std::runtime_error("TensorRT RTMPose enqueueV3 failed");
        }

        for (size_t i = 0; i < outputs.size(); ++i) {
            check_cuda(cudaMemcpyAsync(outputs[i].data.data(), output_devices[i], outputs[i].data.size() * sizeof(float), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync RTMPose output");
        }
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize RTMPose");
        return outputs;
    }

    std::vector<vp_objects::vp_pose_keypoint> vp_rtmpose_trtexec_secondary_node::decode_simcc(
        const TensorOutput& simcc_x,
        const TensorOutput& simcc_y,
        const BBox& bbox) {
        assert(simcc_x.dims.nbDims == 2 || simcc_x.dims.nbDims == 3);
        assert(simcc_y.dims.nbDims == 2 || simcc_y.dims.nbDims == 3);

        int keypoint_count = pose_dim_at(simcc_x.dims, 0, 1);
        int x_length = pose_dim_at(simcc_x.dims, 1, 2);
        int y_length = pose_dim_at(simcc_y.dims, 1, 2);
        const float* x_data = simcc_x.data.data();
        const float* y_data = simcc_y.data.data();

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

    std::vector<vp_objects::vp_pose_keypoint> vp_rtmpose_trtexec_secondary_node::decode_heatmap(
        const TensorOutput& heatmap,
        const BBox& bbox) {
        assert(heatmap.dims.nbDims == 3 || heatmap.dims.nbDims == 4);

        int keypoint_count = heatmap.dims.nbDims == 4 ? heatmap.dims.d[1] : heatmap.dims.d[0];
        int heatmap_height = heatmap.dims.nbDims == 4 ? heatmap.dims.d[2] : heatmap.dims.d[1];
        int heatmap_width = heatmap.dims.nbDims == 4 ? heatmap.dims.d[3] : heatmap.dims.d[2];
        const float* data = heatmap.data.data();
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

    std::vector<vp_objects::vp_pose_keypoint> vp_rtmpose_trtexec_secondary_node::decode_outputs(
        const std::vector<TensorOutput>& outputs,
        const BBox& bbox) {
        if (outputs.size() >= 2) {
            return decode_simcc(outputs[0], outputs[1], bbox);
        }
        if (outputs.size() == 1) {
            return decode_heatmap(outputs[0], bbox);
        }
        return {};
    }

    void vp_rtmpose_trtexec_secondary_node::run_infer_combinations(
        const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        assert(frame_meta_with_batch.size() == 1);
        auto& frame_meta = frame_meta_with_batch[0];

        if (infer_every_n_frames > 1 && frame_meta->frame_index >= 0 &&
            frame_meta->frame_index % infer_every_n_frames != 0) {
            return;
        }

        auto start_time = std::chrono::system_clock::now();
        long preprocess_ms = 0;
        long infer_ms = 0;
        long postprocess_ms = 0;
        int infer_count = 0;
        bool frame_uploaded = false;

        for (auto& target: frame_meta->targets) {
            if (!need_apply(target->primary_class_id, target->width, target->height)) {
                continue;
            }

            auto bbox = target_bbox(*target, frame_meta->frame);
            if (bbox.x2 <= bbox.x1 || bbox.y2 <= bbox.y1) {
                continue;
            }

            if (!frame_uploaded) {
                upload_frame_cuda(frame_meta->frame);
                frame_uploaded = true;
            }

            start_time = std::chrono::system_clock::now();
            preprocess_cuda(frame_meta->frame, bbox);
            preprocess_ms += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count();

            start_time = std::chrono::system_clock::now();
            auto outputs = infer_engine();
            infer_ms += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count();

            start_time = std::chrono::system_clock::now();
            auto keypoints = decode_outputs(outputs, bbox);
            if (!keypoints.empty()) {
                frame_meta->pose_targets.push_back(
                    std::make_shared<vp_objects::vp_frame_pose_target>(
                        vp_objects::vp_pose_type::yolov8_pose_17,
                        keypoints));
            }
            postprocess_ms += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count();
            ++infer_count;
        }

        infer_combinations_time_cost(std::max(1, infer_count), 0, preprocess_ms, infer_ms, postprocess_ms);
    }

    void vp_rtmpose_trtexec_secondary_node::postprocess(
        const std::vector<cv::Mat>& raw_outputs,
        const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
    }

    size_t vp_rtmpose_trtexec_secondary_node::dims_volume(const nvinfer1::Dims& dims) {
        size_t volume = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
            if (dims.d[i] < 0) {
                throw std::runtime_error("cannot calculate volume for dynamic TensorRT dims");
            }
            volume *= static_cast<size_t>(dims.d[i]);
        }
        return volume;
    }

    bool vp_rtmpose_trtexec_secondary_node::has_dynamic_dim(const nvinfer1::Dims& dims) {
        for (int i = 0; i < dims.nbDims; ++i) {
            if (dims.d[i] < 0) {
                return true;
            }
        }
        return false;
    }
}

#endif
