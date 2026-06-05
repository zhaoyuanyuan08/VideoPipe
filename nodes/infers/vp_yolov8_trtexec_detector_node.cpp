#ifdef VP_WITH_TENSORRT_RUNTIME

#include "vp_yolov8_trtexec_detector_node.h"
#include "vp_trtexec_preprocess_cuda.h"

#include <algorithm>
#include <fstream>
#include <cstring>
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
}

namespace vp_nodes {
    void vp_yolov8_trtexec_detector_node::TrtLogger::log(Severity severity, const char* msg) noexcept {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }

    vp_yolov8_trtexec_detector_node::vp_yolov8_trtexec_detector_node(
        std::string node_name,
        std::string engine_path,
        std::string labels_path,
        int input_width,
        int input_height,
        int batch_size,
        int class_id_offset,
        float score_threshold,
        float nms_threshold,
        int max_detections,
        std::vector<int> class_ids_applied_to,
        int device_id):
        vp_primary_infer_node(
            node_name,
            "",
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
        device_id(device_id),
        class_ids_applied_to(class_ids_applied_to) {
        if (batch_size != 1) {
            throw std::runtime_error("vp_yolov8_trtexec_detector_node supports batch_size=1 only");
        }
        load_engine(engine_path);
        check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
        check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
        this->initialized();
    }

    vp_yolov8_trtexec_detector_node::~vp_yolov8_trtexec_detector_node() {
        deinitialized();
        if (frame_device != nullptr) {
            cudaFree(frame_device);
            frame_device = nullptr;
        }
        if (input_device != nullptr) {
            cudaFree(input_device);
            input_device = nullptr;
        }
        if (output_device != nullptr) {
            cudaFree(output_device);
            output_device = nullptr;
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
        delete context;
        delete engine;
        delete runtime;
    }

    void vp_yolov8_trtexec_detector_node::load_engine(const std::string& engine_path) {
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
            else if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT && output_tensor_name.empty()) {
                output_tensor_name = name;
            }
        }
        if (input_tensor_name.empty() || output_tensor_name.empty()) {
            throw std::runtime_error("TensorRT engine must have one input and at least one output tensor");
        }
        if (engine->getTensorDataType(input_tensor_name.c_str()) != nvinfer1::DataType::kFLOAT ||
            engine->getTensorDataType(output_tensor_name.c_str()) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("vp_yolov8_trtexec_detector_node currently supports FP32 input/output tensors only");
        }

        input_dims = engine->getTensorShape(input_tensor_name.c_str());
        input_dims = make_static_input_dims(input_dims, input_height, input_width);
        if (input_dims.nbDims != 4 || input_dims.d[0] != 1 || input_dims.d[1] != 3) {
            throw std::runtime_error("expected YOLOv8 TensorRT input shape [1,3,H,W]");
        }
        input_height = input_dims.d[2];
        input_width = input_dims.d[3];
        this->input_height = input_dims.d[2];
        this->input_width = input_dims.d[3];
    }

    void vp_yolov8_trtexec_detector_node::ensure_device_buffer(
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

    vp_yolov8_trtexec_detector_node::LetterboxMeta vp_yolov8_trtexec_detector_node::preprocess_letterbox_cuda(
        const cv::Mat& image) {
        if (image.empty()) {
            throw std::runtime_error("YOLOv8 TensorRT input frame is empty");
        }
        if (image.type() != CV_8UC3) {
            throw std::runtime_error("YOLOv8 TensorRT CUDA preprocess expects CV_8UC3 BGR frames");
        }
        const int image_w = image.cols;
        const int image_h = image.rows;
        const float scale = std::min(static_cast<float>(input_width) / image_w, static_cast<float>(input_height) / image_h);
        const int resized_w = static_cast<int>(std::round(image_w * scale));
        const int resized_h = static_cast<int>(std::round(image_h * scale));
        const float pad_x = (input_width - resized_w) / 2.0f;
        const float pad_y = (input_height - resized_h) / 2.0f;
        const int left = static_cast<int>(std::round(pad_x - 0.1f));
        const int top = static_cast<int>(std::round(pad_y - 0.1f));

        const size_t frame_bytes = static_cast<size_t>(image.step[0]) * image.rows;
        const size_t input_bytes = dims_volume(input_dims) * sizeof(float);
        ensure_device_buffer(reinterpret_cast<void**>(&frame_device), frame_device_bytes, frame_bytes, "cudaMalloc YOLO frame");
        ensure_device_buffer(&input_device, input_device_bytes, input_bytes, "cudaMalloc YOLO input");
        check_cuda(cudaMemcpyAsync(frame_device, image.data, frame_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync YOLO frame");
        vp_yolo_letterbox_preprocess_cuda(
            frame_device,
            image_w,
            image_h,
            static_cast<int>(image.step[0]),
            static_cast<float*>(input_device),
            input_width,
            input_height,
            scale,
            static_cast<float>(left),
            static_cast<float>(top),
            stream);
        check_cuda(cudaGetLastError(), "YOLO CUDA preprocess launch");
        return LetterboxMeta {scale, static_cast<float>(left), static_cast<float>(top), image_w, image_h};
    }

    cv::Mat vp_yolov8_trtexec_detector_node::infer_engine() {
        if (has_dynamic_dim(engine->getTensorShape(input_tensor_name.c_str()))) {
            if (!context->setInputShape(input_tensor_name.c_str(), input_dims)) {
                throw std::runtime_error("failed to set TensorRT input shape");
            }
        }

        auto output_dims = context->getTensorShape(output_tensor_name.c_str());
        if (has_dynamic_dim(output_dims)) {
            throw std::runtime_error("TensorRT output shape is still dynamic after setting input shape");
        }
        const size_t output_count = dims_volume(output_dims);
        ensure_device_buffer(&output_device, output_device_bytes, output_count * sizeof(float), "cudaMalloc YOLO output");

        if (!context->setTensorAddress(input_tensor_name.c_str(), input_device) ||
            !context->setTensorAddress(output_tensor_name.c_str(), output_device)) {
            throw std::runtime_error("failed to set TensorRT tensor address");
        }
        if (!context->enqueueV3(stream)) {
            throw std::runtime_error("TensorRT enqueueV3 failed");
        }

        output_host.resize(output_count);
        check_cuda(cudaMemcpyAsync(output_host.data(), output_device, output_count * sizeof(float), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync output");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        std::vector<int> sizes;
        sizes.reserve(output_dims.nbDims);
        for (int i = 0; i < output_dims.nbDims; ++i) {
            sizes.push_back(output_dims.d[i]);
        }
        cv::Mat output_mat(output_dims.nbDims, sizes.data(), CV_32F);
        std::memcpy(output_mat.ptr<float>(), output_host.data(), output_count * sizeof(float));
        return output_mat;
    }

    std::vector<std::pair<cv::Rect, std::pair<int, float>>> vp_yolov8_trtexec_detector_node::decode_predictions(
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
            throw std::runtime_error("unsupported YOLOv8 TensorRT output dimensions");
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

    void vp_yolov8_trtexec_detector_node::run_infer_combinations(
        const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        assert(frame_meta_with_batch.size() == 1);
        auto& frame_meta = frame_meta_with_batch[0];

        auto start_time = std::chrono::system_clock::now();
        auto prepare_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        start_time = std::chrono::system_clock::now();
        auto meta = preprocess_letterbox_cuda(frame_meta->frame);
        auto preprocess_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        start_time = std::chrono::system_clock::now();
        auto output = infer_engine();
        auto infer_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

        start_time = std::chrono::system_clock::now();
        auto detections = decode_predictions(output, meta);
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

        infer_combinations_time_cost(1, prepare_time.count(), preprocess_time.count(), infer_time.count(), postprocess_time.count());
    }

    void vp_yolov8_trtexec_detector_node::postprocess(
        const std::vector<cv::Mat>& raw_outputs,
        const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
    }

    size_t vp_yolov8_trtexec_detector_node::dims_volume(const nvinfer1::Dims& dims) {
        size_t volume = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
            if (dims.d[i] < 0) {
                throw std::runtime_error("cannot calculate volume for dynamic TensorRT dims");
            }
            volume *= static_cast<size_t>(dims.d[i]);
        }
        return volume;
    }

    bool vp_yolov8_trtexec_detector_node::has_dynamic_dim(const nvinfer1::Dims& dims) {
        for (int i = 0; i < dims.nbDims; ++i) {
            if (dims.d[i] < 0) {
                return true;
            }
        }
        return false;
    }
}

#endif
