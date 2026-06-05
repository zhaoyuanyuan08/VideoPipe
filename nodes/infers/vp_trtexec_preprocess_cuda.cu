#ifdef VP_WITH_TENSORRT_RUNTIME

#include "vp_trtexec_preprocess_cuda.h"

#include <cuda_runtime.h>

namespace {
    __device__ unsigned char sample_bgr(
        const unsigned char* src,
        int src_width,
        int src_height,
        int src_step,
        int x,
        int y,
        int channel) {
        if (x < 0 || x >= src_width || y < 0 || y >= src_height) {
            return 114;
        }
        return src[y * src_step + x * 3 + channel];
    }

    __device__ unsigned char sample_bgr_clamped(
        const unsigned char* src,
        int src_width,
        int src_height,
        int src_step,
        int x,
        int y,
        int channel) {
        x = max(0, min(src_width - 1, x));
        y = max(0, min(src_height - 1, y));
        return src[y * src_step + x * 3 + channel];
    }

    __global__ void yolo_letterbox_kernel(
        const unsigned char* src,
        int src_width,
        int src_height,
        int src_step,
        float* dst,
        int dst_width,
        int dst_height,
        float scale,
        float pad_x,
        float pad_y) {
        const int idx = blockDim.x * blockIdx.x + threadIdx.x;
        const int area = dst_width * dst_height;
        if (idx >= area) {
            return;
        }

        const int dx = idx % dst_width;
        const int dy = idx / dst_width;
        const float src_x = (static_cast<float>(dx) - pad_x) / scale;
        const float src_y = (static_cast<float>(dy) - pad_y) / scale;

        float b = 114.0f;
        float g = 114.0f;
        float r = 114.0f;
        if (src_x > -1.0f && src_x < static_cast<float>(src_width) &&
            src_y > -1.0f && src_y < static_cast<float>(src_height)) {
            const int x0 = static_cast<int>(floorf(src_x));
            const int y0 = static_cast<int>(floorf(src_y));
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;
            const float lx = src_x - static_cast<float>(x0);
            const float ly = src_y - static_cast<float>(y0);
            const float hx = 1.0f - lx;
            const float hy = 1.0f - ly;
            const float w00 = hx * hy;
            const float w01 = lx * hy;
            const float w10 = hx * ly;
            const float w11 = lx * ly;

            const float b00 = sample_bgr(src, src_width, src_height, src_step, x0, y0, 0);
            const float g00 = sample_bgr(src, src_width, src_height, src_step, x0, y0, 1);
            const float r00 = sample_bgr(src, src_width, src_height, src_step, x0, y0, 2);
            const float b01 = sample_bgr(src, src_width, src_height, src_step, x1, y0, 0);
            const float g01 = sample_bgr(src, src_width, src_height, src_step, x1, y0, 1);
            const float r01 = sample_bgr(src, src_width, src_height, src_step, x1, y0, 2);
            const float b10 = sample_bgr(src, src_width, src_height, src_step, x0, y1, 0);
            const float g10 = sample_bgr(src, src_width, src_height, src_step, x0, y1, 1);
            const float r10 = sample_bgr(src, src_width, src_height, src_step, x0, y1, 2);
            const float b11 = sample_bgr(src, src_width, src_height, src_step, x1, y1, 0);
            const float g11 = sample_bgr(src, src_width, src_height, src_step, x1, y1, 1);
            const float r11 = sample_bgr(src, src_width, src_height, src_step, x1, y1, 2);

            b = b00 * w00 + b01 * w01 + b10 * w10 + b11 * w11;
            g = g00 * w00 + g01 * w01 + g10 * w10 + g11 * w11;
            r = r00 * w00 + r01 * w01 + r10 * w10 + r11 * w11;
        }

        dst[idx] = r / 255.0f;
        dst[area + idx] = g / 255.0f;
        dst[area * 2 + idx] = b / 255.0f;
    }

    __global__ void rtmpose_crop_kernel(
        const unsigned char* src,
        int src_width,
        int src_height,
        int src_step,
        float* dst,
        int dst_width,
        int dst_height,
        int crop_x1,
        int crop_y1,
        int crop_x2,
        int crop_y2) {
        const int idx = blockDim.x * blockIdx.x + threadIdx.x;
        const int area = dst_width * dst_height;
        if (idx >= area) {
            return;
        }

        const int dx = idx % dst_width;
        const int dy = idx / dst_width;
        const int crop_width = max(1, crop_x2 - crop_x1);
        const int crop_height = max(1, crop_y2 - crop_y1);
        const float src_x = (static_cast<float>(dx) + 0.5f) * static_cast<float>(crop_width) / static_cast<float>(dst_width) - 0.5f + crop_x1;
        const float src_y = (static_cast<float>(dy) + 0.5f) * static_cast<float>(crop_height) / static_cast<float>(dst_height) - 0.5f + crop_y1;
        const int x0 = static_cast<int>(floorf(src_x));
        const int y0 = static_cast<int>(floorf(src_y));
        const int x1 = x0 + 1;
        const int y1 = y0 + 1;
        const float lx = src_x - static_cast<float>(x0);
        const float ly = src_y - static_cast<float>(y0);
        const float hx = 1.0f - lx;
        const float hy = 1.0f - ly;
        const float w00 = hx * hy;
        const float w01 = lx * hy;
        const float w10 = hx * ly;
        const float w11 = lx * ly;

        const float b00 = sample_bgr_clamped(src, src_width, src_height, src_step, x0, y0, 0);
        const float g00 = sample_bgr_clamped(src, src_width, src_height, src_step, x0, y0, 1);
        const float r00 = sample_bgr_clamped(src, src_width, src_height, src_step, x0, y0, 2);
        const float b01 = sample_bgr_clamped(src, src_width, src_height, src_step, x1, y0, 0);
        const float g01 = sample_bgr_clamped(src, src_width, src_height, src_step, x1, y0, 1);
        const float r01 = sample_bgr_clamped(src, src_width, src_height, src_step, x1, y0, 2);
        const float b10 = sample_bgr_clamped(src, src_width, src_height, src_step, x0, y1, 0);
        const float g10 = sample_bgr_clamped(src, src_width, src_height, src_step, x0, y1, 1);
        const float r10 = sample_bgr_clamped(src, src_width, src_height, src_step, x0, y1, 2);
        const float b11 = sample_bgr_clamped(src, src_width, src_height, src_step, x1, y1, 0);
        const float g11 = sample_bgr_clamped(src, src_width, src_height, src_step, x1, y1, 1);
        const float r11 = sample_bgr_clamped(src, src_width, src_height, src_step, x1, y1, 2);

        const float b = b00 * w00 + b01 * w01 + b10 * w10 + b11 * w11;
        const float g = g00 * w00 + g01 * w01 + g10 * w10 + g11 * w11;
        const float r = r00 * w00 + r01 * w01 + r10 * w10 + r11 * w11;

        dst[idx] = ((r / 255.0f) - 0.485f) / 0.229f;
        dst[area + idx] = ((g / 255.0f) - 0.456f) / 0.224f;
        dst[area * 2 + idx] = ((b / 255.0f) - 0.406f) / 0.225f;
    }
}

namespace vp_nodes {
    void vp_yolo_letterbox_preprocess_cuda(
        const unsigned char* src_bgr,
        int src_width,
        int src_height,
        int src_step,
        float* dst_chw,
        int dst_width,
        int dst_height,
        float scale,
        float pad_x,
        float pad_y,
        cudaStream_t stream) {
        const int jobs = dst_width * dst_height;
        const int threads = 256;
        const int blocks = (jobs + threads - 1) / threads;
        yolo_letterbox_kernel<<<blocks, threads, 0, stream>>>(
            src_bgr,
            src_width,
            src_height,
            src_step,
            dst_chw,
            dst_width,
            dst_height,
            scale,
            pad_x,
            pad_y);
    }

    void vp_rtmpose_crop_preprocess_cuda(
        const unsigned char* src_bgr,
        int src_width,
        int src_height,
        int src_step,
        float* dst_chw,
        int dst_width,
        int dst_height,
        int crop_x1,
        int crop_y1,
        int crop_x2,
        int crop_y2,
        cudaStream_t stream) {
        const int jobs = dst_width * dst_height;
        const int threads = 256;
        const int blocks = (jobs + threads - 1) / threads;
        rtmpose_crop_kernel<<<blocks, threads, 0, stream>>>(
            src_bgr,
            src_width,
            src_height,
            src_step,
            dst_chw,
            dst_width,
            dst_height,
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2);
    }
}

#endif
