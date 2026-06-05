#pragma once

#ifdef VP_WITH_TENSORRT_RUNTIME

#include <cuda_runtime_api.h>

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
        cudaStream_t stream);

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
        cudaStream_t stream);
}

#endif
