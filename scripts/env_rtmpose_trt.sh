#!/usr/bin/env bash

export VP_ROOT=/home/yuan/venture/VideoPipe

export LD_LIBRARY_PATH="$VP_ROOT/third_party/onnxruntime-gpu/lib:$VP_ROOT/third_party/tensorrt_python/tensorrt_libs:$VP_ROOT/third_party/tensorrt_python/nvidia/cudnn/lib:$VP_ROOT/third_party/tensorrt_python/nvidia/cublas/lib:$VP_ROOT/third_party/tensorrt_python/nvidia/cuda_runtime/lib:$VP_ROOT/third_party/tensorrt_python/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"

