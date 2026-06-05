#!/usr/bin/env bash
set -euo pipefail

export VP_ROOT="${VP_ROOT:-/workspace/VideoPipe}"
export TENSORRT_ROOT="${TENSORRT_ROOT:-/workspace/jetson_tensorrt}"
export VP_BUILD_DIR="${VP_BUILD_DIR:-${VP_ROOT}/build-jetson-sdk}"
export LD_LIBRARY_PATH="${VP_BUILD_DIR}/libs:${VP_BUILD_DIR}/sdk:${TENSORRT_ROOT}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${VP_BUILD_DIR}/python:${PYTHONPATH:-}"

YOLO_ONNX="${VP_YOLO_ONNX:-${VP_ROOT}/vp_data/models/onnx/yolo/yolov8n-person.onnx}"
RTMPOSE_ONNX="${VP_RTMPOSE_ONNX:-${VP_ROOT}/vp_data/models/onnx/rtmpose/rtmpose-small.onnx}"
YOLO_ENGINE="${VP_YOLO_ENGINE:-${VP_ROOT}/vp_data/models/trt/yolo/yolov8n-person.engine}"
RTMPOSE_ENGINE="${VP_RTMPOSE_ENGINE:-${VP_ROOT}/vp_data/models/trt/rtmpose/rtmpose-small.engine}"
LABELS_PATH="${VP_LABELS_PATH:-${VP_ROOT}/vp_data/models/coco_80classes.txt}"

if command -v trtexec >/dev/null 2>&1; then
  TRTEXEC="$(command -v trtexec)"
elif [[ -x /usr/src/tensorrt/bin/trtexec ]]; then
  TRTEXEC="/usr/src/tensorrt/bin/trtexec"
else
  echo "trtexec not found. Check JetPack/TensorRT installation." >&2
  exit 1
fi

mkdir -p "$(dirname "${YOLO_ENGINE}")" "$(dirname "${RTMPOSE_ENGINE}")" "$(dirname "${LABELS_PATH}")"

if [[ ! -f "${LABELS_PATH}" ]]; then
  printf "person\n" > "${LABELS_PATH}"
fi

if [[ ! -f "${YOLO_ENGINE}" ]]; then
  if [[ ! -f "${YOLO_ONNX}" ]]; then
    echo "Missing YOLO ONNX: ${YOLO_ONNX}" >&2
    echo "Mount vp_data with models, for example: -v /host/vp_data:${VP_ROOT}/vp_data" >&2
    exit 1
  fi
  "${TRTEXEC}" \
    --onnx="${YOLO_ONNX}" \
    --saveEngine="${YOLO_ENGINE}" \
    --skipInference
fi

if [[ ! -f "${RTMPOSE_ENGINE}" ]]; then
  if [[ ! -f "${RTMPOSE_ONNX}" ]]; then
    echo "Missing RTMPose ONNX: ${RTMPOSE_ONNX}" >&2
    echo "Mount vp_data with models, for example: -v /host/vp_data:${VP_ROOT}/vp_data" >&2
    exit 1
  fi
  "${TRTEXEC}" \
    --onnx="${RTMPOSE_ONNX}" \
    --saveEngine="${RTMPOSE_ENGINE}" \
    --skipInference
fi

exec "$@"
