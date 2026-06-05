#!/usr/bin/env bash
set -euo pipefail

export VP_ROOT="${VP_ROOT:-/workspace/VideoPipe}"
export TENSORRT_ROOT="${TENSORRT_ROOT:-/workspace/jetson_tensorrt}"
export LD_LIBRARY_PATH="${TENSORRT_ROOT}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

YOLO_ONNX="${VP_YOLO_ONNX:-${VP_ROOT}/vp_data/models/onnx/yolo/yolov8n-person.onnx}"
RTMPOSE_ONNX="${VP_RTMPOSE_ONNX:-${VP_ROOT}/vp_data/models/onnx/rtmpose/rtmpose-small.onnx}"
YOLO_ENGINE="${VP_YOLO_ENGINE:-${VP_ROOT}/vp_data/models/trt/yolo/yolov8n-person.engine}"
RTMPOSE_ENGINE="${VP_RTMPOSE_ENGINE:-${VP_ROOT}/vp_data/models/trt/rtmpose/rtmpose-small.engine}"
VIDEO_PATH="${VP_VIDEO_PATH:-${VP_ROOT}/vp_data/test_video/pose.mp4}"
LABELS_PATH="${VP_LABELS_PATH:-${VP_ROOT}/vp_data/models/coco_80classes.txt}"
WEB_HOST="${VP_WEB_HOST:-0.0.0.0}"
WEB_PORT="${VP_WEB_PORT:-9301}"
POSE_HOST="${VP_HOSPITAL_UDP_HOST:-127.0.0.1}"
POSE_PORT="${VP_HOSPITAL_UDP_PORT:-9977}"
FRAME_PORT="${VP_HOSPITAL_FRAME_TCP_PORT:-9978}"

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

if [[ ! -f "${VIDEO_PATH}" ]]; then
  echo "Missing video: ${VIDEO_PATH}" >&2
  echo "Mount vp_data with test video, for example: -v /host/vp_data:${VP_ROOT}/vp_data" >&2
  exit 1
fi

cd "${VP_ROOT}/apps/hospital_monitor_web"

uv run --no-sync uvicorn app.main:app \
  --host "${WEB_HOST}" \
  --port "${WEB_PORT}" &
WEB_PID=$!

cleanup() {
  kill "${WEB_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 2

cd "${VP_ROOT}"
exec "${VP_ROOT}/build-jetson-trtexec/bin/yolov8_person_pose_trtexec_sample" \
  "${VIDEO_PATH}" \
  "${YOLO_ENGINE}" \
  "${RTMPOSE_ENGINE}" \
  "${LABELS_PATH}" \
  "${POSE_HOST}" \
  "${POSE_PORT}" \
  "${FRAME_PORT}"
