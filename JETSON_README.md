# Jetson Native Commands

## 1. Get Code

```bash
mkdir -p ~/venture
cd ~/venture
git clone <your-github-repo-url> VideoPipe
cd ~/venture/VideoPipe
```

If the repo already exists:

```bash
cd ~/venture/VideoPipe
git pull
```

## 2. Install System Packages

```bash
sudo apt update

sudo apt install -y \
  build-essential \
  cmake \
  git \
  pkg-config \
  curl \
  ca-certificates \
  python3 \
  python3-dev \
  python3-pip \
  python3-opencv \
  libopencv-dev \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev \
  libgstrtspserver-1.0-dev \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-libav
```

Check JetPack/TensorRT:

```bash
cat /etc/nv_tegra_release
dpkg -l | grep -E 'nvidia-jetpack|nvinfer|cudnn|cuda'
```

Install `uv` if missing:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

## 3. TensorRT Paths

```bash
cd ~/venture/VideoPipe

mkdir -p third_party/jetson_tensorrt
ln -sfn /usr/include/aarch64-linux-gnu third_party/jetson_tensorrt/include
ln -sfn /usr/lib/aarch64-linux-gnu third_party/jetson_tensorrt/lib

export VP_ROOT=$PWD
export TENSORRT_ROOT=$VP_ROOT/third_party/jetson_tensorrt
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

```bash
test -f "$TENSORRT_ROOT/include/NvInferRuntime.h" && echo "NvInferRuntime.h OK"
find "$TENSORRT_ROOT/lib" -maxdepth 1 -name 'libnvinfer.so*' -print
```

```bash
if command -v trtexec >/dev/null 2>&1; then
  export TRTEXEC=$(command -v trtexec)
else
  export TRTEXEC=/usr/src/tensorrt/bin/trtexec
fi

"$TRTEXEC" --help | head
```

## 4. Put Models And Test Video

```bash
cd ~/venture/VideoPipe
export VP_ROOT=$PWD

mkdir -p \
  "$VP_ROOT/vp_data/models/onnx/yolo" \
  "$VP_ROOT/vp_data/models/onnx/rtmpose" \
  "$VP_ROOT/vp_data/models/trt/yolo" \
  "$VP_ROOT/vp_data/models/trt/rtmpose" \
  "$VP_ROOT/vp_data/test_video"
```

Put files here:

```text
~/venture/VideoPipe/vp_data/models/onnx/yolo/yolov8n-person.onnx
~/venture/VideoPipe/vp_data/models/onnx/rtmpose/rtmpose-small.onnx
~/venture/VideoPipe/vp_data/test_video/pose.mp4
```

Create labels:

```bash
printf "person\n" > "$VP_ROOT/vp_data/models/coco_80classes.txt"
```

## 5. Generate Jetson Engines

```bash
cd ~/venture/VideoPipe
export VP_ROOT=$PWD

"$TRTEXEC" \
  --onnx="$VP_ROOT/vp_data/models/onnx/yolo/yolov8n-person.onnx" \
  --saveEngine="$VP_ROOT/vp_data/models/trt/yolo/yolov8n-person.engine" \
  --skipInference
```

```bash
"$TRTEXEC" \
  --onnx="$VP_ROOT/vp_data/models/onnx/rtmpose/rtmpose-small.onnx" \
  --saveEngine="$VP_ROOT/vp_data/models/trt/rtmpose/rtmpose-small.engine" \
  --skipInference
```

```bash
ls -lh \
  "$VP_ROOT/vp_data/models/trt/yolo/yolov8n-person.engine" \
  "$VP_ROOT/vp_data/models/trt/rtmpose/rtmpose-small.engine"
```

## 6. Build SDK And Sample

```bash
cd ~/venture/VideoPipe
export VP_ROOT=$PWD
export TENSORRT_ROOT=$VP_ROOT/third_party/jetson_tensorrt
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

cmake -S . -B build-jetson-sdk \
  -DVP_WITH_TENSORRT_RUNTIME=ON \
  -DVP_BUILD_SDK=ON \
  -DTENSORRT_ROOT=$TENSORRT_ROOT \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DVP_WITH_ONNXRUNTIME=OFF \
  -DVP_WITH_CUDA=OFF \
  -DVP_WITH_TRT=OFF \
  -DVP_WITH_PADDLE=OFF \
  -DVP_WITH_KAFKA=OFF \
  -DVP_WITH_LLM=OFF \
  -DVP_WITH_FFMPEG=OFF

cmake --build build-jetson-sdk \
  --target videopipe_sdk_cpp videopipe_sdk_file_sample videopipe_sdk \
  -j$(nproc)
```

## 7. Run C++ Sample

```bash
cd ~/venture/VideoPipe
export VP_ROOT=$PWD
export TENSORRT_ROOT=$VP_ROOT/third_party/jetson_tensorrt
export LD_LIBRARY_PATH=$VP_ROOT/build-jetson-sdk/libs:$TENSORRT_ROOT/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

./build-jetson-sdk/bin/videopipe_sdk_file_sample \
  "$VP_ROOT/vp_data/test_video/pose.mp4" \
  "$VP_ROOT/vp_data/models/trt/yolo/yolov8n-person.engine" \
  "$VP_ROOT/vp_data/models/trt/rtmpose/rtmpose-small.engine" \
  "$VP_ROOT/vp_data/models/coco_80classes.txt" \
  100 \
  0.15 \
  jetson_hw
```

CPU decode fallback:

```bash
./build-jetson-sdk/bin/videopipe_sdk_file_sample \
  "$VP_ROOT/vp_data/test_video/pose.mp4" \
  "$VP_ROOT/vp_data/models/trt/yolo/yolov8n-person.engine" \
  "$VP_ROOT/vp_data/models/trt/rtmpose/rtmpose-small.engine" \
  "$VP_ROOT/vp_data/models/coco_80classes.txt" \
  100 \
  0.15 \
  avdec_h264
```

## 8. Use Python SDK

```bash
cd ~/venture/VideoPipe
export VP_ROOT=$PWD
export TENSORRT_ROOT=$VP_ROOT/third_party/jetson_tensorrt
export LD_LIBRARY_PATH=$VP_ROOT/build-jetson-sdk/libs:$TENSORRT_ROOT/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$VP_ROOT/build-jetson-sdk/sdk/python:$PYTHONPATH
```

```bash
uv venv --system-site-packages
uv pip install numpy
```

```bash
uv run --no-sync python - <<'PY'
from videopipe_sdk import Pipeline

pipe = Pipeline(
    yolo_engine="vp_data/models/trt/yolo/yolov8n-person.engine",
    pose_engine="vp_data/models/trt/rtmpose/rtmpose-small.engine",
    labels="vp_data/models/coco_80classes.txt",
    score_threshold=0.15,
    decoder="jetson_hw",
    enable_frame_output=True,
)

pipe.start_file("vp_data/test_video/pose.mp4")
for _ in range(10):
    result = pipe.read_result(timeout_ms=3000)
    if result is None:
        print("timeout")
        break
    print(result["frame_index"], len(result["targets"]), len(result["poses"]))
pipe.stop()
PY
```

RTSP:

```bash
uv run --no-sync python - <<'PY'
from videopipe_sdk import Pipeline

pipe = Pipeline(
    yolo_engine="vp_data/models/trt/yolo/yolov8n-person.engine",
    pose_engine="vp_data/models/trt/rtmpose/rtmpose-small.engine",
    labels="vp_data/models/coco_80classes.txt",
    score_threshold=0.15,
    decoder="jetson_hw",
    enable_frame_output=False,
)

pipe.start_rtsp("rtsp://user:pass@host/path")
while True:
    result = pipe.read_result(timeout_ms=3000)
    if result is not None:
        print(result["channel_index"], result["frame_index"], len(result["targets"]), len(result["poses"]))
PY
```

## 9. Quick Checks

```bash
gst-inspect-1.0 nvv4l2decoder
gst-inspect-1.0 nvvidconv
```

```bash
python3 - <<'PY'
import cv2
print(cv2.__version__)
PY
```

```bash
ldd build-jetson-sdk/libs/libvideo_pipe.so | grep -E 'nvinfer|cudart|opencv|gst'
```

## 10. Common Fixes

```bash
export VP_ROOT=$HOME/venture/VideoPipe
export TENSORRT_ROOT=$VP_ROOT/third_party/jetson_tensorrt
export LD_LIBRARY_PATH=$VP_ROOT/build-jetson-sdk/libs:$TENSORRT_ROOT/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$VP_ROOT/build-jetson-sdk/sdk/python:$PYTHONPATH
```

```bash
rm -rf build-jetson-sdk
```

```bash
find vp_data/models/trt -name '*.engine' -delete
```
