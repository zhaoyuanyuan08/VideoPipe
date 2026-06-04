# Jetson Build And Run Guide

This guide starts from a fresh Jetson checkout and runs the VideoPipe hospital
monitor frontend with the full TensorRT pipeline:

```text
file_src
  -> YOLOv8 TensorRT engine
  -> RTMPose TensorRT engine
  -> pose JSON UDP 9977
  -> pose OSD
  -> JPEG TCP 9978
  -> FastAPI /api/video/stream + /ws/monitor
  -> browser
```

Yes, on Jetson you should rebuild the C++ binary and regenerate TensorRT
engines on the Jetson. Do not reuse x86 build directories or x86 TensorRT
engines.

## 1. Pull Code

```bash
cd ~/venture
git clone <your-github-repo-url> VideoPipe
cd ~/venture/VideoPipe
```

If the repo already exists:

```bash
cd ~/venture/VideoPipe
git pull
```

## 2. Install Jetson System Dependencies

JetPack should provide CUDA, cuDNN, TensorRT, and the NVIDIA driver stack.

Check JetPack/TensorRT:

```bash
cat /etc/nv_tegra_release
dpkg -l | grep -E 'nvidia-jetpack|nvinfer|cudnn|cuda'
```

Install base build/runtime packages:

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
  python3-pip \
  python3-venv \
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

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

## 3. Set TensorRT Root For CMake

On Jetson, TensorRT headers and libs usually live under system directories.
Create a small local wrapper directory so the VideoPipe CMake option
`TENSORRT_ROOT` works consistently:

```bash
cd ~/venture/VideoPipe

mkdir -p third_party/jetson_tensorrt
ln -sfn /usr/include/aarch64-linux-gnu third_party/jetson_tensorrt/include
ln -sfn /usr/lib/aarch64-linux-gnu third_party/jetson_tensorrt/lib

export TENSORRT_ROOT=$PWD/third_party/jetson_tensorrt
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Verify:

```bash
test -f "$TENSORRT_ROOT/include/NvInfer.h" && echo "NvInfer.h OK"
find "$TENSORRT_ROOT/lib" -maxdepth 1 -name 'libnvinfer.so*' -print
```

Find `trtexec`:

```bash
if command -v trtexec >/dev/null 2>&1; then
  export TRTEXEC=$(command -v trtexec)
else
  export TRTEXEC=/usr/src/tensorrt/bin/trtexec
fi

"$TRTEXEC" --help | head
```

## 4. Put Models And Test Video In Place

Git should not store model/video/engine artifacts. Put them under `vp_data`
on the Jetson:

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

Copy these files from your model source or artifact storage:

```text
$VP_ROOT/vp_data/models/onnx/yolo/yolov8n-person.onnx
$VP_ROOT/vp_data/models/onnx/rtmpose/rtmpose-small.onnx
$VP_ROOT/vp_data/test_video/pose.mp4
```

These ONNX paths are local Jetson paths used by this guide. On the current
x86 workstation the source files were:

```text
/home/yuan/venture/vision_fullstack/vision_service/model_assets/yolo/yolov8n-person/model.onnx
/home/yuan/venture/vision_fullstack/vision_service/model_assets/rtmpose/small/model.onnx
```

Copy or rename them to the `vp_data/models/onnx/...` paths above before running
`trtexec`, or change the `--onnx=` argument to your actual ONNX location.

Create a minimal label file if you do not have COCO labels:

```bash
mkdir -p vp_data/models
printf "person\n" > "$VP_ROOT/vp_data/models/coco_80classes.txt"
```

## 5. Generate TensorRT Engines On Jetson

Generate YOLO engine:

```bash
cd ~/venture/VideoPipe
export VP_ROOT=$PWD

"$TRTEXEC" \
  --onnx="$VP_ROOT/vp_data/models/onnx/yolo/yolov8n-person.onnx" \
  --saveEngine="$VP_ROOT/vp_data/models/trt/yolo/yolov8n-person.engine" \
  --skipInference
```

Generate RTMPose engine:

```bash
"$TRTEXEC" \
  --onnx="$VP_ROOT/vp_data/models/onnx/rtmpose/rtmpose-small.onnx" \
  --saveEngine="$VP_ROOT/vp_data/models/trt/rtmpose/rtmpose-small.engine" \
  --skipInference
```

Verify:

```bash
ls -lh \
  "$VP_ROOT/vp_data/models/trt/yolo/yolov8n-person.engine" \
  "$VP_ROOT/vp_data/models/trt/rtmpose/rtmpose-small.engine"
```

If TensorRT asks for dynamic shapes, inspect the ONNX input name with the
`trtexec` log and rerun with explicit shapes. For the current RTMPose model,
the expected input shape is:

```text
1x3x256x192
```

## 6. Build VideoPipe C++ Sample

Important: use `VP_WITH_TENSORRT_RUNTIME=ON`, keep old `VP_WITH_TRT=OFF`.
The old VideoPipe TRT subproject uses older TensorRT APIs and should not be
used for this path.

```bash
cd ~/venture/VideoPipe

cmake -S . -B build-jetson-trtexec \
  -DVP_WITH_TENSORRT_RUNTIME=ON \
  -DTENSORRT_ROOT=$TENSORRT_ROOT \
  -DVP_WITH_ONNXRUNTIME=OFF \
  -DVP_WITH_CUDA=OFF \
  -DVP_WITH_TRT=OFF \
  -DVP_WITH_PADDLE=OFF \
  -DVP_WITH_KAFKA=OFF \
  -DVP_WITH_LLM=OFF \
  -DVP_WITH_FFMPEG=OFF

cmake --build build-jetson-trtexec \
  --target yolov8_person_pose_trtexec_sample \
  -j$(nproc)
```

Verify:

```bash
ls -lh build-jetson-trtexec/bin/yolov8_person_pose_trtexec_sample
```

## 7. Install Web Frontend Dependencies

Use Jetson system OpenCV (`python3-opencv`) and avoid building OpenCV from pip.

```bash
cd ~/venture/VideoPipe/apps/hospital_monitor_web

uv venv --system-site-packages
uv pip install fastapi "uvicorn[standard]" numpy

uv run --no-sync python -c "import fastapi, uvicorn, cv2, numpy; print('web deps ok')"
```

## 8. Run Frontend And VideoPipe

### Terminal A: Start The Web Sidecar

Use `0.0.0.0` if you want to open the UI from another computer on the same
network.

```bash
cd ~/venture/VideoPipe/apps/hospital_monitor_web

uv run --no-sync uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 9301
```

### Terminal B: Start VideoPipe Producer

```bash
cd ~/venture/VideoPipe
export VP_ROOT=$PWD

export TENSORRT_ROOT=$VP_ROOT/third_party/jetson_tensorrt
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

./build-jetson-trtexec/bin/yolov8_person_pose_trtexec_sample \
  "$VP_ROOT/vp_data/test_video/pose.mp4" \
  "$VP_ROOT/vp_data/models/trt/yolo/yolov8n-person.engine" \
  "$VP_ROOT/vp_data/models/trt/rtmpose/rtmpose-small.engine" \
  "$VP_ROOT/vp_data/models/coco_80classes.txt" \
  127.0.0.1 \
  9977 \
  9978
```

Open on the Jetson:

```text
http://127.0.0.1:9301
```

Open from another machine:

```bash
hostname -I
```

Then open:

```text
http://<JETSON_IP>:9301
```

## 9. Optional: Let FastAPI Start VideoPipe

This starts the web server and launches VideoPipe as a child process. Use this
after manual Terminal A/B testing works.

```bash
cd ~/venture/VideoPipe/apps/hospital_monitor_web

export VP_ROOT=$HOME/venture/VideoPipe
export TENSORRT_ROOT=$VP_ROOT/third_party/jetson_tensorrt
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

export VP_HOSPITAL_VIDEOPIPE_COMMAND="$VP_ROOT/build-jetson-trtexec/bin/yolov8_person_pose_trtexec_sample $VP_ROOT/vp_data/test_video/pose.mp4 $VP_ROOT/vp_data/models/trt/yolo/yolov8n-person.engine $VP_ROOT/vp_data/models/trt/rtmpose/rtmpose-small.engine $VP_ROOT/vp_data/models/coco_80classes.txt 127.0.0.1 9977 9978"

uv run --no-sync uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 9301
```

## 10. What To Expect

Successful VideoPipe logs should include:

```text
hospital pose json udp broker -> 127.0.0.1:9977
hospital frame jpeg tcp broker -> 127.0.0.1:9978
connected frame jpeg tcp broker -> 127.0.0.1:9978
```

The browser should show the hospital monitor UI with real MJPEG frames from
VideoPipe and monitor state updates over WebSocket.

## 11. Common Issues

Port already used:

```bash
sudo ss -ltnp | grep -E ':9301|:9977|:9978'
```

Frontend opens on Jetson but not from another computer:

```bash
uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port 9301
```

TensorRT library not found:

```bash
export TENSORRT_ROOT=$HOME/venture/VideoPipe/third_party/jetson_tensorrt
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Engine fails to load:

```text
Regenerate the .engine on the Jetson. TensorRT engines are tied to TensorRT,
CUDA, plugins, GPU architecture, and sometimes input shapes.
```

No real video, only placeholder:

```text
Start VideoPipe producer after the web sidecar is listening. The TCP frame
receiver listens on 9978, and VideoPipe connects to it.
```
