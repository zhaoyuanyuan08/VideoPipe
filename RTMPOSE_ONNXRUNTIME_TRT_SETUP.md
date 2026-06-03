# RTMPose ONNXRuntime TensorRT Setup

This note records the environment setup for the RTMPose ONNXRuntime node in
VideoPipe. The route used here is:

```text
VideoPipe C++ node -> ONNXRuntime C++ SDK -> TensorRT EP -> CUDA EP fallback -> CPU fallback
```

Do not enable VideoPipe's original `VP_WITH_TRT` for this route. That flag builds
the existing native TensorRT subprojects. For RTMPose route A, only enable
`VP_WITH_ONNXRUNTIME`.

## 1. Ubuntu x86_64 Setup

Install C++ build dependencies:

```bash
sudo apt update

sudo apt install -y \
  build-essential \
  cmake \
  git \
  pkg-config \
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

Check GPU and CUDA:

```bash
nvidia-smi
nvcc --version
```

Expected on the current workstation:

```text
GPU: NVIDIA GeForce RTX 4070
Driver: 595.71.05
CUDA from nvidia-smi: 13.2
CUDA toolkit from nvcc: 12.4
```

## 2. ONNXRuntime And TensorRT Runtime Under VideoPipe

The current workstation uses these local directories:

```text
/home/yuan/venture/VideoPipe/third_party/onnxruntime-gpu
/home/yuan/venture/VideoPipe/third_party/tensorrt_python
```

ONNXRuntime GPU C++ SDK:

```bash
cd /home/yuan/venture/VideoPipe

mkdir -p third_party/downloads third_party/onnxruntime-gpu

curl -L --fail --retry 3 --retry-delay 2 \
  -o third_party/downloads/onnxruntime-linux-x64-gpu-1.23.2.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-gpu-1.23.2.tgz

tar -xzf third_party/downloads/onnxruntime-linux-x64-gpu-1.23.2.tgz \
  --strip-components=1 \
  -C third_party/onnxruntime-gpu
```

TensorRT/CUDA/cuDNN runtime libraries:

```bash
cd /home/yuan/venture/VideoPipe

mkdir -p third_party/tensorrt_python

uv pip install --target third_party/tensorrt_python \
  tensorrt-cu12-libs==10.9.0.34 \
  nvidia-cudnn-cu12
```

Verify the key files:

```bash
find third_party/onnxruntime-gpu -maxdepth 3 -type f | \
  rg 'onnxruntime_cxx_api.h|libonnxruntime|providers'

find third_party/tensorrt_python -maxdepth 6 -type f | \
  rg 'libnvinfer|libnvonnxparser|libcudnn|libcublas|libcudart|libnvrtc'
```

Expected key libraries:

```text
third_party/onnxruntime-gpu/include/onnxruntime_cxx_api.h
third_party/onnxruntime-gpu/lib/libonnxruntime.so
third_party/onnxruntime-gpu/lib/libonnxruntime_providers_tensorrt.so
third_party/onnxruntime-gpu/lib/libonnxruntime_providers_cuda.so
third_party/tensorrt_python/tensorrt_libs/libnvinfer.so.10
third_party/tensorrt_python/tensorrt_libs/libnvonnxparser.so.10
third_party/tensorrt_python/nvidia/cudnn/lib/libcudnn.so.9
third_party/tensorrt_python/nvidia/cublas/lib/libcublas.so.12
third_party/tensorrt_python/nvidia/cuda_runtime/lib/libcudart.so.12
```

Check provider library dependencies:

```bash
cd /home/yuan/venture/VideoPipe

export VP_ROOT=/home/yuan/venture/VideoPipe
export LD_LIBRARY_PATH=$VP_ROOT/third_party/onnxruntime-gpu/lib:$VP_ROOT/third_party/tensorrt_python/tensorrt_libs:$VP_ROOT/third_party/tensorrt_python/nvidia/cudnn/lib:$VP_ROOT/third_party/tensorrt_python/nvidia/cublas/lib:$VP_ROOT/third_party/tensorrt_python/nvidia/cuda_runtime/lib:$VP_ROOT/third_party/tensorrt_python/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH

ldd third_party/onnxruntime-gpu/lib/libonnxruntime_providers_tensorrt.so
ldd third_party/onnxruntime-gpu/lib/libonnxruntime_providers_cuda.so
```

There should be no `not found` entries.

## 3. Configure And Build VideoPipe

The `apt` install commands can be run from any directory. CMake must either be
run from the VideoPipe source directory, or use absolute `-S` and `-B` paths.

Configure:

```bash
cd /home/yuan/venture/VideoPipe

cmake -S . -B build-rtmpose \
  -DVP_WITH_ONNXRUNTIME=ON \
  -DONNXRUNTIME_INCLUDE_DIR=/home/yuan/venture/VideoPipe/third_party/onnxruntime-gpu/include \
  -DONNXRUNTIME_LIB_DIR=/home/yuan/venture/VideoPipe/third_party/onnxruntime-gpu/lib \
  -DVP_WITH_CUDA=OFF \
  -DVP_WITH_TRT=OFF \
  -DVP_WITH_PADDLE=OFF \
  -DVP_WITH_KAFKA=OFF \
  -DVP_WITH_LLM=OFF \
  -DVP_WITH_FFMPEG=OFF
```

Equivalent absolute-path configure command, runnable from any directory:

```bash
cmake -S /home/yuan/venture/VideoPipe \
  -B /home/yuan/venture/VideoPipe/build-rtmpose \
  -DVP_WITH_ONNXRUNTIME=ON \
  -DONNXRUNTIME_INCLUDE_DIR=/home/yuan/venture/VideoPipe/third_party/onnxruntime-gpu/include \
  -DONNXRUNTIME_LIB_DIR=/home/yuan/venture/VideoPipe/third_party/onnxruntime-gpu/lib \
  -DVP_WITH_CUDA=OFF \
  -DVP_WITH_TRT=OFF \
  -DVP_WITH_PADDLE=OFF \
  -DVP_WITH_KAFKA=OFF \
  -DVP_WITH_LLM=OFF \
  -DVP_WITH_FFMPEG=OFF
```

Build only the RTMPose sample:

```bash
cmake --build build-rtmpose --target rtmpose_onnx_sample -j$(nproc)
```

Equivalent absolute-path build command, runnable from any directory:

```bash
cmake --build /home/yuan/venture/VideoPipe/build-rtmpose \
  --target rtmpose_onnx_sample \
  -j$(nproc)
```

## 4. Run RTMPose Sample

Set runtime library path:

```bash
cd /home/yuan/venture/VideoPipe

export VP_ROOT=/home/yuan/venture/VideoPipe
export LD_LIBRARY_PATH=$VP_ROOT/third_party/onnxruntime-gpu/lib:$VP_ROOT/third_party/tensorrt_python/tensorrt_libs:$VP_ROOT/third_party/tensorrt_python/nvidia/cudnn/lib:$VP_ROOT/third_party/tensorrt_python/nvidia/cublas/lib:$VP_ROOT/third_party/tensorrt_python/nvidia/cuda_runtime/lib:$VP_ROOT/third_party/tensorrt_python/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
```

Run TensorRT EP:

```bash
./build-rtmpose/bin/rtmpose_onnx_sample \
  /home/yuan/venture/vision_fullstack/service_backend/vision_service/models/rtmpose/small/model.onnx \
  ./vp_data/test_video/pose.mp4 \
  tensorrt \
  ./vp_data/trt_engine_cache/rtmpose
```

Run CUDA EP:

```bash
./build-rtmpose/bin/rtmpose_onnx_sample \
  /home/yuan/venture/vision_fullstack/service_backend/vision_service/models/rtmpose/small/model.onnx \
  ./vp_data/test_video/pose.mp4 \
  cuda
```

Run CPU:

```bash
./build-rtmpose/bin/rtmpose_onnx_sample \
  /home/yuan/venture/vision_fullstack/service_backend/vision_service/models/rtmpose/small/model.onnx \
  ./vp_data/test_video/pose.mp4 \
  cpu
```

The first TensorRT run may be slower because TensorRT builds an engine. Re-run
with the same cache directory to measure steady-state speed.

## 5. Jetson Setup

On Jetson, install JetPack first. JetPack provides the matched CUDA, cuDNN,
TensorRT, and driver stack for the board.

Check Jetson version:

```bash
cat /etc/nv_tegra_release
dpkg -l | rg 'nvidia-jetpack|nvinfer|cudnn|cuda'
```

If JetPack components are not installed:

```bash
sudo apt update
sudo apt install -y nvidia-jetpack
```

Install VideoPipe build dependencies:

```bash
sudo apt install -y \
  build-essential \
  cmake \
  git \
  pkg-config \
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

Check Jetson TensorRT headers and libs:

```bash
find /usr -name NvInfer.h
ldconfig -p | rg 'nvinfer|nvonnxparser|cudnn|cublas|cudart'
```

### Jetson ONNXRuntime C++ SDK

Do not use the x86_64 ONNXRuntime tarball on Jetson. Jetson is `aarch64`.

If an official ONNXRuntime GPU aarch64 C++ SDK matching your JetPack/CUDA/TensorRT
exists, install it under:

```text
/home/yuan/venture/VideoPipe/third_party/onnxruntime-gpu
```

If there is no matching prebuilt SDK, build ONNXRuntime on the Jetson:

```bash
cd /home/yuan/venture/VideoPipe
mkdir -p third_party/src
cd third_party/src

git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime
git checkout v1.23.2
git submodule update --init --recursive

./build.sh \
  --config Release \
  --build_shared_lib \
  --parallel \
  --skip_tests \
  --use_cuda \
  --cuda_home /usr/local/cuda \
  --cudnn_home /usr \
  --use_tensorrt \
  --tensorrt_home /usr
```

After build, copy the ONNXRuntime headers and libraries into the same layout used
by VideoPipe:

```bash
cd /home/yuan/venture/VideoPipe

mkdir -p third_party/onnxruntime-gpu/include
mkdir -p third_party/onnxruntime-gpu/lib

cp -a third_party/src/onnxruntime/include/onnxruntime/core/session/*.h \
  third_party/onnxruntime-gpu/include/

cp -a third_party/src/onnxruntime/build/Linux/Release/libonnxruntime*.so* \
  third_party/onnxruntime-gpu/lib/
```

If the build output path differs, locate it:

```bash
find third_party/src/onnxruntime/build -name 'libonnxruntime*.so*'
```

Configure VideoPipe on Jetson:

```bash
cd /home/yuan/venture/VideoPipe

cmake -S . -B build-rtmpose \
  -DVP_WITH_ONNXRUNTIME=ON \
  -DONNXRUNTIME_INCLUDE_DIR=/home/yuan/venture/VideoPipe/third_party/onnxruntime-gpu/include \
  -DONNXRUNTIME_LIB_DIR=/home/yuan/venture/VideoPipe/third_party/onnxruntime-gpu/lib \
  -DVP_WITH_CUDA=OFF \
  -DVP_WITH_TRT=OFF \
  -DVP_WITH_PADDLE=OFF \
  -DVP_WITH_KAFKA=OFF \
  -DVP_WITH_LLM=OFF \
  -DVP_WITH_FFMPEG=OFF

cmake --build build-rtmpose --target rtmpose_onnx_sample -j$(nproc)
```

Jetson runtime library path normally only needs ONNXRuntime because JetPack
installs CUDA/cuDNN/TensorRT system-wide:

```bash
export VP_ROOT=/home/yuan/venture/VideoPipe
export LD_LIBRARY_PATH=$VP_ROOT/third_party/onnxruntime-gpu/lib:$LD_LIBRARY_PATH
```

If `ldd` shows missing TensorRT/CUDA libraries, add the Jetson library paths:

```bash
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Run:

```bash
./build-rtmpose/bin/rtmpose_onnx_sample \
  /home/yuan/venture/vision_fullstack/service_backend/vision_service/models/rtmpose/small/model.onnx \
  ./vp_data/test_video/pose.mp4 \
  tensorrt \
  ./vp_data/trt_engine_cache/rtmpose
```

## 6. Notes

- `uv` is useful for Python dependencies and for installing Python wheels on the
  x86_64 workstation. It does not replace CMake, OpenCV dev headers, GStreamer
  dev headers, or a C++ ONNXRuntime SDK.
- `VP_WITH_TRT=OFF` is intentional for this route.
- TensorRT EP should be followed by CUDA EP fallback. The RTMPose node does this.
- For speed testing, ignore the first TensorRT run because engine building can
  dominate startup time.

## 7. Official References

- ONNXRuntime install: https://onnxruntime.ai/docs/install/
- ONNXRuntime C++ API: https://onnxruntime.ai/docs/get-started/with-cpp.html
- ONNXRuntime TensorRT EP: https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
- ONNXRuntime build with EPs: https://onnxruntime.ai/docs/build/eps.html
- NVIDIA JetPack install: https://docs.nvidia.com/jetson/jetpack/install-jetpack/
