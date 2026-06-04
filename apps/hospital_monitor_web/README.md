# VideoPipe Hospital Monitor Web

Standalone hospital monitor web sidecar for VideoPipe person-pose pipelines.

It is intentionally decoupled from `vision_fullstack`.

## Data Flow

```text
VideoPipe C++ pipeline
  -> vp_hospital_pose_json_udp_broker_node
  -> UDP JSON
  -> vp_hospital_frame_jpeg_tcp_broker_node
  -> TCP JPEG frames
  -> FastAPI sidecar
  -> /api/status, /api/video/stream, /ws/monitor
  -> browser UI
```

The browser-facing API mirrors the hospital use-case UI:

```text
GET /api/status
GET /api/video/stream
WS  /ws/monitor
```

## Start Web Sidecar

```bash
cd /home/yuan/venture/VideoPipe/apps/hospital_monitor_web
uv run uvicorn app.main:app --host 127.0.0.1 --port 9301
```

Open:

```text
http://127.0.0.1:9301
```

## Run VideoPipe Producer

From another terminal:

```bash
cd /home/yuan/venture/VideoPipe
export TENSORRT_ROOT=/home/yuan/venture/VideoPipe/tensorRT/TensorRT-11.0.0.114
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH

./build-trtexec-yolo/bin/yolov8_person_pose_trtexec_sample \
  ./vp_data/test_video/pose.mp4 \
  ./vp_data/models/trt/yolo/yolov8n-person.engine \
  ./vp_data/models/trt/rtmpose/rtmpose-small.engine \
  ./vp_data/models/coco_80classes.txt \
  127.0.0.1 \
  9977 \
  9978
```

## Optional Auto-Start

The sidecar can start VideoPipe as a child process:

```bash
export TENSORRT_ROOT=/home/yuan/venture/VideoPipe/tensorRT/TensorRT-11.0.0.114
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH
export VP_HOSPITAL_VIDEOPIPE_COMMAND="/home/yuan/venture/VideoPipe/build-trtexec-yolo/bin/yolov8_person_pose_trtexec_sample /home/yuan/venture/VideoPipe/vp_data/test_video/pose.mp4 /home/yuan/venture/VideoPipe/vp_data/models/trt/yolo/yolov8n-person.engine /home/yuan/venture/VideoPipe/vp_data/models/trt/rtmpose/rtmpose-small.engine /home/yuan/venture/VideoPipe/vp_data/models/coco_80classes.txt 127.0.0.1 9977 9978"
uv run uvicorn app.main:app --host 127.0.0.1 --port 9301
```

For production, prefer a process supervisor or container entrypoint over a long shell command.
