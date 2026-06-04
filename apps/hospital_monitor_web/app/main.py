from __future__ import annotations

import asyncio
import json
import os
import shlex
import socket
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles


APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "static"

UDP_HOST = os.getenv("VP_HOSPITAL_UDP_HOST", "127.0.0.1")
UDP_PORT = int(os.getenv("VP_HOSPITAL_UDP_PORT", "9977"))
FRAME_TCP_HOST = os.getenv("VP_HOSPITAL_FRAME_TCP_HOST", "127.0.0.1")
FRAME_TCP_PORT = int(os.getenv("VP_HOSPITAL_FRAME_TCP_PORT", "9978"))
VIDEOPIPE_COMMAND = os.getenv("VP_HOSPITAL_VIDEOPIPE_COMMAND", "").strip()
STREAM_FPS = float(os.getenv("VP_HOSPITAL_STREAM_FPS", "12"))
JPEG_QUALITY = int(os.getenv("VP_HOSPITAL_JPEG_QUALITY", "82"))
JPEG_MAGIC = 0x56504A46
MAX_JPEG_BYTES = int(os.getenv("VP_HOSPITAL_MAX_JPEG_BYTES", str(8 * 1024 * 1024)))

COCO17 = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}


@dataclass(frozen=True)
class MonitorSnapshot:
    running: bool
    state: str
    alert: bool
    confidence: float
    visible_keypoints: int
    reason: str
    frame_index: int | None
    timestamp_s: float | None
    latency_ms: float | None
    box: dict[str, float] | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VideoPipeBridge:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = MonitorSnapshot(
            running=False,
            state="unknown",
            alert=False,
            confidence=0.0,
            visible_keypoints=0,
            reason="waiting for VideoPipe pose JSON",
            frame_index=None,
            timestamp_s=None,
            latency_ms=None,
            box=None,
        )
        self._started_at = time.perf_counter()
        self._last_packet_at: float | None = None
        self._last_frame_at: float | None = None
        self._latest_jpeg: bytes | None = None
        self._latest_jpeg_frame_index: int | None = None
        self._stop = threading.Event()
        self._udp_thread: threading.Thread | None = None
        self._frame_thread: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None

    def start(self) -> bool:
        self._start_udp()
        self._start_frame_receiver()
        if VIDEOPIPE_COMMAND and self._process is None:
            self._process = subprocess.Popen(
                shlex.split(VIDEOPIPE_COMMAND),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                text=True,
            )
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        for thread in (self._udp_thread, self._frame_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=1.5)
        self._udp_thread = None
        self._frame_thread = None
        self._set_snapshot(
            MonitorSnapshot(
                running=False,
                state=self.snapshot().state,
                alert=False,
                confidence=self.snapshot().confidence,
                visible_keypoints=self.snapshot().visible_keypoints,
                reason="monitor stopped",
                frame_index=self.snapshot().frame_index,
                timestamp_s=self.snapshot().timestamp_s,
                latency_ms=self.snapshot().latency_ms,
                box=self.snapshot().box,
            )
        )

    def status(self) -> dict[str, Any]:
        process_running = self._process is not None and self._process.poll() is None
        recent_packet = self._last_packet_at is not None and (time.perf_counter() - self._last_packet_at) < 2.0
        recent_frame = self._last_frame_at is not None and (time.perf_counter() - self._last_frame_at) < 2.0
        return {
            "connected": process_running or recent_packet or recent_frame,
            "source": f"videopipe://pose-udp/{UDP_HOST}:{UDP_PORT};frame-tcp/{FRAME_TCP_HOST}:{FRAME_TCP_PORT}",
            "process_running": process_running,
            "frame_connected": recent_frame,
            "frame_index": self._latest_jpeg_frame_index,
        }

    def snapshot(self) -> MonitorSnapshot:
        with self._lock:
            return self._snapshot

    def _set_snapshot(self, snapshot: MonitorSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot

    def _start_udp(self) -> None:
        if self._udp_thread is not None and self._udp_thread.is_alive():
            return
        self._stop.clear()
        self._udp_thread = threading.Thread(target=self._udp_loop, name="vp-hospital-udp", daemon=True)
        self._udp_thread.start()

    def _udp_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((UDP_HOST, UDP_PORT))
        sock.settimeout(0.5)
        while not self._stop.is_set():
            try:
                payload, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue
            started = time.perf_counter()
            try:
                data = json.loads(payload.decode("utf-8"))
                self._last_packet_at = time.perf_counter()
                self._set_snapshot(_snapshot_from_videopipe(data, started_at=self._started_at, started=started))
            except Exception as exc:
                self._set_snapshot(
                    MonitorSnapshot(
                        running=True,
                        state="unknown",
                        alert=False,
                        confidence=0.0,
                        visible_keypoints=0,
                        reason="invalid VideoPipe pose JSON",
                        frame_index=None,
                        timestamp_s=round(time.perf_counter() - self._started_at, 3),
                        latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
                        box=None,
                        error=str(exc),
                    )
                )
        sock.close()

    def _start_frame_receiver(self) -> None:
        if self._frame_thread is not None and self._frame_thread.is_alive():
            return
        self._stop.clear()
        self._frame_thread = threading.Thread(target=self._frame_loop, name="vp-hospital-frame-tcp", daemon=True)
        self._frame_thread.start()

    def _frame_loop(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((FRAME_TCP_HOST, FRAME_TCP_PORT))
        server.listen(1)
        server.settimeout(0.5)
        try:
            while not self._stop.is_set():
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue
                with conn:
                    conn.settimeout(1.0)
                    while not self._stop.is_set():
                        header = _recv_exact(conn, 12)
                        if header is None:
                            break
                        magic = int.from_bytes(header[0:4], "big")
                        frame_index = int.from_bytes(header[4:8], "big")
                        jpeg_size = int.from_bytes(header[8:12], "big")
                        if magic != JPEG_MAGIC or jpeg_size <= 0 or jpeg_size > MAX_JPEG_BYTES:
                            break
                        jpeg = _recv_exact(conn, jpeg_size)
                        if jpeg is None:
                            break
                        with self._lock:
                            self._latest_jpeg = jpeg
                            self._latest_jpeg_frame_index = frame_index
                        self._last_frame_at = time.perf_counter()
        finally:
            server.close()

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg


def _snapshot_from_videopipe(data: dict[str, Any], *, started_at: float, started: float) -> MonitorSnapshot:
    targets = data.get("targets") or []
    poses = data.get("pose_targets") or []
    target = targets[0] if targets else None
    pose = poses[0] if poses else {}
    keypoints = _keypoints_by_name(pose.get("keypoints") or [])
    visible = sum(1 for point in keypoints.values() if float(point.get("score", 0.0)) >= 0.2)
    box = _box_from_target(target)
    state, confidence, reason = _classify_state(keypoints, box=box)
    return MonitorSnapshot(
        running=True,
        state=state,
        alert=state in {"sitting", "standing"},
        confidence=round(confidence, 4),
        visible_keypoints=visible,
        reason=reason,
        frame_index=int(data.get("frame_index", -1)),
        timestamp_s=round(time.perf_counter() - started_at, 3),
        latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
        box=box,
    )


def _keypoints_by_name(points: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for point in points:
        point_type = int(point.get("type", -1))
        name = COCO17.get(point_type, str(point_type))
        result[name] = {
            "x": float(point.get("x", 0.0)),
            "y": float(point.get("y", 0.0)),
            "score": float(point.get("score", 0.0)),
        }
    return result


def _box_from_target(target: dict[str, Any] | None) -> dict[str, float] | None:
    if not target:
        return None
    x1 = float(target.get("x", 0.0))
    y1 = float(target.get("y", 0.0))
    width = float(target.get("width", 0.0))
    height = float(target.get("height", 0.0))
    return {"x1": x1, "y1": y1, "x2": x1 + width, "y2": y1 + height}


def _classify_state(
    keypoints: dict[str, dict[str, float]],
    *,
    box: dict[str, float] | None,
) -> tuple[str, float, str]:
    if not box or len(keypoints) < 5:
        return "unknown", 0.0, "not enough person pose data"
    width = max(1.0, box["x2"] - box["x1"])
    height = max(1.0, box["y2"] - box["y1"])
    if width / height > 1.2:
        return "lying", 0.8, "person box is mostly horizontal"

    shoulder_y = _avg_y(keypoints, "left_shoulder", "right_shoulder")
    hip_y = _avg_y(keypoints, "left_hip", "right_hip")
    ankle_y = _avg_y(keypoints, "left_ankle", "right_ankle")
    if shoulder_y is None or hip_y is None:
        return "unknown", 0.2, "missing shoulder or hip keypoints"

    torso = hip_y - shoulder_y
    if torso < height * 0.16:
        return "reclining", 0.65, "torso is close to horizontal"
    if ankle_y is not None and (ankle_y - hip_y) > torso * 0.75 and height / width > 1.45:
        return "standing", 0.75, "hips and ankles indicate upright posture"
    return "sitting", 0.7, "upright torso without strong standing evidence"


def _avg_y(keypoints: dict[str, dict[str, float]], left: str, right: str) -> float | None:
    values = [
        keypoints[name]["y"]
        for name in (left, right)
        if name in keypoints and keypoints[name].get("score", 0.0) >= 0.2
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _draw_monitor_overlay(image_bgr: np.ndarray, snapshot: MonitorSnapshot) -> np.ndarray:
    frame = image_bgr.copy()
    state = snapshot.state.upper()
    color = (40, 40, 230) if snapshot.alert else (55, 180, 80)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 74), (20, 24, 28), -1)
    cv2.putText(frame, f"STATE: {state}", (24, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"conf {snapshot.confidence:.2f}  kpts {snapshot.visible_keypoints}",
        (280, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    if snapshot.box:
        cv2.rectangle(
            frame,
            (int(snapshot.box["x1"]), int(snapshot.box["y1"])),
            (int(snapshot.box["x2"]), int(snapshot.box["y2"])),
            color,
            3,
        )
    return frame


def _placeholder_frame(snapshot: MonitorSnapshot) -> np.ndarray:
    frame = np.full((540, 960, 3), (32, 36, 40), dtype=np.uint8)
    cv2.putText(frame, "VideoPipe hospital monitor", (32, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
    cv2.putText(frame, snapshot.reason, (32, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    return _draw_monitor_overlay(frame, snapshot)


bridge = VideoPipeBridge()
app = FastAPI(title="VideoPipe Hospital Monitor", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")


@app.on_event("startup")
async def startup() -> None:
    bridge.start()


@app.on_event("shutdown")
async def shutdown() -> None:
    bridge.stop()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/api/status")
async def status() -> dict[str, Any]:
    return {
        "camera": bridge.status(),
        "monitor": bridge.snapshot().to_dict(),
        "settings": {
            "udp_host": UDP_HOST,
            "udp_port": UDP_PORT,
            "frame_tcp_host": FRAME_TCP_HOST,
            "frame_tcp_port": FRAME_TCP_PORT,
            "videopipe_command": VIDEOPIPE_COMMAND,
        },
    }


@app.get("/api/camera/devices")
async def camera_devices() -> dict[str, Any]:
    return {"providers": [{"name": "videopipe", "devices": [{"id": "videopipe", "display_name": "VideoPipe UDP bridge"}]}]}


@app.post("/api/camera/connect")
async def camera_connect(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    bridge.start()
    return {"connected": True, "status": bridge.status()}


@app.post("/api/camera/disconnect")
async def camera_disconnect() -> dict[str, Any]:
    bridge.stop()
    return {"disconnected": True, "status": bridge.status()}


@app.get("/api/video/stream")
async def video_stream() -> StreamingResponse:
    return StreamingResponse(_mjpeg_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws/monitor")
async def monitor_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json({"camera": bridge.status(), "monitor": bridge.snapshot().to_dict()})
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return


async def _mjpeg_frames() -> AsyncIterator[bytes]:
    delay = 1.0 / max(STREAM_FPS, 1.0)
    while True:
        jpeg = bridge.latest_jpeg()
        if jpeg is None:
            frame = _placeholder_frame(bridge.snapshot())
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                jpeg = encoded.tobytes()
        if jpeg is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        await asyncio.sleep(delay)


def _recv_exact(conn: socket.socket, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        try:
            chunk = conn.recv(remaining)
        except socket.timeout:
            return None
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
