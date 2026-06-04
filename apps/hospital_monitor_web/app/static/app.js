const ids = {
  alert: document.getElementById("alert"),
  state: document.getElementById("state"),
  confidence: document.getElementById("confidence"),
  keypoints: document.getElementById("keypoints"),
  latency: document.getElementById("latency"),
  camera: document.getElementById("camera"),
  reason: document.getElementById("reason"),
  device: document.getElementById("device"),
  refresh: document.getElementById("refresh"),
  connect: document.getElementById("connect"),
  disconnect: document.getElementById("disconnect"),
};

function setText(el, value) {
  el.textContent = value == null || value === "" ? "-" : String(value);
}

function update(payload) {
  const monitor = payload.monitor || {};
  const camera = payload.camera || {};
  const alerting = Boolean(monitor.alert);
  ids.alert.classList.toggle("active", alerting);
  ids.alert.textContent = alerting ? "ALERT: Patient sat up" : "No alert";
  setText(ids.state, monitor.state || "unknown");
  setText(ids.confidence, Number(monitor.confidence || 0).toFixed(2));
  setText(ids.keypoints, monitor.visible_keypoints || 0);
  setText(ids.latency, monitor.latency_ms == null ? "-" : `${monitor.latency_ms} ms`);
  setText(ids.camera, camera.connected ? `connected (${camera.source})` : "disconnected");
  setText(ids.reason, monitor.error || monitor.reason || "Waiting for monitor state.");
}

function connectSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${protocol}//${window.location.host}/ws/monitor`);
  socket.onmessage = (event) => update(JSON.parse(event.data));
  socket.onclose = () => window.setTimeout(connectSocket, 1000);
}

async function post(path) {
  await fetch(path, { method: "POST" });
}

async function postJSON(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return response.json();
}

async function loadDevices() {
  ids.device.innerHTML = '<option value="">Detecting cameras...</option>';
  try {
    const response = await fetch("/api/camera/devices");
    const data = await response.json();
    const devices = [];
    for (const provider of data.providers || []) {
      for (const device of provider.devices || []) devices.push(device);
    }
    ids.device.innerHTML = "";
    if (!devices.length) {
      ids.device.innerHTML = '<option value="">No camera found</option>';
      return;
    }
    for (const device of devices) {
      const option = document.createElement("option");
      option.value = device.id;
      option.textContent = device.display_name || device.id;
      ids.device.appendChild(option);
    }
    const real = devices.find((device) => device.provider !== "stub");
    ids.device.value = (real || devices[0]).id;
  } catch (error) {
    ids.device.innerHTML = '<option value="">Device scan failed</option>';
  }
}

ids.refresh.addEventListener("click", loadDevices);
ids.connect.addEventListener("click", async () => {
  const deviceId = ids.device.value || null;
  const data = await postJSON("/api/camera/connect", { device_id: deviceId });
  update({ camera: data.status || {}, monitor: {} });
});
ids.disconnect.addEventListener("click", async () => {
  await post("/api/camera/disconnect");
});

fetch("/api/status")
  .then((response) => response.json())
  .then(update)
  .catch(() => {});

loadDevices();
connectSocket();
