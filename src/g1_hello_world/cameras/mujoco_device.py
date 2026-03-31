from __future__ import annotations

import math
import threading
import time
from typing import Any

import mujoco
import numpy as np
import zmq


class MujocoCameraStreamer:
    """Render RGB-D from a MuJoCo camera and publish frames over ZMQ."""

    def __init__(
        self,
        sim: Any,
        *,
        camera_name: str,
        endpoint: str,
        width: int,
        height: int,
        fps: int,
        fovy_deg: float = 58.0,
    ) -> None:
        self._sim = sim
        self._camera_name = camera_name
        self._endpoint = endpoint
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._depth_scale = 0.001
        self._fovy = math.radians(float(fovy_deg))
        fy = self._height / (2.0 * math.tan(self._fovy / 2.0))
        fx = fy
        cx = (self._width - 1.0) * 0.5
        cy = (self._height - 1.0) * 0.5
        self._K = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.bind(self._endpoint)

        # Created lazily on the rendering thread: EGL contexts are thread-affine; constructing
        # Renderer on the main thread and rendering from another thread triggers EGL_BAD_ACCESS
        # on many NVIDIA drivers. Multiple Renderers must also be driven from one thread with EGL.
        self._renderer: mujoco.Renderer | None = None

    def _ensure_renderer(self) -> mujoco.Renderer:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self._sim.model, height=self._height, width=self._width
            )
        return self._renderer

    def _render_frame(self) -> tuple[np.ndarray, np.ndarray]:
        renderer = self._ensure_renderer()
        with self._sim.lock:
            renderer.disable_depth_rendering()
            renderer.update_scene(self._sim.data, camera=self._camera_name)
            rgb = np.ascontiguousarray(renderer.render())
            renderer.enable_depth_rendering()
            renderer.update_scene(self._sim.data, camera=self._camera_name)
            depth_m = np.asarray(renderer.render(), dtype=np.float32)
            renderer.disable_depth_rendering()
        depth_raw = np.where(
            np.isfinite(depth_m) & (depth_m > 0.0),
            np.clip(np.rint(depth_m / self._depth_scale), 0.0, 65535.0),
            0.0,
        ).astype(np.uint16)
        return rgb, depth_raw

    def render_and_publish(self) -> None:
        """Render one RGB-D pair and publish on ZMQ. Must be called from a single dedicated thread."""
        rgb, depth = self._render_frame()
        self._socket.send_pyobj(
            {
                "rgb": rgb,
                "depth": depth,
                "K": self._K,
                "depth_scale": self._depth_scale,
                "fov_y": self._fovy,
                "aspect": float(self._width) / float(self._height),
                "width": self._width,
                "height": self._height,
                "fps": self._fps,
            }
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._socket.close(linger=0)


class ZmqSimCameraDevice:
    """Receive RGB-D frames from a remote MuJoCo renderer and expose the camera-device API."""

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(self._endpoint)

        self._K = np.eye(3, dtype=np.float64)
        self._depth_scale = 0.001
        self._fov_y = math.radians(58.0)
        self._aspect = 4.0 / 3.0
        self._width = 640
        self._height = 480
        self._fps = 30

        self.rgb = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        self.depth = np.zeros((self._height, self._width), dtype=np.uint16)
        self.frame_ready = threading.Event()
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def depth_scale(self) -> float:
        return self._depth_scale

    @property
    def K(self) -> np.ndarray:
        return self._K

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def fov_y(self) -> float:
        return self._fov_y

    @property
    def aspect(self) -> float:
        return self._aspect

    def _recv_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self._socket.recv_pyobj(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.002)
                continue
            with self._frame_lock:
                self.rgb = np.ascontiguousarray(payload["rgb"], dtype=np.uint8)
                self.depth = np.ascontiguousarray(payload["depth"], dtype=np.uint16)
                self._K = np.asarray(payload["K"], dtype=np.float64)
                self._depth_scale = float(payload["depth_scale"])
                self._fov_y = float(payload["fov_y"])
                self._aspect = float(payload["aspect"])
                self._width = int(payload["width"])
                self._height = int(payload["height"])
                self._fps = int(payload["fps"])
                self.frame_ready.set()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._recv_loop,
            name="sim-camera-recv",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._socket.close(linger=0)

    def read_aligned_rgb_depth(self, *, timeout_s: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
        if not self.frame_ready.wait(timeout_s):
            raise RuntimeError(
                f"Timed out waiting for simulated camera frame from {self._endpoint!r}."
            )
        with self._frame_lock:
            return self.rgb.copy(), self.depth.copy()

    def compute_camera_points(self, depth: np.ndarray, tracked_points: np.ndarray) -> np.ndarray:
        pts = np.asarray(tracked_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("tracked_points must have shape (N, 2)")
        dmap = np.asarray(depth)
        if dmap.ndim != 2:
            raise ValueError("depth must be a 2D array")
        h, w = dmap.shape[:2]
        fx, fy = float(self._K[0, 0]), float(self._K[1, 1])
        cx, cy = float(self._K[0, 2]), float(self._K[1, 2])
        xyz = np.full((pts.shape[0], 3), np.nan, dtype=np.float32)
        for idx, (x, y) in enumerate(pts):
            xi = int(round(float(x)))
            yi = int(round(float(y)))
            if not (0 <= xi < w and 0 <= yi < h):
                continue
            patch = dmap[max(0, yi - 1) : min(h, yi + 2), max(0, xi - 1) : min(w, xi + 2)]
            valid = patch[patch > 0]
            if valid.size == 0:
                continue
            depth_m = float(np.median(valid) * self._depth_scale)
            if depth_m <= 0.0:
                continue
            xyz[idx, 0] = (float(xi) - cx) * depth_m / fx
            xyz[idx, 1] = (float(yi) - cy) * depth_m / fy
            xyz[idx, 2] = depth_m
        return xyz
