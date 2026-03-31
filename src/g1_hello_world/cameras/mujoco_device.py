from __future__ import annotations

import math
import threading
import time

import mujoco
import numpy as np


class SimulatedCameraDevice:
    """MuJoCo-rendered RGB-D camera with the same surface API as RealSenseDeviceManager."""

    def __init__(
        self,
        sim: "Sim2Sim",
        *,
        camera_name: str,
        width: int,
        height: int,
        fps: int,
        fovy_deg: float = 58.0,
    ) -> None:
        self._sim = sim
        self._camera_name = camera_name
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

        self.rgb = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        self.depth = np.zeros((self._height, self._width), dtype=np.uint16)
        self.frame_ready = threading.Event()
        self._frame_lock = threading.Lock()
        self._stream_stop = threading.Event()
        self._stream_thread: threading.Thread | None = None
        self._renderer: mujoco.Renderer | None = mujoco.Renderer(
            sim.model, height=self._height, width=self._width
        )

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
        return self._fovy

    @property
    def aspect(self) -> float:
        return float(self._width) / float(self._height)

    def _read_frames(self) -> tuple[np.ndarray, np.ndarray]:
        with self._sim.lock:
            if self._renderer is None:
                raise RuntimeError(f"renderer for {self._camera_name!r} is closed")
            self._renderer.disable_depth_rendering()
            self._renderer.update_scene(self._sim.data, camera=self._camera_name)
            rgb = np.ascontiguousarray(self._renderer.render())
            self._renderer.enable_depth_rendering()
            self._renderer.update_scene(self._sim.data, camera=self._camera_name)
            depth_m = np.asarray(self._renderer.render(), dtype=np.float32)
            self._renderer.disable_depth_rendering()

        depth_raw = np.where(
            np.isfinite(depth_m) & (depth_m > 0.0),
            np.clip(np.rint(depth_m / self._depth_scale), 0.0, 65535.0),
            0.0,
        ).astype(np.uint16)
        with self._frame_lock:
            self.rgb = rgb
            self.depth = depth_raw
        return rgb, depth_raw

    def _stream_loop(self) -> None:
        period = 1.0 / float(max(1, self._fps))
        while not self._stream_stop.is_set():
            t0 = time.monotonic()
            try:
                self._read_frames()
                self.frame_ready.set()
            except Exception:
                break
            slack = period - (time.monotonic() - t0)
            if slack > 0.0:
                time.sleep(slack)

    def start(self) -> None:
        if self._stream_thread is not None and self._stream_thread.is_alive():
            return
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self._sim.model, height=self._height, width=self._width
            )
        self._stream_stop.clear()
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            name=f"sim-camera-{self._camera_name}",
            daemon=True,
        )
        self._stream_thread.start()

    def stop(self) -> None:
        self._stream_stop.set()
        if self._stream_thread is not None and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=2.0)
        self._stream_thread = None
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = None

    def read_aligned_rgb_depth(self, *, timeout_s: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
        if not self.frame_ready.wait(timeout_s):
            raise RuntimeError(
                f"Timed out waiting for simulated camera frame {self._camera_name!r}."
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
