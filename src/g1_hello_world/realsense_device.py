from __future__ import annotations

from dataclasses import dataclass
import math
import threading

import numpy as np
import pyrealsense2 as rs


@dataclass(frozen=True, slots=True)
class RealSenseDeviceInfo:
    name: str
    serial: str


class RealSenseDeviceManager:
    """
    Owns a single RealSense pipeline (one device). Pass ``serial`` to pick a camera
    when several are connected; otherwise the SDK default device is used.

    Call :meth:`start` to run capture on a background thread; :meth:`read_aligned_rgb_depth`
    returns copies of the latest aligned RGB (``H×W×3``, ``uint8``) and depth (``H×W``, ``uint16``).
    :attr:`rgb` and :attr:`depth` are updated by that thread for live preview (e.g. Viser).
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        *,
        serial: str | None = None,
        enable_color: bool = True,
        enable_depth: bool = True,
    ) -> None:
        if not enable_color and not enable_depth:
            raise ValueError("At least one of enable_color, enable_depth must be True")

        self._width = width
        self._height = height
        self._fps = fps
        self._serial = serial
        self._enable_color = enable_color
        self._enable_depth = enable_depth

        self._pipeline = rs.pipeline()
        cfg = rs.config()
        if serial is not None:
            cfg.enable_device(serial)
        if enable_depth:
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        if enable_color:
            cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self._pipeline.start(cfg)

        if enable_color:
            profile = (
                self._pipeline.get_active_profile()
                .get_stream(rs.stream.color)
                .as_video_stream_profile()
            )
        else:
            profile = (
                self._pipeline.get_active_profile()
                .get_stream(rs.stream.depth)
                .as_video_stream_profile()
            )
        intr = profile.get_intrinsics()
        self._intrinsics = intr
        self._K = np.array(
            [
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        try:
            self._depth_scale = float(
                self._pipeline.get_active_profile()
                .get_device()
                .first_depth_sensor()
                .get_depth_scale()
            )
        except Exception:
            self._depth_scale = 0.001

        if enable_color and enable_depth:
            self._align_to_color = rs.align(rs.stream.color)
        else:
            self._align_to_color = None

        self.rgb = np.zeros((height, width, 3), dtype=np.uint8)
        self.depth = np.zeros((height, width), dtype=np.uint16)

        self._frame_lock = threading.Lock()
        self._stream_stop = threading.Event()
        self.frame_ready = threading.Event()
        self._stream_thread: threading.Thread | None = None

    @property
    def pipeline(self) -> rs.pipeline:
        """Low-level pipeline. Prefer :meth:`read_aligned_rgb_depth` for RGB-D."""
        return self._pipeline

    @property
    def depth_scale(self) -> float:
        """Meters per raw depth unit (multiply ``uint16`` depth by this for meters)."""
        return self._depth_scale

    def _read_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Wait for one frameset, align depth to color, return ``(rgb, depth)``.

        RGB is RGB order ``uint8``, shape ``(height, width, 3)``, C-contiguous.
        Depth is raw ``uint16`` aligned to the same pixels as ``rgb``.

        Raises:
            RuntimeError: If color or depth stream was not enabled at construction.
        """
        if self._align_to_color is None:
            raise RuntimeError(
                "read_aligned_rgb_depth requires both color and depth streams."
            )
        frameset = self._pipeline.wait_for_frames()
        aligned = self._align_to_color.process(frameset)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Missing color or depth frame from RealSense.")
        bgr = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        depth_u16 = np.ascontiguousarray(depth)
        with self._frame_lock:
            self.rgb = rgb
            self.depth = depth_u16
        return rgb, depth_u16

    def read_aligned_rgb_depth(self, *, timeout_s: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Latest aligned RGB and depth from the capture thread (copies of the same buffers
        as :attr:`rgb` / :attr:`depth`).

        Requires :meth:`start` so frames are arriving; blocks up to ``timeout_s`` for the
        first frame, then returns promptly on later calls.
        """
        if self._align_to_color is None:
            raise RuntimeError(
                "read_aligned_rgb_depth requires both color and depth streams."
            )
        if not self.frame_ready.wait(timeout_s):
            raise RuntimeError(
                "Timed out waiting for a RealSense frame — call start() on the device."
            )
        with self._frame_lock:
            return self.rgb.copy(), self.depth.copy()

    def _stream_loop(self) -> None:
        while not self._stream_stop.is_set():
            try:
                self._read_frames()
                self.frame_ready.set()
            except Exception:
                break

    def start(self) -> None:
        """Begin background capture (``wait_for_frames`` on a daemon thread)."""
        if self._stream_thread is not None and self._stream_thread.is_alive():
            return
        self._stream_stop.clear()
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            name="realsense-capture",
            daemon=True,
        )
        self._stream_thread.start()

    @property
    def K(self) -> np.ndarray:
        """3×3 intrinsics for the primary RGB stream, or depth if color is disabled."""
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
        """Vertical field of view in radians (pinhole model from primary intrinsics)."""
        fy = float(self._K[1, 1])
        if fy <= 0.0:
            raise ValueError("Invalid focal length fy for FOV computation.")
        return 2.0 * math.atan(self._height / (2.0 * fy))

    @property
    def aspect(self) -> float:
        """Width divided by height for the configured stream resolution."""
        return float(self._width) / float(self._height)

    def compute_camera_points(
        self,
        depth: np.ndarray,
        tracked_points: np.ndarray,  # (N, 2) in pixel coordinates
    ) -> np.ndarray:
        """
        Back-project 2D pixels to 3D in the camera frame (meters), same convention
        as Track-On ``client.py`` ``_compute_camera_points``: median depth in a 3×3
        patch, then ``rs2_deproject_pixel_to_point``. ``depth`` must be aligned to
        the same optical frame as these pixels (e.g. depth aligned to color).
        """
        pts = np.asarray(tracked_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("tracked_points must have shape (N, 2)")
        dmap = np.asarray(depth)
        if dmap.ndim != 2:
            raise ValueError("depth must be a 2D array")
        h, w = dmap.shape[:2]
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
            if depth_m <= 0:
                continue
            xyz[idx] = np.asarray(
                rs.rs2_deproject_pixel_to_point(
                    self._intrinsics, [float(xi), float(yi)], depth_m
                ),
                dtype=np.float32,
            )
        return xyz

    @staticmethod
    def list_devices() -> list[RealSenseDeviceInfo]:
        ctx = rs.context()
        out: list[RealSenseDeviceInfo] = []
        for dev in ctx.query_devices():
            out.append(
                RealSenseDeviceInfo(
                    name=str(dev.get_info(rs.camera_info.name)),
                    serial=str(dev.get_info(rs.camera_info.serial_number)),
                )
            )
        return out

    def stop(self) -> None:
        self._stream_stop.set()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=5.0)
            self._stream_thread = None
        self._pipeline.stop()
