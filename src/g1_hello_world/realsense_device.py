from __future__ import annotations

from dataclasses import dataclass
import math

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

    With both color and depth enabled, use :meth:`read_aligned_rgb_depth` for a
    single ``wait_for_frames`` call that returns **RGB** (``H×W×3``, ``uint8``) and
    **depth** (``H×W``, ``uint16`` raw units) **registered to the color image**.
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

    @property
    def pipeline(self) -> rs.pipeline:
        """Low-level pipeline. Prefer :meth:`read_aligned_rgb_depth` for RGB-D."""
        return self._pipeline

    @property
    def depth_scale(self) -> float:
        """Meters per raw depth unit (multiply ``uint16`` depth by this for meters)."""
        return self._depth_scale

    def read_aligned_rgb_depth(self) -> tuple[np.ndarray, np.ndarray]:
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
        self.rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        self.depth = np.ascontiguousarray(depth)
        return self.rgb, self.depth

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
        self._pipeline.stop()
