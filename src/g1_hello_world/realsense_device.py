from __future__ import annotations

from dataclasses import dataclass

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
        self._K = np.array(
            [
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @property
    def pipeline(self) -> rs.pipeline:
        return self._pipeline

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
