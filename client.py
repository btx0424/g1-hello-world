"""
Off-board Track-On2 client.

The client keeps the RealSense + Gradio UI local and forwards RGB frames to
the remote inference server over ZMQ.
"""

from __future__ import annotations

import argparse
import cv2
from dataclasses import dataclass, field
import gradio as gr
import numpy as np
import pyrealsense2 as rs  # type: ignore
import threading
import time
from typing import Any, Optional, Tuple
import zmq


def encode_rgb_jpeg(rgb: np.ndarray, quality: int) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError("Failed to encode RGB frame as JPEG.")
    return encoded.tobytes()


@dataclass
class RealSenseConfig:
    width: int = 640
    height: int = 480
    fps: int = 30


class RealSenseCaptureBackend:
    def __init__(self, cfg: Optional[RealSenseConfig] = None):
        self.cfg = cfg or RealSenseConfig()
        self._pipeline = None
        self._align = None
        self._started = False
        self._color_intrinsics = None
        self._depth_scale = 0.001

    def start(self) -> None:
        if self._started:
            return
        self._pipeline = rs.pipeline()
        rs_cfg = rs.config()
        rs_cfg.enable_stream(rs.stream.color, self.cfg.width, self.cfg.height, rs.format.bgr8, self.cfg.fps)
        rs_cfg.enable_stream(rs.stream.depth, self.cfg.width, self.cfg.height, rs.format.z16, self.cfg.fps)
        profile = self._pipeline.start(rs_cfg)
        self._align = rs.align(rs.stream.color)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self._color_intrinsics = color_profile.get_intrinsics()
        try:
            self._depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        except Exception:
            self._depth_scale = 0.001
        self._started = True

    def read_rgb_depth(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self._started or self._pipeline is None or self._align is None:
            raise RuntimeError("RealSense capture is not started.")
        frameset = self._pipeline.wait_for_frames()
        aligned = self._align.process(frameset)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Missing color or depth frame from RealSense.")
        bgr = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth

    def stop(self) -> None:
        if self._pipeline is not None and self._started:
            self._pipeline.stop()
        self._pipeline = None
        self._align = None
        self._started = False
        self._color_intrinsics = None

    @property
    def color_intrinsics(self):
        return self._color_intrinsics

    @property
    def depth_scale(self) -> float:
        return self._depth_scale


@dataclass
class OffboardClientSession:
    server_endpoint: str
    jpeg_quality: int
    request_timeout_ms: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    request_lock: threading.Lock = field(default_factory=threading.Lock)
    rs_backend: Optional[RealSenseCaptureBackend] = None
    cam_width: int = 640
    cam_height: int = 480
    cam_fps: int = 30
    point_radius: int = 5
    query_radius: int = 6
    max_queries: int = 8

    frozen_frame: Optional[np.ndarray] = None
    query_points_xy: list[tuple[int, int]] = field(default_factory=list)
    query_overlay_rgb: Optional[np.ndarray] = None

    latest_rgb_frame: Optional[np.ndarray] = None
    latest_rgb_preview: Optional[np.ndarray] = None
    tracked_points: Optional[np.ndarray] = None
    tracked_visibility: Optional[np.ndarray] = None
    tracked_points_camera: Optional[np.ndarray] = None

    tracking_active: bool = False
    worker_thread: Optional[threading.Thread] = None
    worker_stop_event: threading.Event = field(default_factory=threading.Event)

    status_message: str = "Idle."
    stats_message: str = "No frames processed yet."
    context: Optional[zmq.Context] = None
    socket: Optional[zmq.Socket] = None

    def _set_status(self, message: str) -> None:
        self.status_message = message

    def _ensure_socket(self) -> zmq.Socket:
        if self.context is None:
            self.context = zmq.Context.instance()
        if self.socket is None:
            sock = self.context.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, self.request_timeout_ms)
            sock.setsockopt(zmq.SNDTIMEO, self.request_timeout_ms)
            sock.connect(self.server_endpoint)
            self.socket = sock
        return self.socket

    def _reset_socket(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = None

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.request_lock:
            sock = self._ensure_socket()
            try:
                sock.send_pyobj(payload)
                return sock.recv_pyobj()
            except Exception:
                self._reset_socket()
                raise

    def ping_server(self) -> str:
        try:
            response = self._request({"op": "ping"})
            if response.get("ok"):
                return response.get("status", "server-ready")
            return response.get("error", "server ping failed")
        except Exception as exc:
            return f"Server unreachable: {exc}"

    def _render_query_image(self, rgb: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
        out = rgb.copy()
        for x, y in points:
            cv2.circle(out, (int(x), int(y)), self.query_radius, (255, 255, 0), -1)
        return out

    def _draw_tracks_on_rgb(self, rgb: np.ndarray) -> np.ndarray:
        out = rgb.copy()
        if self.tracked_points is None or self.tracked_visibility is None:
            return out
        h, w = out.shape[:2]
        for idx, ((x, y), visible) in enumerate(zip(self.tracked_points, self.tracked_visibility)):
            xi, yi = int(round(float(x))), int(round(float(y)))
            if 0 <= xi < w and 0 <= yi < h:
                color = (0, 255, 0) if bool(visible) else (255, 0, 0)
                cv2.circle(out, (xi, yi), self.point_radius, color, -1)
                if self.tracked_points_camera is not None and idx < len(self.tracked_points_camera):
                    xyz = self.tracked_points_camera[idx]
                    if np.all(np.isfinite(xyz)):
                        label = f"{idx}: {xyz[0]:+.2f} {xyz[1]:+.2f} {xyz[2]:+.2f}m"
                        cv2.putText(
                            out,
                            label,
                            (min(xi + 8, w - 190), max(yi - 8, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            color,
                            1,
                            cv2.LINE_AA,
                        )
        return out

    def _compute_camera_points(self, depth: np.ndarray, tracked_points: np.ndarray) -> Optional[np.ndarray]:
        if self.rs_backend is None:
            return None
        intrinsics = self.rs_backend.color_intrinsics
        if intrinsics is None:
            return None
        h, w = depth.shape[:2]
        xyz = np.full((tracked_points.shape[0], 3), np.nan, dtype=np.float32)
        for idx, (x, y) in enumerate(tracked_points):
            xi = int(round(float(x)))
            yi = int(round(float(y)))
            if not (0 <= xi < w and 0 <= yi < h):
                continue
            patch = depth[max(0, yi - 1) : min(h, yi + 2), max(0, xi - 1) : min(w, xi + 2)]
            valid = patch[patch > 0]
            if valid.size == 0:
                continue
            depth_m = float(np.median(valid) * self.rs_backend.depth_scale)
            if depth_m <= 0:
                continue
            xyz[idx] = np.asarray(
                rs.rs2_deproject_pixel_to_point(intrinsics, [float(xi), float(yi)], depth_m),
                dtype=np.float32,
            )
        return xyz

    def _send_track_request(
        self,
        rgb: np.ndarray,
        queries_xy: Optional[list[tuple[int, int]]] = None,
        reset: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "op": "track",
            "frame_jpeg": encode_rgb_jpeg(rgb, self.jpeg_quality),
            "queries_xy": queries_xy,
            "reset": reset,
        }
        return self._request(payload)

    def start_camera(self) -> str:
        self.stop_camera()
        backend = RealSenseCaptureBackend(RealSenseConfig(self.cam_width, self.cam_height, self.cam_fps))
        try:
            backend.start()
        except Exception as exc:
            with self.lock:
                self._set_status(f"Failed to start RealSense: {exc}")
                return self.status_message
        server_status = self.ping_server()
        with self.lock:
            self.rs_backend = backend
            self.worker_stop_event = threading.Event()
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            self._set_status(
                f"RealSense started ({self.cam_width}x{self.cam_height} @ {self.cam_fps} FPS). {server_status}"
            )
            return self.status_message

    def stop_camera(self) -> str:
        thread = None
        backend = None
        with self.lock:
            thread = self.worker_thread
            backend = self.rs_backend
            self.worker_thread = None
            self.rs_backend = None
            self.worker_stop_event.set()
            self.tracking_active = False
            self.latest_rgb_frame = None
            self.latest_rgb_preview = None
            self.tracked_points_camera = None
            self._set_status("Camera stopped.")
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        if backend is not None:
            backend.stop()
        self._reset_socket()
        return self.status_message

    def capture_for_queries(self) -> Tuple[Optional[np.ndarray], str]:
        with self.lock:
            if self.latest_rgb_frame is None:
                self._set_status("Start the RealSense camera first.")
                return None, self.status_message
            self.frozen_frame = self.latest_rgb_frame.copy()
            self.query_points_xy = []
            self.query_overlay_rgb = self._render_query_image(self.frozen_frame, self.query_points_xy)
            self._set_status("Frozen frame captured. Click points, then submit queries.")
            return self.query_overlay_rgb, self.status_message

    def add_query_click(self, evt: gr.SelectData) -> Tuple[Optional[np.ndarray], str]:
        with self.lock:
            if self.frozen_frame is None:
                self._set_status("Capture a frame for queries first.")
                return None, self.status_message
            if len(self.query_points_xy) >= self.max_queries:
                self._set_status(f"Reached max query count ({self.max_queries}).")
                return self.query_overlay_rgb, self.status_message
            x, y = int(evt.index[0]), int(evt.index[1])
            self.query_points_xy.append((x, y))
            self.query_overlay_rgb = self._render_query_image(self.frozen_frame, self.query_points_xy)
            self._set_status(f"Queries: {len(self.query_points_xy)}/{self.max_queries}. Submit to start tracking.")
            return self.query_overlay_rgb, self.status_message

    def clear_queries(self) -> Tuple[Optional[np.ndarray], str]:
        with self.lock:
            self.query_points_xy = []
            if self.frozen_frame is not None:
                self.query_overlay_rgb = self._render_query_image(self.frozen_frame, [])
            self._set_status("Cleared query points.")
            return self.query_overlay_rgb, self.status_message

    def submit_queries(self) -> str:
        with self.lock:
            if self.frozen_frame is None or not self.query_points_xy:
                self._set_status("Capture a frame and add query points first.")
                return self.status_message
            frozen_frame = self.frozen_frame.copy()
            queries_xy = list(self.query_points_xy)

        roundtrip_t0 = time.perf_counter()
        try:
            response = self._send_track_request(frozen_frame, queries_xy=queries_xy, reset=True)
        except Exception as exc:
            with self.lock:
                self._set_status(f"Server request failed: {exc}")
                return self.status_message
        roundtrip_ms = (time.perf_counter() - roundtrip_t0) * 1000.0

        with self.lock:
            if not response.get("ok"):
                self._set_status(f"Server error: {response.get('error', 'unknown error')}")
                return self.status_message
            self.tracked_points = np.asarray(response["points"], dtype=np.float32)
            self.tracked_visibility = np.asarray(response["visibility"], dtype=bool)
            self.tracked_points_camera = None
            self.tracking_active = True
            stats = response.get("stats", {})
            self.stats_message = (
                f"rt={roundtrip_ms:.1f} ms | server_infer={stats.get('infer_ms', 0.0):.1f} ms | "
                f"server_total={stats.get('total_ms', 0.0):.1f} ms | queries={len(queries_xy)}/{self.max_queries}"
            )
            self._set_status("Tracking started via remote server.")
            return self.status_message

    def _worker_loop(self) -> None:
        frame_count = 0
        fps_t0 = time.perf_counter()
        while not self.worker_stop_event.is_set():
            with self.lock:
                backend = self.rs_backend
                tracking_active = self.tracking_active
            if backend is None:
                break

            t0 = time.perf_counter()
            try:
                rgb, depth = backend.read_rgb_depth()
            except Exception as exc:
                with self.lock:
                    self._set_status(f"Failed to read RealSense frames: {exc}")
                time.sleep(0.01)
                continue

            capture_ms = (time.perf_counter() - t0) * 1000.0
            roundtrip_ms = 0.0
            server_infer_ms = 0.0
            server_total_ms = 0.0
            valid_xyz_count = 0

            if tracking_active:
                t_req = time.perf_counter()
                try:
                    response = self._send_track_request(rgb, queries_xy=None, reset=False)
                    roundtrip_ms = (time.perf_counter() - t_req) * 1000.0
                    if response.get("ok"):
                        stats = response.get("stats", {})
                        server_infer_ms = float(stats.get("infer_ms", 0.0))
                        server_total_ms = float(stats.get("total_ms", 0.0))
                        tracked_points = np.asarray(response["points"], dtype=np.float32)
                        tracked_visibility = np.asarray(response["visibility"], dtype=bool)
                        tracked_points_camera = self._compute_camera_points(depth, tracked_points)
                        if tracked_points_camera is not None:
                            valid_xyz_count = int(np.isfinite(tracked_points_camera[:, 2]).sum())
                        with self.lock:
                            self.tracked_points = tracked_points
                            self.tracked_visibility = tracked_visibility
                            self.tracked_points_camera = tracked_points_camera
                    else:
                        with self.lock:
                            self._set_status(f"Server error: {response.get('error', 'unknown error')}")
                            self.tracking_active = False
                except Exception as exc:
                    with self.lock:
                        self._set_status(f"Server request failed: {exc}")
                        self.tracking_active = False

            rgb_vis = self._draw_tracks_on_rgb(rgb)
            total_ms = (time.perf_counter() - t0) * 1000.0
            frame_count += 1
            fps_elapsed = time.perf_counter() - fps_t0
            fps = frame_count / fps_elapsed if fps_elapsed > 0 else 0.0

            with self.lock:
                self.latest_rgb_frame = rgb
                self.latest_rgb_preview = rgb_vis
                self.stats_message = (
                    f"capture={capture_ms:.1f} ms | rt={roundtrip_ms:.1f} ms | "
                    f"server_infer={server_infer_ms:.1f} ms | server_total={server_total_ms:.1f} ms | "
                    f"loop={total_ms:.1f} ms | fps={fps:.1f} | xyz={valid_xyz_count}"
                )

    def tick(self) -> Tuple[Optional[np.ndarray], str, str]:
        with self.lock:
            return (
                self.latest_rgb_preview,
                self.status_message,
                self.stats_message,
            )


def build_ui(session: OffboardClientSession) -> gr.Blocks:
    with gr.Blocks(title="Track-On2 Off-Board Client") as demo:
        gr.Markdown(
            "## Track-On2 Off-Board Client\n"
            "RealSense RGB on this machine, remote inference over ZMQ, with tracked points projected to camera-frame XYZ."
        )
        status = gr.Textbox(label="Status", interactive=False)
        stats = gr.Textbox(label="Timing", interactive=False)

        with gr.Row():
            start = gr.Button("Start RealSense")
            stop = gr.Button("Stop RealSense")
            capture_btn = gr.Button("Capture RGB for queries")

        with gr.Row():
            live_preview = gr.Image(label="RGB preview", type="numpy")
            query_img = gr.Image(
                label="Frozen RGB frame for query selection",
                type="numpy",
                interactive=True,
            )

        with gr.Row():
            submit_btn = gr.Button("Submit queries")
            clear_btn = gr.Button("Clear query points")

        start.click(fn=session.start_camera, outputs=[status])
        stop.click(fn=session.stop_camera, outputs=[status])
        capture_btn.click(fn=session.capture_for_queries, outputs=[query_img, status])
        query_img.select(fn=session.add_query_click, outputs=[query_img, status])
        clear_btn.click(fn=session.clear_queries, outputs=[query_img, status])
        submit_btn.click(fn=session.submit_queries, outputs=[status])

        timer = gr.Timer(0.05)
        timer.tick(fn=session.tick, outputs=[live_preview, status, stats])

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track-On2 off-board RealSense client")
    parser.add_argument("--server-endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--realsense-width", type=int, default=640)
    parser.add_argument("--realsense-height", type=int, default=480)
    parser.add_argument("--realsense-fps", type=int, default=30)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--request-timeout-ms", type=int, default=3000)
    parser.add_argument("--max-queries", type=int, default=8)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = OffboardClientSession(
        server_endpoint=args.server_endpoint,
        jpeg_quality=args.jpeg_quality,
        request_timeout_ms=args.request_timeout_ms,
    )
    session.cam_width = args.realsense_width
    session.cam_height = args.realsense_height
    session.cam_fps = args.realsense_fps
    session.max_queries = args.max_queries
    app = build_ui(session)
    app.queue()
    app.launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
