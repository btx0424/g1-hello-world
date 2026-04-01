"""
Remote Track-On2-style point tracking: Gradio UI + ZMQ, using ``RealSenseDeviceManager``.

The camera runs async capture (:meth:`RealSenseDeviceManager.start`). This tracker is
self-contained: after :meth:`start`, it polls aligned RGB-D snapshots on its own worker
thread and maintains the latest tracked points for the host to consume.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import gradio as gr
import numpy as np
import zmq

from g1_hello_world.constants import R_SITE_FROM_OPENCV
from g1_hello_world.cameras import RealSenseDeviceManager

def encode_rgb_jpeg(rgb: np.ndarray, quality: int) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError("Failed to encode RGB frame as JPEG.")
    return encoded.tobytes()


class PointTrackerRemote:
    """
    ZMQ client for the same ``track`` / ``ping`` API as ``track_on/scripts/client.py``.

    After :meth:`start`, a daemon worker thread repeatedly calls
    ``read_aligned_rgb_depth`` (cheap lock + copy) while the RealSense capture
    thread owns ``wait_for_frames``. The device must already be streaming:
    ``realsense_device.start()`` before :meth:`PointTrackerRemote.start`.
    """

    def __init__(
        self,
        server_endpoint: str,
        realsense_device: RealSenseDeviceManager,
        *,
        aligned_read_timeout_s: float = 2.0,
        poll_period_s: float | None = None,
        jpeg_quality: int = 90,
        request_timeout_ms: int = 3000,
        point_radius: int = 5,
        query_radius: int = 6,
        max_queries: int = 8,
    ) -> None:
        self.server_endpoint = server_endpoint
        self._rs = realsense_device
        self._aligned_read_timeout_s = float(aligned_read_timeout_s)
        self._poll_period_s = poll_period_s
        self.jpeg_quality = jpeg_quality
        self.request_timeout_ms = request_timeout_ms
        self.point_radius = point_radius
        self.query_radius = query_radius
        self.max_queries = max_queries

        self._lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None
        self._worker_stop = threading.Event()
        self._session_live: bool = False
        self._fps_frame_count: int = 0
        self._fps_t0: float = 0.0

        self.frozen_frame: np.ndarray | None = None
        self.query_points_xy: list[tuple[int, int]] = []
        self.query_overlay_rgb: np.ndarray | None = None

        self.latest_rgb_frame: np.ndarray | None = None
        self.latest_rgb_preview: np.ndarray | None = None
        self.tracked_points: np.ndarray | None = None
        self.tracked_visibility: np.ndarray | None = None
        self.tracked_points_link: np.ndarray | None = None

        self.tracking_active: bool = False
        self.status_message: str = "Idle."
        self.stats_message: str = "No frames processed yet."

        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None

    # --- ZMQ (same shape as OffboardClientSession in client.py) ---

    def _set_status(self, message: str) -> None:
        self.status_message = message

    def _ensure_socket(self) -> zmq.Socket:
        if self._context is None:
            self._context = zmq.Context.instance()
        if self._socket is None:
            sock = self._context.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, self.request_timeout_ms)
            sock.setsockopt(zmq.SNDTIMEO, self.request_timeout_ms)
            sock.connect(self.server_endpoint)
            self._socket = sock
        return self._socket

    def _reset_socket(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = None

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._request_lock:
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
                return str(response.get("status", "server-ready"))
            return str(response.get("error", "server ping failed"))
        except Exception as exc:
            return f"Server unreachable: {exc}"

    def _send_track_request(
        self,
        rgb: np.ndarray,
        queries_xy: list[tuple[int, int]] | None = None,
        *,
        reset: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "op": "track",
            "frame_jpeg": encode_rgb_jpeg(rgb, self.jpeg_quality),
            "queries_xy": queries_xy,
            "reset": reset,
        }
        return self._request(payload)

    # --- Geometry / overlays (mirror client.py) ---

    def _render_query_image(
        self, rgb: np.ndarray, points: list[tuple[int, int]]
    ) -> np.ndarray:
        out = rgb.copy()
        for x, y in points:
            cv2.circle(out, (int(x), int(y)), self.query_radius, (255, 255, 0), -1)
        return out

    def _draw_tracks_on_rgb(self, rgb: np.ndarray) -> np.ndarray:
        out = rgb.copy()
        if self.tracked_points is None or self.tracked_visibility is None:
            return out
        h, w = out.shape[:2]
        for idx, ((x, y), visible) in enumerate(
            zip(self.tracked_points, self.tracked_visibility)
        ):
            xi, yi = int(round(float(x))), int(round(float(y)))
            if 0 <= xi < w and 0 <= yi < h:
                color = (0, 255, 0) if bool(visible) else (255, 0, 0)
                cv2.circle(out, (xi, yi), self.point_radius, color, -1)
                if self.tracked_points_link is not None and idx < len(
                    self.tracked_points_link
                ):
                    xyz = self.tracked_points_link[idx]
                    if np.all(np.isfinite(xyz)):
                        label = (
                            f"{idx}: {xyz[0]:+.2f} {xyz[1]:+.2f} {xyz[2]:+.2f}m"
                        )
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

    def _process_frame_rgb_depth(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        *,
        capture_ms: float,
        t0_loop: float,
    ) -> None:
        roundtrip_ms = 0.0
        server_infer_ms = 0.0
        server_total_ms = 0.0
        valid_xyz_count = 0

        with self._lock:
            tracking_active = self.tracking_active

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
                    tracked_points_camera = self._rs.compute_camera_points(
                        depth, tracked_points
                    )
                    tracked_points_link = tracked_points_camera @ R_SITE_FROM_OPENCV.T
                    valid_xyz_count = int(
                        np.isfinite(tracked_points_camera[:, 2]).sum()
                    )
                    with self._lock:
                        self.tracked_points = tracked_points
                        self.tracked_visibility = tracked_visibility
                        self.tracked_points_link = tracked_points_link
                else:
                    with self._lock:
                        self._set_status(
                            f"Server error: {response.get('error', 'unknown error')}"
                        )
                        self.tracking_active = False
            except Exception as exc:
                with self._lock:
                    self._set_status(f"Server request failed: {exc}")
                    self.tracking_active = False

        rgb_vis = self._draw_tracks_on_rgb(rgb)
        total_ms = (time.perf_counter() - t0_loop) * 1000.0
        self._fps_frame_count += 1
        fps_elapsed = time.perf_counter() - self._fps_t0
        fps = self._fps_frame_count / fps_elapsed if fps_elapsed > 0 else 0.0

        with self._lock:
            self.latest_rgb_frame = rgb
            self.latest_rgb_preview = rgb_vis
            self.stats_message = (
                f"capture={capture_ms:.1f} ms | rt={roundtrip_ms:.1f} ms | "
                f"server_infer={server_infer_ms:.1f} ms | server_total={server_total_ms:.1f} ms | "
                f"loop={total_ms:.1f} ms | fps={fps:.1f} | xyz={valid_xyz_count}"
            )
    # --- Worker ---

    def _worker_loop(self) -> None:
        self._fps_frame_count = 0
        self._fps_t0 = time.perf_counter()
        default_period = max(1e-3, 1.0 / float(max(1, self._rs.fps)))
        period = (
            float(self._poll_period_s)
            if self._poll_period_s is not None
            else default_period
        )
        while not self._worker_stop.is_set():
            t0 = time.perf_counter()
            try:
                rgb, depth = self._rs.read_aligned_rgb_depth(timeout_s=0.25)
            except Exception as exc:
                with self._lock:
                    self._set_status(f"Failed to read RealSense frames: {exc}")
                time.sleep(0.05)
                continue

            capture_ms = (time.perf_counter() - t0) * 1000.0
            self._process_frame_rgb_depth(
                rgb, depth, capture_ms=capture_ms, t0_loop=t0
            )
            elapsed = time.perf_counter() - t0
            slack = period - elapsed
            if slack > 0:
                time.sleep(slack)

    def start(self) -> str:
        """Ping the server and begin the asynchronous tracking session."""
        self.stop()
        server_status = self.ping_server()
        if not self._rs.frame_ready.wait(timeout=5.0):
            self._session_live = False
            self._set_status(
                "No camera frames yet — call RealSenseDeviceManager.start() before starting the tracker."
            )
            return self.status_message
        self._session_live = True
        self._fps_frame_count = 0
        self._fps_t0 = time.perf_counter()
        self._worker_stop = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="point-tracker-remote",
        )
        self._worker_thread.start()
        self._set_status(
            f"Remote tracker ({self._rs.width}x{self._rs.height} @ {self._rs.fps} FPS). "
            f"{server_status}"
        )
        return self.status_message

    def stop(self) -> str:
        """Stop worker and ZMQ; leaves ``RealSenseDeviceManager`` running."""
        self._session_live = False
        with self._lock:
            thread = self._worker_thread
            self._worker_thread = None
            self._worker_stop.set()
            self.tracking_active = False
            self.latest_rgb_frame = None
            self.latest_rgb_preview = None
            self.tracked_points_link = None
            self._set_status("Remote tracker stopped.")
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._reset_socket()
        return self.status_message

    # --- Gradio handlers (mirror client.py) ---

    def capture_for_queries(
        self,
    ) -> tuple[np.ndarray | None, str]:
        with self._lock:
            if self.latest_rgb_frame is None:
                self._set_status("Start the remote tracker first.")
                return None, self.status_message
            self.frozen_frame = self.latest_rgb_frame.copy()
            self.query_points_xy = []
            self.query_overlay_rgb = self._render_query_image(
                self.frozen_frame, self.query_points_xy
            )
            self._set_status(
                "Frozen frame captured. Click points, then submit queries."
            )
            return self.query_overlay_rgb, self.status_message

    def add_query_click(
        self, evt: gr.SelectData
    ) -> tuple[np.ndarray | None, str]:
        with self._lock:
            if self.frozen_frame is None:
                self._set_status("Capture a frame for queries first.")
                return None, self.status_message
            if len(self.query_points_xy) >= self.max_queries:
                self._set_status(f"Reached max query count ({self.max_queries}).")
                return self.query_overlay_rgb, self.status_message
            x, y = int(evt.index[0]), int(evt.index[1])
            self.query_points_xy.append((x, y))
            self.query_overlay_rgb = self._render_query_image(
                self.frozen_frame, self.query_points_xy
            )
            self._set_status(
                f"Queries: {len(self.query_points_xy)}/{self.max_queries}. "
                "Submit to start tracking."
            )
            return self.query_overlay_rgb, self.status_message

    def clear_queries(self) -> tuple[np.ndarray | None, str]:
        with self._lock:
            self.query_points_xy = []
            if self.frozen_frame is not None:
                self.query_overlay_rgb = self._render_query_image(
                    self.frozen_frame, []
                )
            self._set_status("Cleared query points.")
            return self.query_overlay_rgb, self.status_message

    def set_query_points(
        self,
        queries_xy: list[tuple[int, int]] | np.ndarray,
        *,
        capture_latest: bool = True,
    ) -> tuple[np.ndarray | None, str]:
        """Programmatically set pixel queries on a frozen RGB frame."""
        points = np.asarray(queries_xy, dtype=np.int32)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("queries_xy must have shape (N, 2)")
        if len(points) == 0:
            return self.clear_queries()
        if len(points) > self.max_queries:
            raise ValueError(
                f"queries_xy has {len(points)} points, exceeding max_queries={self.max_queries}"
            )

        with self._lock:
            if capture_latest or self.frozen_frame is None:
                if self.latest_rgb_frame is None:
                    self._set_status("Start the remote tracker first.")
                    return None, self.status_message
                self.frozen_frame = self.latest_rgb_frame.copy()

            h, w = self.frozen_frame.shape[:2]
            query_points_xy: list[tuple[int, int]] = []
            for x_raw, y_raw in points.tolist():
                x, y = int(x_raw), int(y_raw)
                if not (0 <= x < w and 0 <= y < h):
                    raise ValueError(
                        f"query point {(x, y)} is outside image bounds {(w, h)}"
                    )
                query_points_xy.append((x, y))

            self.query_points_xy = query_points_xy
            self.query_overlay_rgb = self._render_query_image(
                self.frozen_frame, self.query_points_xy
            )
            self._set_status(
                f"Queries: {len(self.query_points_xy)}/{self.max_queries}. "
                "Submit to start tracking."
            )
            return self.query_overlay_rgb, self.status_message

    def submit_query_points(
        self,
        queries_xy: list[tuple[int, int]] | np.ndarray,
        *,
        capture_latest: bool = True,
    ) -> str:
        """Programmatically freeze a frame, set queries, and start tracking."""
        self.set_query_points(queries_xy, capture_latest=capture_latest)
        return self.submit_queries()

    def submit_queries(self) -> str:
        with self._lock:
            if self.frozen_frame is None or not self.query_points_xy:
                self._set_status("Capture a frame and add query points first.")
                return self.status_message
            frozen_frame = self.frozen_frame.copy()
            queries_xy = list(self.query_points_xy)

        roundtrip_t0 = time.perf_counter()
        try:
            response = self._send_track_request(
                frozen_frame, queries_xy=queries_xy, reset=True
            )
        except Exception as exc:
            with self._lock:
                self._set_status(f"Server request failed: {exc}")
                return self.status_message
        roundtrip_ms = (time.perf_counter() - roundtrip_t0) * 1000.0

        with self._lock:
            if not response.get("ok"):
                self._set_status(
                    f"Server error: {response.get('error', 'unknown error')}"
                )
                return self.status_message
            self.tracked_points = np.asarray(response["points"], dtype=np.float32)
            self.tracked_visibility = np.asarray(response["visibility"], dtype=bool)
            self.tracked_points_link = None
            self.tracking_active = True
            stats = response.get("stats", {})
            self.stats_message = (
                f"rt={roundtrip_ms:.1f} ms | server_infer={stats.get('infer_ms', 0.0):.1f} ms | "
                f"server_total={stats.get('total_ms', 0.0):.1f} ms | queries={len(queries_xy)}/{self.max_queries}"
            )
            self._set_status("Tracking started via remote server.")
            return self.status_message

    def tick(self) -> tuple[np.ndarray | None, str, str]:
        with self._lock:
            return (
                self.latest_rgb_preview,
                self.status_message,
                self.stats_message,
            )

    def get_tracked_points_snapshot(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Thread-safe copies of the latest tracked MuJoCo camera-link-frame XYZ and visibility."""
        with self._lock:
            points_link = (
                None
                if self.tracked_points_link is None
                else self.tracked_points_link.copy()
            )
            visibility = (
                None
                if self.tracked_visibility is None
                else self.tracked_visibility.copy()
            )
        return points_link, visibility

    def build_ui(self) -> gr.Blocks:
        with gr.Blocks(title="Point tracker (remote)") as demo:
            gr.Markdown(
                "## Remote point tracker\n"
                "Shared RealSense pipeline, aligned depth, inference over ZMQ — same protocol as "
                "`track_on/scripts/client.py`."
            )
            status = gr.Textbox(label="Status", interactive=False)
            stats_tb = gr.Textbox(label="Timing", interactive=False)

            with gr.Row():
                start = gr.Button("Start remote tracker")
                stop = gr.Button("Stop remote tracker")
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

            start.click(fn=self.start, outputs=[status])
            stop.click(fn=self.stop, outputs=[status])
            capture_btn.click(
                fn=self.capture_for_queries, outputs=[query_img, status]
            )
            query_img.select(fn=self.add_query_click, outputs=[query_img, status])
            clear_btn.click(fn=self.clear_queries, outputs=[query_img, status])
            submit_btn.click(fn=self.submit_queries, outputs=[status])

            timer = gr.Timer(0.05)
            timer.tick(fn=self.tick, outputs=[live_preview, status, stats_tb])

        return demo

    def run_ui(self, **launch_kwargs: Any) -> None:
        """
        ``start()`` then launch Gradio (blocking).
        """
        try:
            self.start()
            demo = self.build_ui()
            demo.queue()
            demo.launch(**launch_kwargs)
        finally:
            self.stop()
