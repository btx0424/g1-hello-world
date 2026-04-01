from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import zmq

from g1_hello_world.estimators.point_track import TRACKED_POINTS_PUB_ENDPOINT


DEFAULT_PROXY_ENDPOINT = "tcp://127.0.0.1:5562"

_BBOX_COLORS: dict[str, tuple[int, int, int]] = {
    "handle": (0, 255, 0),
    "local_white_region": (0, 255, 255),
    "global_white_door_region": (255, 0, 0),
}


def _wait_for_reply(
    *,
    proxy_endpoint: str,
    target_type: str,
    predict_endpoint: str,
    request_timeout_s: float,
    segmentation_timeout_s: float,
) -> tuple[dict[str, Any], float]:
    context = zmq.Context.instance()
    sock = context.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, int(request_timeout_s * 1000.0))
    sock.setsockopt(zmq.SNDTIMEO, int(request_timeout_s * 1000.0))
    sock.connect(proxy_endpoint)
    try:
        t0 = time.time()
        sock.send_pyobj(
            {
                "op": "segment_head",
                "target_type": target_type,
                "predict_endpoint": predict_endpoint,
                "timeout_s": segmentation_timeout_s,
            }
        )
        reply = sock.recv_pyobj()
        if not isinstance(reply, dict):
            raise RuntimeError(f"unexpected reply type: {type(reply)!r}")
        return reply, t0
    finally:
        sock.close(linger=0)


def _wait_for_tracked_points(
    *,
    tracked_points_endpoint: str,
    timeout_s: float,
    min_timestamp: float,
) -> dict[str, Any]:
    context = zmq.Context.instance()
    sock = context.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000.0))
    sock.connect(tracked_points_endpoint)
    try:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                raise TimeoutError(
                    f"timed out waiting for tracked points from {tracked_points_endpoint}"
                )
            sock.setsockopt(zmq.RCVTIMEO, max(1, int(remaining_s * 1000.0)))
            payload = sock.recv_pyobj()
            if not isinstance(payload, dict):
                continue
            if payload.get("op") != "tracked_points_link":
                continue
            timestamp = float(payload.get("timestamp", 0.0))
            if timestamp < min_timestamp:
                continue
            return payload
    finally:
        sock.close(linger=0)


def _save_segmentation_visualization(
    *,
    reply: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Write a bbox visualization image for the segmentation reply."""
    width = int(reply.get("image_width", 1280))
    height = int(reply.get("image_height", 720))
    canvas = np.zeros((max(1, height), max(1, width), 3), dtype=np.uint8)

    bboxes = reply.get("bboxes", {})
    if isinstance(bboxes, dict):
        for label, bbox_xyxy in bboxes.items():
            if not isinstance(bbox_xyxy, list) or len(bbox_xyxy) < 4:
                continue
            x_min, y_min, x_max, y_max = [int(v) for v in bbox_xyxy[:4]]
            color = _BBOX_COLORS.get(str(label), (200, 200, 200))
            cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(
                canvas,
                str(label),
                (x_min, max(16, y_min - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
                cv2.LINE_AA,
            )

    out_path = Path(output_path).expanduser().resolve()
    if not cv2.imwrite(str(out_path), canvas):
        raise RuntimeError(f"failed to save segmentation visualization to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Send `segment_head` to app.py, wait for the segmentation reply, "
            "then wait for the next tracked-points publication."
        )
    )
    parser.add_argument(
        "--proxy-endpoint",
        default=DEFAULT_PROXY_ENDPOINT,
        help=f"ZMQ REP endpoint exposed by app.py (default: {DEFAULT_PROXY_ENDPOINT}).",
    )
    parser.add_argument(
        "--tracked-points-endpoint",
        default=TRACKED_POINTS_PUB_ENDPOINT,
        help=(
            "ZMQ PUB endpoint used by PointTrackerRemote "
            f"(default: {TRACKED_POINTS_PUB_ENDPOINT})."
        ),
    )
    parser.add_argument(
        "--target-type",
        default="handle",
        choices=["handle", "frame"],
        help="Segmentation target_type passed through to app.py.",
    )
    parser.add_argument(
        "--predict-endpoint",
        default="/predict",
        help="Segmentation HTTP endpoint path forwarded by app.py.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=20.0,
        help="Timeout in seconds for the ZMQ request/reply with app.py.",
    )
    parser.add_argument(
        "--segmentation-timeout",
        type=float,
        default=15.0,
        help="Timeout in seconds that app.py should use for the HTTP segmentation request.",
    )
    parser.add_argument(
        "--tracked-points-timeout",
        type=float,
        default=3.0,
        help="Timeout in seconds to wait for a fresh tracked-points publication.",
    )
    parser.add_argument(
        "--viz-output",
        default="head_segmentation_viz.png",
        help="Output path for segmentation bbox visualization image.",
    )
    args = parser.parse_args()

    reply, request_sent_at = _wait_for_reply(
        proxy_endpoint=args.proxy_endpoint,
        target_type=args.target_type,
        predict_endpoint=args.predict_endpoint,
        request_timeout_s=args.request_timeout,
        segmentation_timeout_s=args.segmentation_timeout,
    )
    print("Segmentation reply:")
    print(json.dumps(reply, indent=2))
    viz_path = _save_segmentation_visualization(
        reply=reply,
        output_path=args.viz_output,
    )
    print(f"Saved segmentation bbox visualization to: {viz_path}")

    if not bool(reply.get("ok", False)):
        raise SystemExit("segmentation request failed")
    if not bool(reply.get("tracker_submit_ok", False)):
        raise SystemExit(
            f"tracker was not seeded: {reply.get('tracker_submit_status', 'unknown error')}"
        )

    tracked = _wait_for_tracked_points(
        tracked_points_endpoint=args.tracked_points_endpoint,
        timeout_s=args.tracked_points_timeout,
        min_timestamp=request_sent_at,
    )
    print("Tracked points publication:")
    print(
        json.dumps(
            {
                "op": tracked.get("op"),
                "timestamp": tracked.get("timestamp"),
                "num_points": (
                    None
                    if tracked.get("points_link") is None
                    else int(len(tracked["points_link"]))
                ),
                "num_visible": (
                    None
                    if tracked.get("visibility") is None
                    else int(sum(bool(v) for v in tracked["visibility"]))
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
