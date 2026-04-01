from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


DEFAULT_SERVER_ADDRESS = "36.103.198.235"
DEFAULT_PORT = 8000


@dataclass(frozen=True)
class BoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    @classmethod
    def from_xyxy(cls, data: list[int]) -> BoundingBox:
        if len(data) != 4:
            raise ValueError(f"bbox_xyxy must contain 4 values, got {data}")
        return cls(
            x_min=int(data[0]),
            y_min=int(data[1]),
            x_max=int(data[2]),
            y_max=int(data[3]),
        )


@dataclass(frozen=True)
class RegionResult:
    bbox: BoundingBox
    mask_path: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegionResult:
        return cls(
            bbox=BoundingBox.from_xyxy(data["bbox_xyxy"]),
            mask_path=data.get("mask_path"),
        )


@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageSize:
        return cls(width=int(data["width"]), height=int(data["height"]))


@dataclass(frozen=True)
class PredictResponse:
    success: bool
    handle_found: bool | None
    warnings: list[str]
    image_size: ImageSize | None
    handle: RegionResult | None
    local_white_region: RegionResult | None
    global_white_door_region: RegionResult | None
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PredictResponse:
        return cls(
            success=bool(data["success"]),
            handle_found=(
                bool(data["handle_found"]) if "handle_found" in data else None
            ),
            warnings=[str(x) for x in data.get("warnings", [])],
            image_size=(
                ImageSize.from_dict(data["image_size"])
                if "image_size" in data and isinstance(data["image_size"], dict)
                else None
            ),
            handle=(
                RegionResult.from_dict(data["handle"])
                if "handle" in data and isinstance(data["handle"], dict)
                else None
            ),
            local_white_region=(
                RegionResult.from_dict(data["local_white_region"])
                if "local_white_region" in data
                and isinstance(data["local_white_region"], dict)
                else None
            ),
            global_white_door_region=(
                RegionResult.from_dict(data["global_white_door_region"])
                if "global_white_door_region" in data
                and isinstance(data["global_white_door_region"], dict)
                else None
            ),
            raw=data,
        )


@dataclass(frozen=True)
class QueryResult:
    predict: PredictResponse
    mask_bytes: bytes | None
    mask_output_path: Path | None


def _post_image(
    *,
    server: str,
    port: int,
    endpoint: str,
    target_type: str,
    image_path: Path,
    timeout: float,
) -> requests.Response:
    url = f"http://{server}:{port}{endpoint}"
    with image_path.open("rb") as image_file:
        files = {"file": (image_path.name, image_file, "application/octet-stream")}
        response = requests.post(
            url,
            data={"target_type": target_type},
            files=files,
            timeout=timeout,
        )
    response.raise_for_status()
    return response


def _dump_json_if_possible(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def run_query(
    *,
    image_path: str | Path,
    server: str = DEFAULT_SERVER_ADDRESS,
    port: int = DEFAULT_PORT,
    timeout: float = 15.0,
    target_type: str = "handle",
    predict_endpoint: str = "/predict",
    fetch_mask: bool = True,
    mask_output: str | Path = "handle_mask.png",
) -> QueryResult:
    """Programmatic API: send query image and parse typed response."""
    image = Path(image_path).expanduser().resolve()
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")
    if not image.is_file():
        raise ValueError(f"Image path is not a file: {image}")

    predict_resp = _post_image(
        server=server,
        port=port,
        endpoint=predict_endpoint,
        target_type=target_type,
        image_path=image,
        timeout=timeout,
    )
    predict_json = _dump_json_if_possible(predict_resp)
    if not isinstance(predict_json, dict):
        raise ValueError(f"/predict did not return JSON object: {predict_resp.text}")
    parsed = PredictResponse.from_dict(predict_json)

    if not fetch_mask:
        return QueryResult(predict=parsed, mask_bytes=None, mask_output_path=None)

    mask_resp = _post_image(
        server=server,
        port=port,
        endpoint="/predict/mask",
        target_type=target_type,
        image_path=image,
        timeout=timeout,
    )
    mask_path = Path(mask_output).expanduser().resolve()
    mask_path.write_bytes(mask_resp.content)
    return QueryResult(
        predict=parsed,
        mask_bytes=mask_resp.content,
        mask_output_path=mask_path,
    )


def visualize_bboxes(
    *,
    image_path: str | Path,
    predict: PredictResponse,
    output_path: str | Path,
    show: bool = False,
) -> Path:
    """Draw returned bboxes on the image and save output."""
    import cv2

    image = cv2.imread(str(Path(image_path).expanduser().resolve()))
    if image is None:
        raise ValueError(f"Failed to read image for visualization: {image_path}")

    overlays: list[tuple[str, BoundingBox, tuple[int, int, int]]] = []
    if (
        "handle" in predict.raw
        and isinstance(predict.raw["handle"], dict)
        and predict.handle is not None
    ):
        overlays.append(("handle", predict.handle.bbox, (0, 255, 0)))
    if "local_white_region" in predict.raw and isinstance(
        predict.raw["local_white_region"], dict
    ) and predict.local_white_region is not None:
        overlays.append(("local_white", predict.local_white_region.bbox, (0, 255, 255)))
    if "global_white_door_region" in predict.raw and isinstance(
        predict.raw["global_white_door_region"], dict
    ) and predict.global_white_door_region is not None:
        overlays.append(
            ("global_white", predict.global_white_door_region.bbox, (255, 0, 0))
        )
    for label, bbox, color in overlays:
        cv2.rectangle(
            image,
            (bbox.x_min, bbox.y_min),
            (bbox.x_max, bbox.y_max),
            color,
            2,
        )
        cv2.putText(
            image,
            label,
            (bbox.x_min, max(0, bbox.y_min - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    out = Path(output_path).expanduser().resolve()
    ok = cv2.imwrite(str(out), image)
    if not ok:
        raise RuntimeError(f"Failed to save visualization image to {out}")

    if show:
        cv2.imshow("query visualization", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send an image to the remote prediction server."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the local image file to upload.",
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER_ADDRESS,
        help=f"Server IP or hostname (default: {DEFAULT_SERVER_ADDRESS}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Server port (default: {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Request timeout in seconds (default: 15).",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Only call /predict and skip /predict/mask.",
    )
    parser.add_argument(
        "--target-type",
        default="handle",
        choices=["handle", "frame"],
        help="target_type form field sent to the server.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use /predict/full instead of /predict.",
    )
    parser.add_argument(
        "--mask-output",
        default="handle_mask.png",
        help="Output path for /predict/mask response bytes.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Draw returned bounding boxes using OpenCV.",
    )
    parser.add_argument(
        "--viz-output",
        default="query_viz.png",
        help="Output path for bbox visualization image.",
    )
    parser.add_argument(
        "--show-viz",
        action="store_true",
        help="Show visualization window (requires GUI environment).",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    print(f"Sending image: {image_path}")
    print(f"Server: http://{args.server}:{args.port}")

    result = run_query(
        image_path=image_path,
        server=args.server,
        port=args.port,
        timeout=args.timeout,
        target_type=args.target_type,
        predict_endpoint="/predict/full" if args.full else "/predict",
        fetch_mask=not args.predict_only,
        mask_output=args.mask_output,
    )
    print("`/predict` response (JSON):")
    print(json.dumps(result.predict.raw, indent=2))
    if result.predict.handle is not None:
        print(
            "Handle bbox center/size:",
            f"{result.predict.handle.bbox.center},",
            f"{result.predict.handle.bbox.width}x{result.predict.handle.bbox.height}",
        )
    if result.mask_output_path is not None:
        print(f"Saved mask output to: {result.mask_output_path}")
    if args.visualize:
        viz_out = visualize_bboxes(
            image_path=image_path,
            predict=result.predict,
            output_path=args.viz_output,
            show=args.show_viz,
        )
        print(f"Saved bbox visualization to: {viz_out}")


if __name__ == "__main__":
    main()

