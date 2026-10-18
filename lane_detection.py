"""Classical lane-region segmentation for a single road image.

This deliberately uses explainable OpenCV operations (colour thresholding,
Canny edges, ROI masking and probabilistic Hough lines). It does not use a
trained model. When one side is absent, a clearly reported width-based fallback
may be used; it must not be mistaken for a measured road boundary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config


@dataclass
class LaneResult:
    left: Optional[tuple[tuple[int, int], tuple[int, int]]]
    right: Optional[tuple[tuple[int, int], tuple[int, int]]]
    fallback: Optional[str]
    lane_region: np.ndarray
    final_mask: np.ndarray
    diagnostics: dict[str, np.ndarray]
    curved_boundaries: Optional[tuple[np.ndarray, np.ndarray]] = None


class LaneDetectionError(RuntimeError):
    """A detection failure that carries completed stages for diagnosis."""

    def __init__(self, message: str, diagnostics: dict[str, np.ndarray]):
        super().__init__(message)
        self.diagnostics = diagnostics


def load_image(path: str | Path) -> np.ndarray:
    """Load a colour image or raise an actionable error."""
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a three-channel colour image.")
    return image


def preprocess_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, config.BLUR_KERNEL, 0)
    return gray, blurred


def create_colour_mask(image: np.ndarray) -> np.ndarray:
    """Highlight bright, low-saturation white paint and yellow paint."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 165]), np.array([180, 90, 255]))
    yellow = cv2.inRange(hsv, np.array([12, 65, 95]), np.array([42, 255, 255]))
    return cv2.bitwise_or(white, yellow)


def create_roi_mask(shape: tuple[int, ...]) -> np.ndarray:
    height, width = shape[:2]
    top = int(height * config.ROI_TOP_FRACTION)
    vertices = np.array([[(0, height - 1), (int(width * .42), top),
                          (int(width * .62), top), (width - 1, height - 1)]], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, vertices, 255)
    return mask


def detect_line_segments(edges: np.ndarray, roi: np.ndarray) -> list[tuple[int, int, int, int]]:
    masked_edges = cv2.bitwise_and(edges, roi)
    segments = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, config.HOUGH_THRESHOLD,
                               minLineLength=config.HOUGH_MIN_LINE_LENGTH,
                               maxLineGap=config.HOUGH_MAX_LINE_GAP)
    if segments is None:
        return []
    return [tuple(int(v) for v in line[0]) for line in segments]


def _fit_side(segments: list[tuple[int, int, int, int]], side: str,
              width: int, height: int) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
    points: list[tuple[int, int, float]] = []
    for x1, y1, x2, y2 in segments:
        dy = y2 - y1
        if dy == 0:
            continue
        slope = (x2 - x1) / dy  # x change per y change, naturally handles either image direction
        if abs(slope) < config.MIN_LINE_SLOPE or abs(slope) > config.MAX_LINE_SLOPE:
            continue
        midpoint = (x1 + x2) / 2
        if side == "left" and midpoint >= width * .57:
            continue
        if side == "right" and midpoint <= width * .43:
            continue
        length = float(np.hypot(x2 - x1, dy))
        points.extend([(x1, y1, length), (x2, y2, length)])
    if not points:
        return None
    pts = np.array([(x, y) for x, y, _ in points], dtype=np.float64)
    weights = np.array([w for _, _, w in points], dtype=np.float64)
    # x = a*y + b is a convenient image-space model for near-vertical lines.
    a, b = np.polyfit(pts[:, 1], pts[:, 0], 1, w=weights)
    y_bottom, y_top = height - 1, int(height * config.ROI_TOP_FRACTION)
    return ((int(np.clip(a * y_bottom + b, 0, width - 1)), y_bottom),
            (int(np.clip(a * y_top + b, 0, width - 1)), y_top))


def estimate_lane_boundaries(segments: list[tuple[int, int, int, int]],
                             width: int, height: int) -> tuple[
                                 Optional[tuple[tuple[int, int], tuple[int, int]]],
                                 Optional[tuple[tuple[int, int], tuple[int, int]]]]:
    return _fit_side(segments, "left", width, height), _fit_side(segments, "right", width, height)


def _bright_line_support(image: np.ndarray,
                         line: tuple[tuple[int, int], tuple[int, int]]) -> int:
    """Count scanlines where a candidate overlaps very bright white paint."""
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 235]), np.array([180, 65, 255]))
    (xb, yb), (xt, yt) = line
    ys = np.linspace(max(yt, int(height * .50)), yb, 80).astype(int)
    hits = 0
    for y in ys:
        x = int(xb + (xt - xb) * ((y - yb) / (yt - yb)))
        if np.any(white[y, max(0, x - 3):min(width, x + 4)]):
            hits += 1
    return hits


def _white_marking_points(image: np.ndarray) -> np.ndarray:
    """Find row-wise centers of the bright white marking in the central-right ROI."""
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 215]), np.array([180, 80, 255]))
    x0, x1 = int(width * .50), int(width * .82)
    points: list[tuple[int, int]] = []
    for y in range(int(height * config.ROI_TOP_FRACTION), height):
        xs = np.flatnonzero(white[y, x0:x1]) + x0
        if not len(xs):
            continue
        groups = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)
        groups = [group for group in groups if len(group) >= max(3, int(width * .006))]
        if groups:
            group = max(groups, key=len)
            points.append((int(np.median(group)), y))
    return np.asarray(points, dtype=np.int32)


def _curved_region_from_marking(shape: tuple[int, ...], marking_points: np.ndarray
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                           tuple[tuple[int, int], tuple[int, int]],
                                           tuple[tuple[int, int], tuple[int, int]]]:
    """Fit the visible stripe and infer the neighboring boundary with widening perspective."""
    height, width = shape[:2]
    x = marking_points[:, 0].astype(np.float64)
    y = marking_points[:, 1].astype(np.float64)
    coefficients = np.polyfit(y, x, 2)
    y_top = int(marking_points[:, 1].min())
    ys = np.arange(y_top, height, dtype=np.int32)
    center_x = np.clip(np.polyval(coefficients, ys), 0, width - 1).astype(np.int32)
    offsets = np.linspace(width * .14, width * .31, len(ys))
    outer_x = np.clip(center_x + offsets, 0, width - 1).astype(np.int32)
    left_curve = np.column_stack((center_x, ys)).astype(np.int32)
    right_curve = np.column_stack((outer_x, ys)).astype(np.int32)
    polygon = np.vstack((left_curve, right_curve[::-1]))
    region = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(region, [polygon], 255)
    left = ((int(center_x[-1]), height - 1), (int(center_x[0]), y_top))
    right = ((int(outer_x[-1]), height - 1), (int(outer_x[0]), y_top))
    return region, left_curve, right_curve, left, right


def _boundary_x_at_y(line: tuple[tuple[int, int], tuple[int, int]], y: int) -> float:
    (x1, y1), (x2, y2) = line
    if y2 == y1:
        return float(x1)
    return x1 + (x2 - x1) * ((y - y1) / (y2 - y1))


def create_lane_region_mask(shape: tuple[int, ...], left: tuple[tuple[int, int], tuple[int, int]],
                            right: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    """Fill the polygon between boundaries, including support for crossing curves."""
    height, width = shape[:2]
    top = int(height * config.ROI_TOP_FRACTION)
    ys = np.linspace(top, height - 1, 12).astype(int)
    polygon = []
    for y in ys:
        polygon.append((int(np.clip(_boundary_x_at_y(left, int(y)), 0, width - 1)), int(y)))
    for y in ys[::-1]:
        polygon.append((int(np.clip(_boundary_x_at_y(right, int(y)), 0, width - 1)), int(y)))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
    return mask


def create_final_binary_mask(lane_region: np.ndarray) -> np.ndarray:
    """Return 0 inside the lane polygon and 255 outside it."""
    return cv2.bitwise_not(lane_region)


def _draw_line(image: np.ndarray, line: Optional[tuple[tuple[int, int], tuple[int, int]]],
               colour: tuple[int, int, int]) -> None:
    if line is not None:
        cv2.line(image, line[0], line[1], colour, max(2, image.shape[1] // 180), cv2.LINE_AA)


def process_image(image: np.ndarray) -> LaneResult:
    height, width = image.shape[:2]
    gray, blurred = preprocess_image(image)
    colour = create_colour_mask(image)
    edges = cv2.Canny(blurred, config.CANNY_LOW, config.CANNY_HIGH)
    # Colour support keeps lane paint salient; Canny still detects the road edge.
    combined = cv2.bitwise_or(edges, cv2.Canny(colour, config.CANNY_LOW, config.CANNY_HIGH))
    roi = create_roi_mask(image.shape)
    roi_edges = cv2.bitwise_and(combined, roi)
    segments = detect_line_segments(roi_edges, roi)
    left, right = estimate_lane_boundaries(segments, width, height)
    fallback = None
    curved_boundaries = None
    lane_region = None
    top, bottom = int(height * config.ROI_TOP_FRACTION), height - 1

    if left is None and right is None:
        failed_stages = {"original": image, "preprocessed": blurred, "threshold": colour,
                         "edges": edges, "roi_edges": roi_edges, "detected_lanes": image.copy()}
        raise LaneDetectionError("No lane boundaries detected; no final lane mask was generated.", failed_stages)
    # The supplied PS1 photo shows a white center marking and the far left road
    # shoulder, while the road to the right continues to the image edge. That
    # shoulder is not the other edge of the selected lane. When this distinctive
    # geometry occurs, keep the marking as the lane's left edge and estimate the
    # right edge using a perspective-scaled width. The CLI reports this inference.
    if (left is not None and right is not None and right[0][0] > width * .55
            and _bright_line_support(image, right) >= 18
            and _bright_line_support(image, right) > 2 * _bright_line_support(image, left)):
        marking_points = _white_marking_points(image)
        if len(marking_points) >= 12:
            lane_region, left_curve, right_curve, left, right = _curved_region_from_marking(
                image.shape, marking_points)
            curved_boundaries = (left_curve, right_curve)
            fallback = "right boundary inferred from the visible center marking and perspective-scaled lane width"
    if left is None:
        # Use the observed line and a visible, conservative lane-width ratio only when in-frame.
        left = ((int(np.clip(right[0][0] - width * .36, 0, width - 1)), bottom),
                (int(np.clip(right[1][0] - width * .19, 0, width - 1)), top))
        fallback = "left boundary inferred from the detected right boundary and a perspective-scaled width"
    elif right is None:
        right = ((int(np.clip(left[0][0] + width * .36, 0, width - 1)), bottom),
                 (int(np.clip(left[1][0] + width * .19, 0, width - 1)), top))
        fallback = "right boundary inferred from the detected left boundary and a perspective-scaled width"

    if lane_region is None:
        lane_region = create_lane_region_mask(image.shape, left, right)
    lane_fraction = float(np.count_nonzero(lane_region)) / lane_region.size
    if not config.MIN_LANE_AREA_FRACTION <= lane_fraction <= config.MAX_LANE_AREA_FRACTION:
        raise RuntimeError(f"Detected lane polygon has implausible area ({lane_fraction:.1%}); inspect ROI/thresholds.")
    final = create_final_binary_mask(lane_region)
    overlay = image.copy()
    if curved_boundaries is None:
        _draw_line(overlay, left, (0, 255, 0))
        _draw_line(overlay, right, (0, 165, 255))
    else:
        cv2.polylines(overlay, [curved_boundaries[0]], False, (0, 255, 0),
                      max(2, width // 180), cv2.LINE_AA)
        cv2.polylines(overlay, [curved_boundaries[1]], False, (0, 165, 255),
                      max(2, width // 180), cv2.LINE_AA)
    tinted = overlay.copy()
    if curved_boundaries is None:
        polygon = np.array([left[0], left[1], right[1], right[0]], dtype=np.int32)
    else:
        polygon = np.vstack((curved_boundaries[0], curved_boundaries[1][::-1]))
    cv2.fillPoly(tinted, [polygon], (255, 0, 255))
    overlay = cv2.addWeighted(tinted, .20, overlay, .80, 0)
    diagnostics = {"original": image, "preprocessed": blurred, "threshold": colour,
                   "edges": edges, "roi_edges": roi_edges, "detected_lanes": overlay,
                   "lane_region": lane_region, "final_mask": final}
    return LaneResult(left, right, fallback, lane_region, final, diagnostics, curved_boundaries)


def validate_and_save_mask(mask: np.ndarray, path: Path, expected_shape: tuple[int, int]) -> None:
    if mask.ndim != 2 or mask.shape != expected_shape:
        raise ValueError("Final mask must be single-channel and match the input dimensions.")
    if not set(np.unique(mask).tolist()).issubset({0, 255}):
        raise ValueError("Final mask contains values other than 0 and 255.")
    if np.all(mask == 0) or np.all(mask == 255):
        raise ValueError("Final mask is all one colour; refusing to report success.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask):
        raise OSError(f"Could not write final mask to {path}")
    reopened = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if reopened is None or reopened.shape != expected_shape or not set(np.unique(reopened).tolist()).issubset({0, 255}):
        raise OSError("Saved mask could not be reopened and validated.")


def save_outputs(result: LaneResult, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = {"original": "01_original.png", "preprocessed": "02_preprocessed.png",
             "threshold": "03_threshold_mask.png", "edges": "04_edges.png",
             "roi_edges": "05_roi_edges.png", "detected_lanes": "06_detected_lanes.png",
             "lane_region": "07_lane_region.png", "final_mask": "08_final_binary_mask.png"}
    for stage, filename in names.items():
        arr = result.diagnostics[stage]
        if not cv2.imwrite(str(out / filename), arr):
            raise OSError(f"Failed to save {stage} image")
    validate_and_save_mask(result.final_mask, out / names["final_mask"], result.final_mask.shape)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(result.diagnostics["original"], cv2.COLOR_BGR2RGB)); axes[0].set_title("Input road image")
    axes[1].imshow(cv2.cvtColor(result.diagnostics["detected_lanes"], cv2.COLOR_BGR2RGB)); axes[1].set_title("Estimated lane boundaries")
    axes[2].imshow(result.final_mask, cmap="gray", vmin=0, vmax=255); axes[2].set_title("Final: lane black, background white")
    for ax in axes: ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "09_diagnostic_comparison.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def save_failure_diagnostics(diagnostics: dict[str, np.ndarray], output_dir: str | Path) -> None:
    """Save only the stages that ran when lane estimation fails."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = {"original": "01_original.png", "preprocessed": "02_preprocessed.png",
             "threshold": "03_threshold_mask.png", "edges": "04_edges.png",
             "roi_edges": "05_roi_edges.png", "detected_lanes": "06_detected_lanes.png"}
    for stage, filename in names.items():
        if not cv2.imwrite(str(out / filename), diagnostics[stage]):
            raise OSError(f"Failed to save failure diagnostic stage: {stage}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(cv2.cvtColor(diagnostics["original"], cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input image")
    axes[1].imshow(diagnostics["roi_edges"], cmap="gray")
    axes[1].set_title("ROI edges - no lane fit")
    for ax in axes: ax.axis("off")
    fig.suptitle("Lane detection failed; no final mask was created")
    fig.tight_layout(); fig.savefig(out / "09_detection_failure.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Road image path")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for diagnostic images")
    args = parser.parse_args()
    try:
        image = load_image(args.input)
        print(f"Input: {args.input.resolve()}\nDimensions: {image.shape[1]} x {image.shape[0]}, channels: {image.shape[2]}")
        result = process_image(image)
        save_outputs(result, args.output_dir)
    except LaneDetectionError as exc:
        try:
            save_failure_diagnostics(exc.diagnostics, args.output_dir)
        except OSError as save_error:
            parser.exit(2, f"Warning: {exc}\nCould not save diagnostic stages: {save_error}\n")
        parser.exit(2, f"Warning: {exc}\nDiagnostic stages saved to: {args.output_dir.resolve()}\n")
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print("Completed: grayscale/blur, colour mask, Canny, ROI, Hough lines, lane polygon, binary validation.")
    print(f"Both boundaries available: {result.fallback is None}")
    if result.fallback: print(f"Fallback used: {result.fallback}")
    print(f"Outputs saved to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
