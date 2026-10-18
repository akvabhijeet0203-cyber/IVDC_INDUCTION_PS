# IVDC Autonomy 2026 - PS1 Computer Vision

This project implements a classical, explainable lane-region pipeline for Problem Statement 1. The official 492 x 396 RGB road image embedded on page 5 of the assignment is stored at `input/official_ps1_road.jpg`.

## Objective and output convention

Estimate the roadway lane region from image evidence and save a single-channel bitmap with **black (0) inside the estimated lane** and **white (255) outside**. The final output is a region mask, not merely lane lines drawn on the photograph. This small illustrative image has one prominent painted line and a curved road; a missing boundary may therefore be estimated from the observed line and a perspective-scaled lane width. Every such estimate is printed and documented as an inference.

## Pipeline

1. Load a BGR image and report its dimensions.
2. Convert to grayscale and Gaussian blur with a 5 x 5 kernel.
3. Threshold HSV pixels for bright, low-saturation white paint and yellow paint.
4. Combine Canny edges from the blurred image and colour mask.
5. Keep the lower road area with a resolution-scaled trapezoidal ROI beginning at 48% of image height.
6. Detect segments with probabilistic Hough transform; group by side and fit candidate boundaries. On the supplied photo, bright white scanline evidence tracks the curved center marking with a quadratic fit.
7. Since the photo does not clearly show the opposite lane edge, infer it from that marking using a perspective-scaled width, and report the assumption.
8. Fill the lane polygon and invert it to the required polarity.
8. Validate the mask dimensions, channels, binary values, saved file, and nonempty area.

## Structure

```text
config.py                       Tunable processing defaults
lane_detection.py               CLI, pipeline, validation and visualizations
input/official_ps1_road.jpg     Official image extracted from the assignment PDF
outputs/                        Generated stage images and comparison figure
research/PS1_research.md        Numbered theory answers and references
research/PS1_research.pdf       Print-ready research hand-in
reports/experiment_report.md    Observations and limitations
tests/test_lane_detection.py    Synthetic unit/integration tests
requirements.txt                Runtime dependencies
```

## Setup

Python 3.10 or newer is recommended. From the project directory:

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
python lane_detection.py --input input/official_ps1_road.jpg --output-dir outputs
```

Use `python lane_detection.py --help` to see options. For another image, replace the `--input` path. The pipeline handles image dimensions dynamically, but its simple ROI and fixed colour thresholds can need adjustment for a different camera, lighting, road layout, or scene.

## Outputs

`01_original.png`, `02_preprocessed.png`, `03_threshold_mask.png`, `04_edges.png`, `05_roi_edges.png`, `06_detected_lanes.png`, `07_lane_region.png`, `08_final_binary_mask.png`, and `09_diagnostic_comparison.png`. In stage 7 the lane itself is white to make the polygon visible; stage 8 is inverted to the required black-lane/white-background convention. The comparison figure shows the input, fitted boundary overlay, and final bitmap.

## Tests

```bash
python -m unittest discover -s tests -v
```

Synthetic fixtures cover two-boundary geometry, a missing line, no-line failure, a small image, loading errors, polarity, and saved output. An additional integration check protects the intended lane side on the official image. They do not establish performance on varied real roads. See `reports/experiment_report.md` for the actual run and test results.

## Research and references

The complete numbered answers and cited references are in [`research/PS1_research.md`](research/PS1_research.md), with a PDF copy at [`research/PS1_research.pdf`](research/PS1_research.pdf).

## Limitations

This is a single-frame demonstration, not a safety system. Straight fitted segments approximate curved boundaries. Fixed HSV thresholds respond to exposure and shadows. The supplied photo has one clearly visible marking; any opposite side derived from lane width is only a geometric estimate, not an observed boundary. Hough lines can respond to texture and road edges, and this implementation does not estimate uncertainty or temporal consistency. The PDF's embedded image is low resolution.

## Assistance acknowledgement

AI assistance was used to draft and review parts of the code, tests, documentation, and research outline. The implementation is classical computer vision; no pretrained model or copied external code is used. OpenCV documentation and the listed references informed the explanations. The author should review the generated mask and be ready to explain the pipeline, thresholds, Hough fitting, fallback geometry, and limitations.
