# Experiment report

## Images and scope

- **Official assignment image:** `input/official_ps1_road.jpg`, extracted without rescaling from the image embedded on page 5 of the supplied induction PDF (492 x 396, RGB source). It contains a curved rural road and one prominent white painted marking; the opposite lane edge is visually less clear in the low-resolution PDF image.
- **Synthetic tests:** generated in memory by `tests/test_lane_detection.py`, with a dark road-like background and two high-contrast converging white boundaries. Separate synthetic inputs omit one boundary, omit both, and use a small resolution. These are controlled software checks, not real-road performance evidence.

## Parameters and expected observations

Defaults are in `config.py`: 5 x 5 Gaussian kernel, Canny thresholds 50/150, Hough threshold 18, minimum segment length 18 pixels, maximum gap 35 pixels, ROI top at 48% of image height. White paint uses HSV value >=165 and saturation <=90; yellow paint uses H 12..42, S >=65 and V >=95. These choices are fixed starting values, not tuned on a labelled dataset.

The script writes original, blurred, threshold, Canny, ROI edge, fitted boundary overlay, filled lane polygon and final binary mask stages. It also generates a three-panel side-by-side diagnostic. Final mask output is validated to be single-channel, input-sized, binary-valued and non-constant. Lane pixels are black and surrounding pixels white.

## Results log

**Verification status:** the research PDF was generated and reopened as a 4-page PDF; its extracted text was checked for completeness and its rendered pages were inspected. The official 492 x 396 image was processed with OpenCV 4.14.0.94, NumPy 2.5.3, and Matplotlib 3.11.2. Both unit and integration tests passed (7 tests).

The first diagnostic exposed a lane-side error: the Hough fit paired the far left shoulder with the white center marking, producing a black mask over the left part of the road. The code now selects the high-brightness white marking evidence, traces its row-wise center points, and fits a quadratic curve. It retains the white center marking as the selected lane's left boundary and estimates the right boundary using a perspective-scaled width. The final mask was visually inspected: the black region lies to the right of the painted center marking, on the road surface, and the surrounding area is white. The right edge remains an inference, not a measured marking. The ROI top was moved down from 42% to 48% of image height to avoid covering sky above the visible roadway.

The mask checks confirmed matching 492 x 396 dimensions, one grayscale channel, values limited to 0 and 255, a non-constant region, successful save, and successful reopen. No precision/recall or accuracy is reported because ground-truth pixel annotations are not provided.

## Limitations and failure modes

- Hough fitting assumes locally straight line segments; curved roads are represented only approximately.
- The ROI encodes a camera-view assumption and can cut off useful evidence in a differently framed image.
- Bright, low-saturation HSV values can include pale road, sky, or glare; thresholds shift under lighting and camera changes.
- Road seams, shadows, and texture may generate competing edges.
- A missing side is estimated with a perspective-scaled width offset and must be interpreted as an inference. It can be wrong on curves, unusual lane widths, or a different lane position.
- The provided embedded image is small and provides no annotation ground truth.
- This is a single-image educational pipeline, not a safety-critical autonomous driving system.
