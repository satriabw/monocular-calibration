# Monocular Calibration

Monocular video calibration using pinhole camera model with support for various camera distortion models.


## Run the test on a GPU server

**Current validation status:** 14 lightweight tests pass locally, including a
synthetic image-to-image calibration test. Public Hasselt aerial-image retrieval
and AnyCalib intrinsic inference were exercised. Full learned camera-to-aerial
calibration has **not** been validated: local GPU-model work was stopped at the
user's request. The AnyCalib weights and unfinished downloads were removed from
the Mac. Do not treat the prototype as a verified Hasselt calibration yet.

### 1. Transfer the project from the Mac

Replace `USER` and `SERVER` with your SSH login. This copies source and sample
data, excluding the Mac environment, model caches, and local results:

```bash
rsync -av \
  --exclude '.venv/' --exclude '.cache/' --exclude '.git/' \
  --exclude '__pycache__/' --exclude '.pytest_cache/' --exclude 'results/' \
  /Users/satriabw/Downloads/monocular-calibration/ \
  USER@SERVER:~/monocular-calibration/
```

The automatic test does not need either of the supplied control-point CSVs.
They are useful only for a separate comparison. To copy them as well:

```bash
scp /Users/satriabw/Downloads/points_lambert.csv \
    /Users/satriabw/Downloads/Remapped_Control_Points.csv \
    USER@SERVER:~/monocular-calibration/data/hasselt/
```

### 2. Install on the server

Run the remaining commands **on the GPU server**, from the project directory.
Use Python 3.12 and an NVIDIA GPU with a compatible installed driver.

```bash
cd ~/monocular-calibration
nvidia-smi
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-auto.txt
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: fix the PyTorch/driver installation before running inference"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

If the CUDA check fails, install the PyTorch build appropriate for the server's
driver using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/),
then repeat the check. Do not proceed with CPU fallback for this server test.
The optional dependencies pin AnyCalib to a source revision and RoMa to a release.
First inference needs Internet access and several GB of model downloads. Models
are cached in `.cache/`; later runs reuse them. Camera images are processed on
the server, not uploaded to an inference service.

### 3. Run the lightweight tests first

```bash
python -m pytest -q
```

Expected: **14 passed**. These tests do not download model weights or contact
imagery services. They test geometry, distortion handling, outlier rejection,
reference coordinates, video consistency, bounded focal refinement, and a
synthetic automatic-matching CLI run.

### 4. Run the real Hasselt automatic test

This command downloads a 40-metre aerial crop from the public Flemish service,
estimates camera intrinsics, matches the images with full RoMa, and validates
the resulting pose. The approximate location comes from the existing sample
coordinates. No manually selected correspondences are used for this run.

```bash
python auto_calibrate.py \
  --image data/hasselt/Background.bmp \
  --location 50.926946 5.336048 \
  --reference-width 40 \
  --matcher roma \
  --output-dir results/hasselt-gpu
```

First model downloads can take time. Exit code **0** means the internal checks
passed; **2** means rejected or partial calibration. A rejection is a valid test
outcome and must not be fixed simply by loosening thresholds.

### 5. Inspect the result

```bash
python -m json.tool results/hasselt-gpu/quality_report.json
```

For success, check `status`, `selected_frame`, and the frame's
`validation_median_px`, `validation_inlier_fraction`, `image_coverage`, and
`camera_center_enu_m`. Inspect `overlay_0.png` and the downloaded `reference.png`
visually. The ground-plane model must align the intended crossing, not merely
another repeated pattern or a roof. Internal match agreement alone does not
prove metric accuracy.

`camera_model.yml` is exported only on success. `intrinsics_0.json` preserves the
initial lens estimate; the final lens parameters in the model/report may differ
after bounded focal refinement. `matches_0.csv` preserves candidate matches even
when pose fitting rejects them. No calibration model is exported on failure.

If the imagery service is unavailable after a successful download, rerun with
that saved reference and a fresh output directory:

```bash
python auto_calibrate.py \
  --image data/hasselt/Background.bmp \
  --reference results/hasselt-gpu/reference.png \
  --georef results/hasselt-gpu/reference.json \
  --matcher roma \
  --output-dir results/hasselt-gpu-repeat
```

For a smaller matcher, explicitly select `--matcher tiny-roma` and a fresh output
directory. It uses the official TinyRoMa model and a pinned XFeat dependency;
its accuracy on this scene is also unverified. Full RoMa remains the primary test.

To compare the two manually annotated tables after copying them:

```bash
python scripts/compare_control_points.py \
  data/hasselt/points_lambert.csv \
  data/hasselt/Remapped_Control_Points.csv \
  --output results/control-point-comparison.json
```

On the supplied files, leave-one-out planar RMS was **21.43 px** for
`points_lambert.csv` and **17.29 px** for `Remapped_Control_Points.csv`. The latter
is more internally consistent, but CSV line 25 (camera pixel `485,248`) is suspect:
its coordinate differs by about 1 m between files and has about 49 px held-out
error. Neither table is certified ground truth. Column names `lambert_lat` and
`lambert_lon` were interpreted as Lambert 72 (EPSG:31370) easting and northing,
consistent with the known Hasselt location.

## Automatic calibration (prototype)

The automatic workflow needs a camera image (or a fixed-camera video) and a
photographic overhead reference with metric coordinates. It does not require a
control-point CSV. For Flanders, provide an approximate location and the program
fetches a public Digitaal Vlaanderen orthophoto with its coordinate metadata.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-auto.txt
python auto_calibrate.py --image data/hasselt/Background.bmp \
  --location 50.926946 5.336048 --output-dir results/hasselt-auto
```

First inference downloads several GB of model weights into `.cache/` (override
with `--cache-dir`) and requires Internet access. CPU and
CUDA are supported; CPU inference can be slow. The default matcher is RoMa.
`--matcher sift` runs a lightweight baseline, which is unlikely to handle extreme
viewpoint differences. AnyCalib currently initializes a simple pinhole model;
for a distorted lens, supply known OpenCV intrinsics using `--intrinsics`:

```json
{"camera_matrix": [[800, 0, 640], [0, 800, 360], [0, 0, 1]],
 "dist_coeffs": [0.02, -0.01, 0, 0, 0]}
```

These are illustrative values, not Hasselt's measured calibration. Intrinsics
must correspond to the input image resolution, crop, and zoom. With known
intrinsics and SIFT, only `requirements.txt` is needed.

For a local reference, replace `--location` with
`--reference aerial.png --georef aerial.json`. Metadata format:

```json
{"crs": "EPSG:31370", "origin": [218004, 180070],
 "pixel_to_world": [[0.1, 0, 217954.05], [0, -0.1, 180119.95]]}
```

`pixel_to_world` maps zero-based image pixel centres to projected easting/northing
in metres. The example is a north-up image at 10 cm/pixel. Rotation is allowed;
coordinate CRS must use metres. A screenshot's dimensions alone are insufficient
to derive this transform. Drawn maps and screenshots with markers are not ideal
matching references. The public reference path avoids those annotations.

For video use `--video clip.mp4 --frames 5`; only stationary cameras with fixed
zoom are supported. At least two frames must pass and their poses and focal lengths must agree.
The most reliable accepted frame is exported; this is not joint video optimization.

Outputs:

- `camera_model.yml`: written only for a successful calibration.
- `quality_report.json`: success/partial/failed, thresholds' measurements, camera
  position, reference metadata, per-frame results, and failure reasons.
- `matches_<frame>.csv`: automatic correspondences, confidence, and pixel errors.
- `overlay_<frame>.png`: green/red reprojection residuals.
- `intrinsics_<frame>.json`: lens estimate and image dimensions for reuse.
- `reference.png` and `reference.json`: downloaded aerial reference, where used.

Exit code is 0 for success and 2 for rejected/partial calibration. Use a fresh
output directory for each run; an existing camera model is never overwritten.
Outputs from this workflow use **local ENU metres**, relative to the recorded
projected origin, with ground at z=0. Legacy CSV workflows use NED. Both branches
of `CameraModel.project_to_ground` now return an N-by-2 array and the pose branch
accounts for OpenCV lens distortion.

Quality checks include held-out correspondences (split before pose fitting),
training inliers, image coverage, positive depth, and plausible camera height.
Learned focal length may be refined within a factor of 1.2 using training
correspondences only; supplied intrinsics are held fixed. Default limits are
4 px residual and 5% image coverage, configurable with
`--max-error` and `--min-coverage`. These checks measure internal consistency,
not surveyed geographic accuracy. Repeated markings, roofs, non-flat roads,
shadows, old aerial imagery, and very different viewpoints can still cause
incorrect matches. No semantic ground segmentation is implemented yet.

Validation:

```bash
python -m pytest -q
```

The suite covers synthetic pose recovery with distortion/outliers, rejected
matches, coverage, georeferencing, save/load, failure reports, and an image-matching
CLI integration test. A successful synthetic test does not establish accuracy
on an untested real intersection.

Public imagery source: [Digitaal Vlaanderen WMS](https://geo.api.vlaanderen.be/OMWRGBMRVL/wms?SERVICE=WMS&REQUEST=GetCapabilities).
The provider covers Flanders only. The requested location is sent to that public
service; input camera images are processed locally.

## Overview

This repository provides tools for calibrating monocular cameras from video data using the pinhole camera model. It integrates AnyCalib, a deep learning-based camera calibration library, for automatic intrinsic parameter estimation along with traditional calibration methods.

## Features

- **Multiple Calibration Methods**:
  - AnyCalib (deep learning-based intrinsic estimation)
  - OpenCV traditional calibration
  - Homography-based calibration
  - Custom focal length estimation with optimization

- **Camera Model Support**:
  - Pinhole camera model
  - Support for various distortion models (radial, division, fisheye, etc.)
  - Extrinsic parameter estimation (rotation and translation)

- **Coordinate Systems**:
  - Latitude/Longitude to NED (North-East-Down) conversion
  - CARLA coordinate system support
  - EPSG projection support via pyproj

- **Visualization**:
  - Calibration result visualization with reprojected points
  - Comparison of actual vs. reprojected points

## Project Structure

```
monocular-calibration/
├── calibrate_camera.py          # Main calibration script
├── camera_model.py              # Camera model class for saving/loading
├── scripts/
│   └── create_calibration_file.py
├── data/
│   ├── hasselt/                 # Sample calibration data (Hasselt dataset)
│   └── hasselt-bev/             # Bird's Eye View sample data
└── libs/
    └── AnyCalib/                # AnyCalib library for intrinsics prediction
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd monocular-calibration

# Install dependencies
pip install -r requirements.txt

# For AnyCalib support (optional)
cd libs/AnyCalib
pip install -e .
```

### Requirements

- Python 3.10+ (tested with 3.12)
- NumPy
- OpenCV (cv2)
- Pillow
- Pandas
- PyTorch (for AnyCalib)
- SciPy
- AnyCalib and RoMa (optional automatic inference dependencies)

## Usage

### Basic Calibration

```bash
python calibrate_camera.py \
    --image <path_to_image> \
    --csv <path_to_calibration_points_csv> \
    --method anycalib \
    --output_dir <output_directory> \
    --visualize
```

### Parameters

- `--image`: Path to the input image for calibration
- `--csv`: Path to CSV file containing calibration points in format: `x, y, lat, lon, origin_lat, origin_lon`
- `--method`: Calibration method (`anycalib`, `opencv`, `estimate`, `homography`) - default: `anycalib`
- `--satellite_mode`: Enable satellite mode (sets focal length to image height)
- `--carla_mode`: Use CARLA coordinate conversion instead of lat/lon
- `--epsg`: Enable projection mode with specified EPSG code (e.g., `EPSG:24047`)
- `--visualize`: Generate visualization of calibration results
- `--output_dir`: Directory to save output files (camera model and visualization)

### Example with EPSG Projection

```bash
python calibrate_camera.py \
    --image /path/to/image.jpg \
    --csv /path/to/calibration_points.csv \
    --method anycalib \
    --epsg EPSG:24047 \
    --visualize \
    --output_dir ./results
```

## CSV Format

The calibration CSV file should have the following columns:

```
x,y,lat,lon,origin_lat,origin_lon
```

- `x, y`: Pixel coordinates in the image
- `lat, lon`: Latitude and longitude of the world point
- `origin_lat, origin_lon`: Reference origin for coordinate transformation

## Calibration Methods

### AnyCalib
Deep learning-based automatic intrinsic parameter estimation. Trained models available for:
- `anycalib_pinhole`: Perspective/pinhole images only
- `anycalib_gen`: Perspective, distorted, and strongly distorted images
- `anycalib_dist`: Distorted and strongly distorted images
- `anycalib_edit`: Edited (stretched and cropped) perspective images

### OpenCV
Traditional camera calibration using multiple views (requires multiple images/frames).

### Homography
Direct 2D transformation when assuming planar scene.

### Estimate
Custom focal length estimation using optimization with reprojection error minimization.

## Output

The script generates:

1. **camera_model.yml**: Calibrated camera parameters including:
   - Camera intrinsic matrix (K)
   - Distortion coefficients
   - Rotation matrix
   - Translation vector
   - Homography matrix (if applicable)

2. **calibration_viz.png**: Visualization showing:
   - Green circles: Actual detected points
   - Red circles: Reprojected points from the calibration

## Calibration Error Metrics

- **RMS (Root Mean Square) Error**: Reprojection error between detected and projected points
- Lower RMS values indicate better calibration accuracy

## AnyCalib Integration

This project integrates [AnyCalib](https://github.com/javrtg/AnyCalib) for deep learning-based camera intrinsics prediction. The library supports multiple camera models beyond the standard pinhole model.

## Coordinate Systems

### NED (North-East-Down)
Default coordinate system conversion:
```
North = (lat_diff) × 111318.845 m/degree
East = (lon_diff) × 111318.845 × cos(latitude) m/degree
Down = 0 (planar assumption)
```

### CARLA
When using `--carla_mode`, coordinates are treated as 3D XYZ points and converted to NED directly.

### EPSG Projections
Custom projections can be specified using EPSG codes (e.g., UTM zones).

## Scripts

### create_calibration_file.py
Helper script to create calibration CSV files from annotated data.

## Sample Data

The `data/` directory contains sample calibration data:
- `hasselt/`: Standard view calibration data
- `hasselt-bev/`: Bird's Eye View calibration data

## License

See LICENSE file for details.

## References

- Zhang, Z. (2000). A flexible new technique for camera calibration. IEEE transactions on pattern analysis and machine intelligence, 22(11), 1330-1334.
- AnyCalib: https://github.com/javrtg/AnyCalib

### Comparing existing Lambert control points

Tables with `x,y,lambert_lat,lambert_lon` are supported by the legacy workflow
when the CRS is explicitly supplied. Despite these historical column names,
`lambert_lat` is interpreted as easting and `lambert_lon` as northing.

```bash
python calibrate_camera.py --image data/hasselt/Background.bmp \
  --csv Remapped_Control_Points.csv --csv-crs EPSG:31370 \
  --method homography --output_dir results/manual-baseline --visualize
python scripts/compare_control_points.py points_lambert.csv Remapped_Control_Points.csv \
  --output results/control-point-comparison.json
```

The comparison fits planar homographies and reports leave-one-out pixel errors.
Lower error indicates better internal agreement, not proven absolute accuracy.
The legacy homography output is a ground mapping only: its identity rotation
and zero translation are placeholders, not a calibrated physical pose.
