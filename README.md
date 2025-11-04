# Monocular Calibration

Monocular video calibration using pinhole camera model with support for various camera distortion models.

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

- Python 3.8+
- NumPy
- OpenCV (cv2)
- Pillow
- Pandas
- PyTorch (for AnyCalib)
- SciPy
- scikit-image (siclib)

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

This project integrates [AnyCalib](https://github.com/vccimaging/AnyCalib) for deep learning-based camera intrinsics prediction. The library supports multiple camera models beyond the standard pinhole model.

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
- AnyCalib: https://github.com/vccimaging/AnyCalib
