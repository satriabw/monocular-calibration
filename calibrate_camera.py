import numpy as np
import torch
from PIL import Image  # the library of choice to load images
import pandas as pd  # for reading the pixel coordinates from a CSV file
import cv2
import scipy.optimize as opt
import math
import argparse
import os

import sys
sys.path.append('./libs/AnyCalib')
from anycalib import AnyCalib
from camera_model import CameraModel

LOCATION_SCALING_FACTOR = 111318.84502145034
LOCATION_SCALING_FACTOR_INV = 0.000008983204953368922


# ----- Callibration Functions -----
#  Taken from https://github.com/AubreyC/trajectory-extractor/blob/master/traj_ext/
def get_scale_longitude_factor(lat):
    scale = math.cos(math.radians(lat))
    return scale

def latlon_to_NED(latlon_origin, latlon_point):
    origin = latlon_origin[0]
    lat_diff = latlon_point[:, 0] - origin[0]
    lon_diff = latlon_point[:, 1] - origin[1]

    north = lat_diff * LOCATION_SCALING_FACTOR
    east = lon_diff * LOCATION_SCALING_FACTOR * get_scale_longitude_factor(origin[0])

    return np.column_stack((north, east))

def convert_latlon_F(latlon_origin, latlon_points):
    # Convert lat/lon points to NED coordinates
    ned_2d = latlon_to_NED(latlon_origin, latlon_points)
    # Add a zero column for the vertical component
    ned_3d = np.column_stack([ned_2d, np.zeros((ned_2d.shape[0], 1))])
    return ned_3d

def xyz_to_NED(origin_xyz, points_xyz):
    """
    Convert 3D points (x, y, z) in meters to NED coordinates relative to origin.
    """
    # Calculate differences from origin
    diff = points_xyz - origin_xyz
    
    # Convert to NED coordinates:
    # North = difference in Y direction (positive Y is North)
    # East = difference in X direction (positive X is East)  
    # Down = negative difference in Z direction (positive Z is Up, but NED uses Down)
    ned_coordinates = np.column_stack([
        diff[:, 1],   # North (Y difference)
        diff[:, 0],   # East (X difference)
        np.zeros(diff.shape[0])  # Down (negative Z difference)
    ])
    
    return ned_coordinates

def convert_xyz_to_NED(origin_xyz, points_xyz):
    """
    Convert 3D points (x, y, z) in meters to NED coordinates.
    This is a wrapper function that matches the naming convention of convert_latlon_F.
    """
    return xyz_to_NED(origin_xyz, points_xyz)

# Anycallib camera instrinsics prediction
# This function predicts camera intrinsics using the AnyCalib library.
# It takes an image as input and returns the focal length and principal point (cx, cy).
# The model_id can be adjusted based on the type of images being processed.
def predict_camera_intrinsics(image):
    dev = torch.device("cuda")

    # load input image and convert it to a (3, H, W) tensor with RGB values in [0, 1]
    image = torch.tensor(image, dtype=torch.float32, device=dev).permute(2, 0, 1) / 255

    # instantiate AnyCalib according to the desired model_id. Options:
    # "anycalib_pinhole": model trained with *only* perspective (pinhole) images,
    # "anycalib_gen": trained with perspective, distorted and strongly distorted images,
    # "anycalib_dist": trained with distorted and strongly distorted images,
    # "anycalib_edit": Trained on edited (stretched and cropped) perspective images.
    model = AnyCalib(model_id="anycalib_pinhole").to(dev)
    
    # predict according to the desired camera model. Implemented camera models are detailed further below.
    output = model.predict(image, cam_id="simple_pinhole")

    # Estimated intrinsics for the selected camera model
    intrinsics = output["intrinsics"].to('cpu')
    focal_length = intrinsics[0]
    cx = intrinsics[1]
    cy = intrinsics[2]

    return focal_length.item(), (cx.item(), cy.item())


# Source: https://github.com/SoccerNet/sn-calibration/blob/main/src/camera.py
# Algorithm 8.2 of Multiple View Geometry in computer vision, p225
def get_K_from_homography(H, image_size):
        H = np.reshape(H, (9,))
        A = np.zeros((5, 6))
        A[0, 1] = 1.
        A[1, 0] = 1.
        A[1, 2] = -1.
        A[2, 3] = image_size[0] / image_size[1] # Principal point set to image center
        A[2, 4] = -1.0
        A[3, 0] = H[0] * H[1]
        A[3, 1] = H[0] * H[4] + H[1] * H[3]
        A[3, 2] = H[3] * H[4]
        A[3, 3] = H[0] * H[7] + H[1] * H[6]
        A[3, 4] = H[3] * H[7] + H[4] * H[6]
        A[3, 5] = H[6] * H[7]
        A[4, 0] = H[0] * H[0] - H[1] * H[1]
        A[4, 1] = 2 * H[0] * H[3] - 2 * H[1] * H[4]
        A[4, 2] = H[3] * H[3] - H[4] * H[4]
        A[4, 3] = 2 * H[0] * H[6] - 2 * H[1] * H[7]
        A[4, 4] = 2 * H[3] * H[6] - 2 * H[4] * H[7]
        A[4, 5] = H[6] * H[6] - H[7] * H[7]

        # May not converge
        u, s, vh = np.linalg.svd(A)
        w = vh[-1]
        W = np.zeros((3, 3))
        W[0, 0] = w[0] / w[5]
        W[0, 1] = w[1] / w[5]
        W[0, 2] = w[3] / w[5]
        W[1, 0] = w[1] / w[5]
        W[1, 1] = w[2] / w[5]
        W[1, 2] = w[4] / w[5]
        W[2, 0] = w[3] / w[5]
        W[2, 1] = w[4] / w[5]
        W[2, 2] = w[5] / w[5]

        Ktinv = np.linalg.cholesky(W)
        K = np.linalg.pinv(Ktinv.T)
        K /= K[2, 2]


        fx = K[0, 0]
        fy = K[1, 1]
        cx = image_size[1] / 2.0
        cy = image_size[0] / 2.0
        return fx, fy, cx, cy


def estimate_camera_intrinsics(image_size, image_points, model_points_3d, satellite_mode=False, objective_func=None, guess_focal=False):
    if satellite_mode:
        # Set focal length to image height for satellite mode
        return image_size[0], (image_size[1] / 2, image_size[0] / 2)
    
    # Initial guess
    if guess_focal:
        initial_focal = [image_size[1]]
        cx, cy = image_size[1] / 2, image_size[0] / 2
    else:
        H, _ = cv2.findHomography(model_points_3d[:, :2], image_points, method=cv2.RANSAC)
        fx, fy, cx, cy = get_K_from_homography(H, image_size)
        initial_focal = [(fx + fy) / 2.0]

    # Constraint: focal length must be positive
    constraints = {'type': 'ineq', 'fun': (lambda x: x[0])}
    
    # Run optimization
    result = opt.minimize(
        objective_func,
        initial_focal,
        constraints=constraints,
        args=((cx, cy), image_points, model_points_3d)
    )
    focal_length = result.x[0]
    
    return focal_length, (cx, cy)

def calculate_reprojection_rms(predicted_points, actual_points):
    """
    Calculate the RMS reprojection error between predicted and actual points.
    """
    # Calculate squared differences
    squared_diffs = np.sum((predicted_points - actual_points)**2, axis=1)
    
    # Calculate mean squared error
    mean_squared_error = np.mean(squared_diffs)
    
    # Calculate RMS error
    rms_error = np.sqrt(mean_squared_error)
    
    return rms_error

def get_reprojection_error(opti_params, center, image_points, model_points_F):
    # Camera internals
    focal_length = opti_params[0]

    # Find camera parms
    _, _, _, _, _, image_points_reproj = find_camera_params(focal_length, center, image_points, model_points_F)

    # Compute error
    error_reproj = calculate_reprojection_rms(image_points_reproj, image_points)

    return error_reproj

def find_camera_params(focal_length, center, image_points, model_points_F):
    camera_matrix = build_camera_matrix(focal_length, center)

    _, rot, trans = cv2.solvePnP(model_points_F, image_points, camera_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE)

    imagePoints, _ = cv2.projectPoints(model_points_F, rot, trans, camera_matrix, None)
    image_points_reproj = imagePoints[:,0]

    return None, camera_matrix, np.zeros((4, 1), dtype=np.float32), np.array([rot], dtype=np.float32), np.array([trans], dtype=np.float32), image_points_reproj

def build_camera_matrix(focal_length, center):
    cx, cy = center
    
    # Build intrinsic matrix
    camera_matrix = np.array([
        [focal_length, 0.0, cx],
        [0.0, focal_length, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    return camera_matrix

def find_camera_params_cv2(image_points, model_points_F, image_size):
    # Convert inputs to appropriate types
    image_points = np.array([image_points], dtype=np.float32)
    model_points_F = np.array([model_points_F], dtype=np.float32)

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        model_points_F, image_points, image_size, None, None
    )

    # Project points to get reprojection
    image_points_reproj, _ = cv2.projectPoints(
        model_points_F[0], rvecs[0], tvecs[0], mtx, dist
    )
    image_points_reproj = image_points_reproj[:, 0]

    return ret, mtx, dist, rvecs, tvecs, image_points_reproj


def visualize_calibration(image, image_points, image_points_reproj):
    image_viz = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    for pt in image_points:
        cv2.circle(image_viz, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)  # Green for actual points
    for pt in image_points_reproj:
        cv2.circle(image_viz, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)  # Red for reprojected points
    return image_viz
    
if __name__ == "__main__":
    # Parse command line arguments
    argsparser = argparse.ArgumentParser(description="Camera Calibration Script")
    argsparser.add_argument("--image", type=str, required=True, help="Path to the input image")
    argsparser.add_argument("--csv", type=str, required=True, help="Path to the CSV file with pixel coordinates")
    argsparser.add_argument("--method", type=str, required=False, default="anycalib", 
                          help="Method for camera calibration (anycalib, opencv, estimate, or homography)")
    argsparser.add_argument("--satellite_mode", action="store_true", help="Enable satellite mode for calibration")
    argsparser.add_argument("--carla_mode", action="store_true", help="Use CARLA coordinate conversion instead of lat/lon")
    argsparser.add_argument("--visualize", action="store_true", help="Visualize calibration results")
    argsparser.add_argument("--output_dir", type=str, required=False, default=".", help="Directory to save output files")
    argsparser.add_argument("--epsg", type=str, help="Enable projection mode")
    args = argsparser.parse_args()

    # Load image
    image_path = args.image
    image = Image.open(image_path)
    image_np = np.array(image)

    # Load CSV
    csv_path = args.csv
    df = pd.read_csv(csv_path)
    
    # Extract pixel and world coordinates from CSV as numpy arrays
    pixel_points = df[['x', 'y']].values
    world_points = df[['lat', 'lon']].values
    origin_points = np.array(df[['origin_lat', 'origin_lon']].values)
    
    # Transform coordinates if EPSG is specified
    if args.epsg:
        from pyproj import Transformer
        transformer = Transformer.from_crs("epsg:4326", args.epsg, always_xy=True)
        world_points = [transformer.transform(lon, lat) for lat, lon in world_points]
        origin_points = [transformer.transform(lon, lat) for lat, lon in origin_points]
    
    # Ensure all points are in the correct format
    pixel_points = np.array(pixel_points, dtype=np.float32)
    world_points = np.array(world_points, dtype=np.float32)
    
    # Convert world points to NED coordinates using the origin
    if args.carla_mode or args.epsg:
        latlon_NED = convert_xyz_to_NED(origin_points, world_points)
    else:
        latlon_NED = convert_latlon_F(origin_points, world_points)

    # Step 1: Get camera intrinsics based on method
    if args.method == "anycalib":
        focal_length, center = predict_camera_intrinsics(image_np)
        camera_matrix = build_camera_matrix(focal_length, center)
        dist = np.zeros((4, 1), dtype=np.float32)
    elif args.method == "estimate":
        focal_length, center = estimate_camera_intrinsics(
            image_size=image_np.shape[:2],
            image_points=pixel_points,
            model_points_3d=latlon_NED,
            satellite_mode=args.satellite_mode,
            objective_func=get_reprojection_error
        )
        camera_matrix = build_camera_matrix(focal_length, center)
        dist = np.zeros((4, 1), dtype=np.float32)
    elif args.method == "opencv":
        _, camera_matrix, dist, rvecs, tvecs, image_points_reproj = find_camera_params_cv2(
            image_points=pixel_points,
            model_points_F=latlon_NED,
            image_size=image_np.shape[:2]
        )
    else:  # Default or "homography"
        # For homography method, initialize basic camera matrix
        cx, cy = image_np.shape[1] / 2, image_np.shape[0] / 2
        focal_length = image_np.shape[0]  # Default assumption
        camera_matrix = build_camera_matrix(focal_length, (cx, cy))
        dist = np.zeros((4, 1), dtype=np.float32)

    # Step 2: Calculate camera extrinsics and projection
    if args.method == "homography":
        # For homography, we just compute the direct 2D transformation
        H, _ = cv2.findHomography(latlon_NED[:, :2], pixel_points, method=cv2.RANSAC)
        rot_CF_F = np.eye(3)
        trans_CF_F = np.zeros((3, 1))
        image_points_reproj = cv2.perspectiveTransform(latlon_NED[:, :2].reshape(-1, 1, 2), H).reshape(-1, 2)
    else:
        # For all other methods, solve PnP
        if args.method == "opencv":
            # We already have rvecs, tvecs, image_points_reproj from find_camera_params_cv2
            rot_CF_F = cv2.Rodrigues(rvecs[0])[0]
            trans_CF_F = tvecs[0]
        else:  # anycalib or estimate
            _, _, _, rvecs, tvecs, image_points_reproj = find_camera_params(
                focal_length=focal_length,
                center=(camera_matrix[0, 2], camera_matrix[1, 2]),
                image_points=pixel_points,
                model_points_F=latlon_NED
            )
            rot_CF_F = cv2.Rodrigues(rvecs[0])[0]
            trans_CF_F = tvecs[0]

    # Step 3: Create camera model
    camera_model = CameraModel(
        camera_matrix=camera_matrix,
        dist_coeffs=dist,
        rot_matrix=rot_CF_F,
        tvec=trans_CF_F,
        homography=H if args.method == "homography" else None
    )
    
    # Save results
    base_filename = os.path.basename(image_path).split('.')[0]
    os.makedirs(args.output_dir, exist_ok=True)
    camera_model.save_to_yml(os.path.join(args.output_dir, "camera_model.yml"))

    # Visualize if requested
    if args.visualize:
        image_viz = visualize_calibration(image_np, pixel_points, image_points_reproj)
        cv2.imwrite(os.path.join(args.output_dir, "calibration_viz.png"), image_viz)
    
    # Print calibration error
    print("Calibration Error (RMS):", calculate_reprojection_rms(image_points_reproj, pixel_points))
