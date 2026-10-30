"""Automatic planar camera calibration against a georeferenced orthophoto.

Reference coordinates use a projected metric CRS. All poses use local ENU
(east, north, up), with the reference ground at z=0.
"""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path

import cv2
import numpy as np
from pyproj import CRS

from camera_model import CameraModel


def reference_coordinates(pixels, metadata):
    crs = CRS.from_user_input(metadata['crs'])
    if not crs.is_projected or any(abs(a.unit_conversion_factor - 1) > 1e-9 for a in crs.axis_info[:2]):
        raise ValueError('Reference CRS must be projected and use metres.')
    affine = np.asarray(metadata['pixel_to_world'], dtype=float)
    origin = np.asarray(metadata['origin'], dtype=float)
    if affine.shape != (2, 3) or origin.shape != (2,) or not np.isfinite(affine).all() or not np.isfinite(origin).all():
        raise ValueError('Expected finite 2x3 pixel_to_world and 2-element origin.')
    if abs(np.linalg.det(affine[:, :2])) < 1e-12:
        raise ValueError('Reference transform is singular.')
    xy = np.column_stack((pixels, np.ones(len(pixels)))) @ affine.T - origin
    return np.column_stack((xy, np.zeros(len(xy))))


@lru_cache(maxsize=1)
def roma_model(backend='roma'):
    import torch
    from romatch import roma_outdoor, tiny_roma_v1_outdoor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if backend == 'tiny-roma':
        # Pin the official upstream feature extractor; avoid interactive hub prompts.
        xfeat = torch.hub.load(
            'verlab/accelerated_features:e92685f57f8318b18725c5c8c0bd28c7fe188d9a',
            'XFeat', pretrained=True, top_k=4096, trust_repo=True).net
        model = tiny_roma_v1_outdoor(device=device, xfeat=xfeat).eval()
    else:
        # The fused CUDA kernel is optional; the native torch path gives the same result.
        try:
            import local_corr  # noqa: F401
            fused = device == 'cuda'
        except ImportError:
            fused = False
        model = roma_outdoor(device=device, use_custom_corr=fused).eval()
    # Preserve the original confidence values instead of the sampler's thresholded ones.
    model.sample_mode = 'balanced'
    return model, device


def match_images(camera, reference, backend='roma'):
    if backend in ('roma', 'tiny-roma'):
        import torch
        from PIL import Image
        model, device = roma_model(backend)
        with torch.inference_mode():
            options = {'device': device} if backend == 'roma' else {}
            warp, confidence = model.match(Image.fromarray(cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)),
                                           Image.fromarray(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)), **options)
            matches, confidence = model.sample(warp, confidence, num=2000)
            a, b = model.to_pixel_coordinates(matches, *camera.shape[:2], *reference.shape[:2])
        return a.cpu().numpy(), b.cpu().numpy(), confidence.cpu().numpy().reshape(-1)
    sift = cv2.SIFT_create(nfeatures=8000)
    ka, da = sift.detectAndCompute(cv2.cvtColor(camera, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None or len(db) < 2:
        raise ValueError('Not enough image texture to match.')
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < .7 * p[1].distance]
    # One reference feature must not vote multiple times.
    good = sorted(good, key=lambda m: m.distance)
    selected, seen = [], set()
    for m in good:
        if m.trainIdx not in seen:
            selected.append(m)
            seen.add(m.trainIdx)
    return (np.array([ka[m.queryIdx].pt for m in selected]).reshape(-1, 2),
            np.array([kb[m.trainIdx].pt for m in selected]).reshape(-1, 2),
            np.ones(len(selected)))


def coverage(points, shape):
    return float(cv2.contourArea(cv2.convexHull(np.asarray(points, np.float32))) / (shape[0] * shape[1]))


def solve(world, pixels, K, dist, image_shape, max_error=4., min_coverage=.05, refine_focal=False):
    world, pixels = np.asarray(world, np.float64), np.asarray(pixels, np.float64)
    if world.ndim != 2 or world.shape[1] != 3 or pixels.shape != (len(world), 2):
        raise ValueError('Expected Nx3 ground points and Nx2 image points.')
    if len(world) < 20 or not np.isfinite(world).all() or not np.isfinite(pixels).all():
        raise ValueError('Need at least 20 finite, distinct ground correspondences.')
    if not np.allclose(world[:, 2], 0) or np.linalg.matrix_rank(world[:, :2] - world[:, :2].mean(axis=0)) < 2:
        raise ValueError('Ground points must span the plane z=0.')
    # Hold out points BEFORE any geometric fitting. Matching confidence is not accuracy.
    indices = np.random.default_rng(42).permutation(len(world))
    validation, training = indices[:max(5, len(world)//4)], indices[max(5, len(world)//4):]
    cv2.setRNGSeed(42)
    ok, r, t, inliers = cv2.solvePnPRansac(world[training], pixels[training], K, dist,
                                         iterationsCount=3000, reprojectionError=max_error * (3 if refine_focal else 1),
                                         confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 10:
        raise ValueError('No reliable ground-plane pose found.')
    fit = training[inliers.ravel()]
    r, t = cv2.solvePnPRefineLM(world[fit], pixels[fit], K, dist, r, t)
    focal_scale = 1.
    if refine_focal:
        from scipy.optimize import least_squares
        initial_K = K.copy()

        def residual(parameters):
            trial_K = initial_K.copy()
            trial_K[0, 0] *= np.exp(parameters[6])
            trial_K[1, 1] *= np.exp(parameters[6])
            projection = cv2.projectPoints(world[fit], parameters[:3], parameters[3:6], trial_K, dist)[0].reshape(-1, 2)
            # A modest prior discourages focal drift on nearly fronto-parallel planes.
            return np.r_[(projection - pixels[fit]).ravel(), parameters[6] / .1]

        initial = np.r_[r.ravel(), t.ravel(), 0.]
        bound = np.log(1.2)
        result = least_squares(residual, initial, loss='soft_l1', f_scale=2., max_nfev=200,
                               bounds=(np.r_[np.full(6, -np.inf), -bound], np.r_[np.full(6, np.inf), bound]))
        if not result.success:
            raise ValueError('Focal/pose refinement did not converge.')
        r, t = result.x[:3].reshape(3, 1), result.x[3:6].reshape(3, 1)
        focal_scale = float(np.exp(result.x[6]))
        K = initial_K.copy()
        K[0, 0] *= focal_scale
        K[1, 1] *= focal_scale
    R = cv2.Rodrigues(r)[0]
    center = (-R.T @ t).ravel()
    predicted = cv2.projectPoints(world, r, t, K, dist)[0].reshape(-1, 2)
    errors = np.linalg.norm(predicted - pixels, axis=1)
    if not np.isfinite(errors).all() or not np.isfinite(center).all():
        raise ValueError('Pose produced non-finite coordinates.')
    fit = training[errors[training] <= max_error]
    if len(fit) < 10:
        raise ValueError('Too few matches agree after refinement.')
    held = errors[validation]
    fraction = float(np.mean(held <= max_error))
    area = coverage(pixels[fit], image_shape)
    front = ((R @ world[fit].T + t)[2] > 0).all()
    report = dict(focal_refinement_scale=focal_scale, training_inliers=len(fit), training_count=len(training),
                  validation_count=len(validation), validation_inlier_fraction=fraction,
                  validation_median_px=float(np.median(held)),
                  validation_p90_px=float(np.percentile(held, 90)),
                  training_rms_px=float(np.sqrt(np.mean(errors[fit]**2))),
                  image_coverage=area, camera_center_enu_m=center.tolist())
    reasons = []
    if len(fit)/len(training) < .5: reasons.append('Too few training inliers')
    if fraction < .7 or np.median(held) > max_error: reasons.append('Held-out matches do not agree')
    if area < min_coverage: reasons.append('Matches cover too little of the image')
    if not front or not 0.5 < center[2] < 200: reasons.append('Implausible camera height or points behind camera')
    report.update(status='failed' if reasons else 'success', reasons=reasons)
    return CameraModel(K, R, t, dist), report, predicted, errors


def select_consistent_model(accepted, require_multiple=False):
    """Reject changing video poses instead of silently selecting one view."""
    if not accepted or (require_multiple and len(accepted) < 2):
        return None
    best, quality = min(accepted, key=lambda pair: pair[1]['validation_median_px'])
    best_center = -best.rot_matrix.T @ best.tvec
    for model, _ in accepted:
        angle = np.linalg.norm(cv2.Rodrigues(model.rot_matrix @ best.rot_matrix.T)[0])
        delta = np.linalg.norm(-model.rot_matrix.T @ model.tvec - best_center)
        focal_change = np.max(np.abs(np.diag(model.camera_matrix)[:2] / np.diag(best.camera_matrix)[:2] - 1))
        if angle > np.deg2rad(3) or delta > max(1., .1 * abs(best_center[2, 0])) or focal_change > .05:
            return None
    return best, quality


def read_frames(args):
    if args.image:
        im = cv2.imread(str(args.image))
        if im is None: raise ValueError('Cannot read camera image.')
        return [(0, im)]
    cap = cv2.VideoCapture(str(args.video))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0: raise ValueError('Video must be a seekable file with known frame count.')
        frames = []
        for index in np.unique(np.linspace(0, count-1, args.frames, dtype=int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok: frames.append((int(index), frame))
        if not frames: raise ValueError('No readable video frames.')
        return frames
    finally:
        cap.release()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', type=Path)
    source.add_argument('--video', type=Path)
    p.add_argument('--reference', type=Path)
    p.add_argument('--georef', type=Path, help='JSON: projected CRS, pixel_to_world affine, origin')
    p.add_argument('--location', nargs=2, type=float, metavar=('LAT', 'LON'), help='Fetch public aerial reference in Flanders')
    p.add_argument('--reference-width', type=float, default=100., help='Aerial crop width in metres')
    p.add_argument('--intrinsics', type=Path, help='JSON: camera_matrix and optional OpenCV dist_coeffs; otherwise AnyCalib')
    p.add_argument('--matcher', choices=['roma', 'tiny-roma', 'sift'], default='roma')
    p.add_argument('--cache-dir', type=Path, default=Path('.cache'), help='Local model-weight cache')
    p.add_argument('--frames', type=int, default=5)
    p.add_argument('--max-error', type=float, default=4.)
    p.add_argument('--min-coverage', type=float, default=.05)
    p.add_argument('--output-dir', type=Path, default=Path('results/auto'))
    args = p.parse_args()
    os.environ.setdefault('TORCH_HOME', str(args.cache_dir.resolve() / 'torch'))
    os.environ.setdefault('HF_HOME', str(args.cache_dir.resolve() / 'huggingface'))
    if args.frames < 1 or args.max_error <= 0 or not 0 < args.min_coverage <= 1:
        p.error('Invalid sampling or quality thresholds.')
    if bool(args.reference) != bool(args.georef) or bool(args.location) == bool(args.reference):
        p.error('Supply either --location LAT LON or both --reference and --georef.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir/'camera_model.yml'
    if model_path.exists():
        p.error('Output already contains camera_model.yml; use a fresh output directory.')
    report = dict(status='failed', frames=[], coordinate_system='local ENU metres; ground z=0',
                  image=str(args.image) if args.image else None, video=str(args.video) if args.video else None,
                  matcher=args.matcher, thresholds=dict(max_error_px=args.max_error, min_image_coverage=args.min_coverage),
                  limitations=['Planar visible ground only', 'Match agreement is not surveyed accuracy'])
    try:
        if args.location:
            from reference_imagery import fetch_flanders
            args.reference, args.georef = fetch_flanders(*args.location, args.output_dir, width_m=args.reference_width)
        metadata = json.loads(args.georef.read_text())
        reference_coordinates(np.zeros((1, 2)), metadata)
        reference = cv2.imread(str(args.reference))
        if reference is None: raise ValueError('Cannot read reference image.')
        report['reference'] = metadata
        accepted = []
        for index, image in read_frames(args):
            try:
                if args.intrinsics:
                    intr = json.loads(args.intrinsics.read_text())
                    if 'image_size' in intr and intr['image_size'] != [image.shape[1], image.shape[0]]:
                        raise ValueError('Intrinsics image_size does not match the input resolution.')
                    K = np.asarray(intr['camera_matrix'], float)
                    dist = np.asarray(intr.get('dist_coeffs', [0, 0, 0, 0, 0]), float)
                else:
                    from calibrate_camera import predict_camera_intrinsics, build_camera_matrix
                    focal, center = predict_camera_intrinsics(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    K, dist = build_camera_matrix(focal, center), np.zeros(5)
                if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]) or not np.isfinite(dist).all():
                    raise ValueError('Invalid intrinsics.')
                (args.output_dir/f'intrinsics_{index}.json').write_text(json.dumps(dict(
                    camera_matrix=K.tolist(), dist_coeffs=dist.tolist(), image_size=[image.shape[1], image.shape[0]]), indent=2)+'\n')
                a, b, confidence = match_images(image, reference, args.matcher)
                valid = (np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1) & np.isfinite(confidence)
                         & (a >= 0).all(axis=1) & (b >= 0).all(axis=1)
                         & (a[:, 0] < image.shape[1]) & (a[:, 1] < image.shape[0])
                         & (b[:, 0] < reference.shape[1]) & (b[:, 1] < reference.shape[0])
                         & (confidence >= .2))
                a, b, confidence = a[valid], b[valid], confidence[valid]
                # Deduplicate by camera pixel, preventing dense clusters from dominating.
                _, keep = np.unique(np.floor(a / 8).astype(int), axis=0, return_index=True)
                a, b, confidence = a[keep], b[keep], confidence[keep]
                world = reference_coordinates(b, metadata)
                np.savetxt(args.output_dir/f'matches_{index}.csv', np.column_stack((a, b, world, confidence)),
                           delimiter=',', header='image_x,image_y,reference_x,reference_y,east,north,up,confidence', comments='')
                model, quality, predicted, errors = solve(world, a, K, dist, image.shape, args.max_error, args.min_coverage, refine_focal=not bool(args.intrinsics))
                quality.update(frame=index, camera_matrix=model.camera_matrix.tolist(), dist_coeffs=dist.tolist(), candidate_matches=len(a))
                report['frames'].append(quality)
                np.savetxt(args.output_dir/f'matches_{index}.csv', np.column_stack((a, b, world, confidence, errors)),
                           delimiter=',', header='image_x,image_y,reference_x,reference_y,east,north,up,confidence,error_px', comments='')
                overlay = image.copy()
                for actual, projected, error in zip(a, predicted, errors):
                    if np.isfinite(projected).all() and np.max(np.abs(projected)) < 1e6:
                        cv2.line(overlay, tuple(np.rint(actual).astype(int)), tuple(np.rint(projected).astype(int)),
                                 (0, 200, 0) if error <= args.max_error else (0, 0, 255), 1)
                cv2.imwrite(str(args.output_dir/f'overlay_{index}.png'), overlay)
                if quality['status'] == 'success':
                    accepted.append((model, quality))
            except (ValueError, RuntimeError, ImportError, cv2.error) as exc:
                report['frames'].append(dict(frame=index, status='failed', reason=str(exc)))
        if accepted:
            selection = select_consistent_model(accepted, require_multiple=bool(args.video))
            if selection is None:
                report.update(status='partial', reason='Insufficient consistent video poses; no model exported.')
            else:
                best, quality = selection
                best.save_to_yml(str(model_path))
                report.update(status='success', selected_frame=quality['frame'])
        else:
            report['reason'] = 'No frame passed calibration quality checks.'
    except (ValueError, KeyError, OSError, RuntimeError, ImportError, cv2.error) as exc:
        report['reason'] = str(exc)
    (args.output_dir/'quality_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': report['status'], 'report': str(args.output_dir/'quality_report.json')}))
    return 0 if report['status'] == 'success' else 2


if __name__ == '__main__':
    raise SystemExit(main())
