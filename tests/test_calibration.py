import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from auto_calibrate import reference_coordinates, solve, match_images
from camera_model import CameraModel
from calibrate_camera import convert_latlon_F


def scene():
    rng = np.random.default_rng(9)
    world = np.column_stack((rng.uniform(-7, 7, 150), rng.uniform(-5, 5, 150), np.zeros(150)))
    K = np.array([[800., 0, 640], [0, 800, 360], [0, 0, 1]])
    R = np.diag([1., -1., -1.])
    t = np.array([[0.], [0.], [15.]])
    d = np.array([.04, -.01, 0., 0., 0.])
    pixels = cv2.projectPoints(world, cv2.Rodrigues(R)[0], t, K, d)[0].reshape(-1, 2)
    return world, pixels, K, R, t, d


def test_robust_pose_with_outliers_and_distortion():
    world, pixels, K, R, t, d = scene()
    noisy = pixels + np.random.default_rng(2).normal(0, .15, pixels.shape)
    noisy[:15] += 80
    model, report, _, _ = solve(world, noisy, K, d, (720, 1280))
    assert report['status'] == 'success'
    np.testing.assert_allclose(model.tvec, t, atol=.03)
    np.testing.assert_allclose(model.rot_matrix, R, atol=.003)
    np.testing.assert_allclose(model.project_to_ground(pixels), world[:, :2], atol=.002)


def test_bad_correspondences_rejected():
    world, pixels, K, _, _, d = scene()
    with pytest.raises(ValueError):
        solve(world[:10], pixels[:10], K, d, (720, 1280))
    try:
        _, report, _, _ = solve(world, pixels[::-1].copy(), K, d, (720, 1280))
        assert report['status'] == 'failed'
    except ValueError:
        pass


def test_coverage_gate():
    world, pixels, K, _, _, d = scene()
    _, report, _, _ = solve(world, pixels, K, d, (720, 1280), min_coverage=.9)
    assert report['status'] == 'failed'
    assert 'Matches cover too little of the image' in report['reasons']


def test_georeference_precision_and_units():
    metadata = dict(crs='EPSG:32631', pixel_to_world=[[.1, 0, 600000], [0, -.1, 5600000]], origin=[600000, 5600000])
    np.testing.assert_allclose(reference_coordinates(np.array([[1, 2]]), metadata), [[.1, -.2, 0]])
    metadata['crs'] = 'EPSG:4326'
    with pytest.raises(ValueError): reference_coordinates(np.zeros((1, 2)), metadata)
    origin = np.array([[50.926946, 5.336048]])
    assert convert_latlon_F(origin, origin + [[1e-7, 0]])[0, 0] == pytest.approx(.0111318845, rel=1e-6)


def test_homography_roundtrip(tmp_path):
    H = np.array([[2., .1, 10], [0, 3, 20], [.001, 0, 1]])
    model = CameraModel(np.eye(3), np.eye(3), np.zeros((3, 1)), np.zeros(5), H)
    path = tmp_path/'camera.yml'
    model.save_to_yml(str(path))
    loaded = CameraModel.load_from_yml(str(path))
    points = np.array([[1., 2.], [3., 4.]])
    pixels = np.array([loaded.project_point(p) for p in points])
    np.testing.assert_allclose(loaded.project_to_ground(pixels), points)


def test_failure_report_without_optional_models(tmp_path):
    ref = tmp_path/'ref.png'
    cv2.imwrite(str(ref), np.zeros((100, 100, 3), np.uint8))
    geo = tmp_path/'geo.json'
    geo.write_text(json.dumps(dict(crs='EPSG:4326', pixel_to_world=[[1,0,0],[0,1,0]], origin=[0,0])))
    result = subprocess.run([sys.executable, 'auto_calibrate.py', '--image', str(ref), '--reference', str(ref), '--georef', str(geo), '--output-dir', str(tmp_path/'out')], capture_output=True, text=True)
    assert result.returncode == 2, result.stderr
    report = json.loads((tmp_path/'out/quality_report.json').read_text())
    assert report['status'] == 'failed'
    assert not (tmp_path/'out/camera_model.yml').exists()


def test_sift_and_cli_end_to_end(tmp_path):
    rng = np.random.default_rng(8)
    reference = np.zeros((720, 960, 3), np.uint8)
    for _ in range(2000):
        point = tuple(rng.integers([10, 10], [950, 710]).tolist())
        cv2.circle(reference, point, int(rng.integers(2, 8)), tuple(rng.integers(50, 255, 3).tolist()), -1)
    # Camera at 15 m, looking down: known 0.02 m reference pixels.
    K = np.array([[700., 0, 480], [0, 700, 360], [0, 0, 1.]])
    H = np.array([[700*.02/15, 0, 480-700*.02*480/15], [0, 700*.02/15, 360-700*.02*360/15], [0, 0, 1.]])
    camera = cv2.warpPerspective(reference, H, (960, 720))
    a, b, _ = match_images(camera, reference, 'sift')
    assert len(a) > 50
    cv2.imwrite(str(tmp_path/'reference.png'), reference)
    cv2.imwrite(str(tmp_path/'camera.png'), camera)
    (tmp_path/'geo.json').write_text(json.dumps(dict(crs='EPSG:32631', pixel_to_world=[[.02,0,600000-9.6],[0,-.02,5600000+7.2]], origin=[600000,5600000])))
    (tmp_path/'intr.json').write_text(json.dumps(dict(camera_matrix=K.tolist())))
    command = [sys.executable, 'auto_calibrate.py', '--image', str(tmp_path/'camera.png'), '--reference', str(tmp_path/'reference.png'), '--georef', str(tmp_path/'geo.json'), '--intrinsics', str(tmp_path/'intr.json'), '--matcher', 'sift', '--output-dir', str(tmp_path/'out')]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr + (tmp_path/'out/quality_report.json').read_text()
    assert (tmp_path/'out/camera_model.yml').exists()
    model = CameraModel.load_from_yml(str(tmp_path/'out/camera_model.yml'))
    assert (-model.rot_matrix.T @ model.tvec)[2,0] == pytest.approx(15, abs=.1)


def test_reference_fetch_pixel_centers(tmp_path, monkeypatch):
    import io
    import reference_imagery
    rng = np.random.default_rng(3)
    image = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    encoded = cv2.imencode('.png', image)[1].tobytes()
    monkeypatch.setattr(reference_imagery, 'urlopen', lambda *args, **kwargs: io.BytesIO(encoded))
    _, path = reference_imagery.fetch_flanders(50.926946, 5.336048, tmp_path, width_m=32, pixels=128)
    metadata = json.loads(path.read_text())
    result = reference_coordinates(np.array([[63.5, 63.5], [0., 0.]]), metadata)
    np.testing.assert_allclose(result, [[0,0,0], [-15.875,15.875,0]], atol=1e-8)


def test_reference_fetch_rejects_error_response(tmp_path, monkeypatch):
    import io
    import reference_imagery
    monkeypatch.setattr(reference_imagery, 'urlopen', lambda *args, **kwargs: io.BytesIO(b'<ServiceException>Unavailable</ServiceException>'))
    with pytest.raises(ValueError, match='empty or invalid'):
        reference_imagery.fetch_flanders(50.926946, 5.336048, tmp_path, pixels=128)
    assert not (tmp_path/'reference.png').exists()


def test_video_sampling(tmp_path):
    from argparse import Namespace
    from auto_calibrate import read_frames
    path = tmp_path/'clip.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (160, 120))
    assert writer.isOpened()
    for i in range(12): writer.write(np.full((120,160,3), i*10, np.uint8))
    writer.release()
    frames = read_frames(Namespace(image=None, video=path, frames=3))
    assert [i for i, _ in frames] == [0, 5, 11]


def test_video_requires_multiple_consistent_poses():
    from auto_calibrate import select_consistent_model
    _, _, K, R, t, d = scene()
    model = CameraModel(K, R, t, d)
    quality = dict(validation_median_px=1.)
    assert select_consistent_model([(model, quality)], require_multiple=True) is None
    assert select_consistent_model([(model, quality)] * 2, require_multiple=True) is not None
    moved = CameraModel(K, R, t + np.array([[4.], [0.], [0.]]), d)
    assert select_consistent_model([(model, quality), (moved, quality)], require_multiple=True) is None


def test_empty_and_nonfinite_correspondences():
    world, pixels, K, _, _, d = scene()
    with pytest.raises(ValueError): solve(world[:0], pixels[:0], K, d, (720,1280))
    world[0, 0] = np.nan
    with pytest.raises(ValueError): solve(world, pixels, K, d, (720,1280))


def test_video_rejects_zoom_change():
    from auto_calibrate import select_consistent_model
    _, _, K, R, t, d = scene()
    changed = K.copy()
    changed[0, 0] *= 1.2
    changed[1, 1] *= 1.2
    quality = dict(validation_median_px=1.)
    assert select_consistent_model([(CameraModel(K,R,t,d), quality), (CameraModel(changed,R,t,d), quality)], require_multiple=True) is None


def test_bounded_focal_refinement():
    world, _, K, _, _, d = scene()
    # Tilted plane gives focal information that a straight-down image lacks.
    R = cv2.Rodrigues(np.array([2.7, .2, .1]))[0]
    t = -R @ np.array([[0.], [0.], [15.]])
    pixels = cv2.projectPoints(world, cv2.Rodrigues(R)[0], t, K, d)[0].reshape(-1,2)
    guess = K.copy()
    guess[0,0] *= 1.04
    guess[1,1] *= 1.04
    model, report, _, _ = solve(world, pixels, guess, d, (720,1280), refine_focal=True)
    assert report['status'] == 'success'
    assert abs(model.camera_matrix[0,0]-K[0,0]) < abs(guess[0,0]-K[0,0])/3
    assert 1/1.2 <= report['focal_refinement_scale'] <= 1.2
