"""Compare Lambert-72 control tables without assuming either is ground truth."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pyproj import Transformer


def evaluate(path):
    data = pd.read_csv(path)
    world = data[['lambert_lat', 'lambert_lon']].to_numpy(dtype=float)
    pixels = data[['x', 'y']].to_numpy(dtype=float)
    if len(world) < 6 or not np.isfinite(world).all() or not np.isfinite(pixels).all():
        raise ValueError('Need at least six finite correspondences.')
    origin = world.mean(axis=0)
    world -= origin
    H, _ = cv2.findHomography(world, pixels, 0)
    if H is None: raise ValueError('Degenerate control points.')
    predicted = cv2.perspectiveTransform(world[:, None, :], H)[:, 0]
    errors = np.linalg.norm(predicted - pixels, axis=1)
    held_out = []
    for i in range(len(world)):
        keep = np.arange(len(world)) != i
        h, _ = cv2.findHomography(world[keep], pixels[keep], 0)
        if h is None: raise ValueError('Degenerate leave-one-out fit.')
        held_out.append(float(np.linalg.norm(cv2.perspectiveTransform(world[i:i+1, None, :], h)[0, 0] - pixels[i])))
    return dict(file=str(path), assumed_crs='EPSG:31370',
                columns_interpreted_as={'lambert_lat': 'easting', 'lambert_lon': 'northing'},
                center_lon_lat=Transformer.from_crs(31370, 4326, always_xy=True).transform(*origin),
                point_count=len(world), fit_rms_px=float(np.sqrt(np.mean(errors**2))),
                leave_one_out_rms_px=float(np.sqrt(np.mean(np.square(held_out)))),
                worst_csv_line=int(np.argmax(held_out))+2,
                per_point=[dict(csv_line=i+2, error_px=float(e), held_out_error_px=v) for i, (e,v) in enumerate(zip(errors, held_out))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = [evaluate(path) for path in args.csv]
    report = dict(note='Planar consistency comparison only, not a geographic accuracy certification. Assumes Lambert 72 easting/northing.', results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    for result in results:
        print(f"{result['file']}: fit RMS {result['fit_rms_px']:.2f} px; held-out RMS {result['leave_one_out_rms_px']:.2f} px; worst CSV line {result['worst_csv_line']}")


if __name__ == '__main__':
    main()
