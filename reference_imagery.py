"""Fetch a georeferenced aerial crop from Digitaal Vlaanderen's public WMS."""
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import cv2
import numpy as np
from pyproj import Transformer

ENDPOINT = 'https://geo.api.vlaanderen.be/OMWRGBMRVL/wms'


def fetch_flanders(latitude, longitude, output_dir, width_m=100., pixels=1000):
    if not (50.65 <= latitude <= 51.6 and 2.45 <= longitude <= 5.95):
        raise ValueError('This imagery provider covers Flanders only; supply a local reference elsewhere.')
    if not 20 <= width_m <= 1000 or not 128 <= pixels <= 2048:
        raise ValueError('Reference width must be 20–1000 m and image size 128–2048 pixels.')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    east, north = Transformer.from_crs('EPSG:4326', 'EPSG:31370', always_xy=True).transform(longitude, latitude)
    half = width_m / 2
    bbox = [east-half, north-half, east+half, north+half]
    url = ENDPOINT + '?' + urlencode(dict(SERVICE='WMS', VERSION='1.1.1', REQUEST='GetMap',
            LAYERS='Ortho', STYLES='', SRS='EPSG:31370', BBOX=','.join(map(str, bbox)),
            WIDTH=pixels, HEIGHT=pixels, FORMAT='image/png', TRANSPARENT='FALSE'))
    with urlopen(url, timeout=60) as response:
        data = response.read(20_000_001)
    if len(data) > 20_000_000:
        raise ValueError('Imagery response exceeds size limit.')
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (pixels, pixels) or image.std() < 2:
        raise ValueError('Imagery service returned an empty or invalid crop.')
    step = width_m / pixels
    metadata = dict(crs='EPSG:31370', origin=[east, north],
                    pixel_to_world=[[step, 0, bbox[0]+step/2], [0, -step, bbox[3]-step/2]],
                    source=url, attribution='agentschap Digitaal Vlaanderen',
                    note='Most recent winter orthophoto; pixel transform refers to pixel centres.')
    image_path, georef_path = output/'reference.png', output/'reference.json'
    image_path.write_bytes(data)
    georef_path.write_text(json.dumps(metadata, indent=2)+'\n')
    return image_path, georef_path
