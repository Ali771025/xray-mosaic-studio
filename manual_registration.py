"""Explicit same-surface correspondences with held-out pixel checks.

Coordinates refer to the exact input rasters, never scene/screen coordinates.
Check points are excluded from fitting. A pass applies only to those points.
"""
import hashlib
import json
from pathlib import Path

import cv2 as cv
import numpy as np
from visual_mosaic import Options, sane_homography, transform, compose


def image_identity(image):
    return {'shape': list(image.shape), 'dtype': str(image.dtype),
            'sha256': hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()}


def session_document(images, labels, points, model='projective', tolerance_px=1.):
    return {'version': 1, 'kind': 'manual_visual_correspondences',
            'images': [dict(label=str(label), **image_identity(im)) for im, label in zip(images, labels)],
            'points': points, 'model': model, 'tolerance_px': float(tolerance_px),
            'coordinate_system': 'input raster pixel centers; reference and moving',
            'limitation': 'User-designated check points; no proof of unknown depth or whole-image accuracy.'}


def load_session(path, images):
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    if doc.get('version') != 1 or doc.get('kind') != 'manual_visual_correspondences':
        raise ValueError('Unsupported correspondence document')
    if len(doc.get('images', [])) != 2 or len(images) != 2:
        raise ValueError('Exactly two input images required')
    for saved, image in zip(doc['images'], images):
        if any(saved.get(k) != v for k, v in image_identity(image).items()):
            raise ValueError('Input image content/order differs from saved session')
    # Validate point schema even before sufficient points exist.
    validate_points(doc.get('points', []), [im.shape for im in images])
    if doc.get('model') not in ('projective', 'affine'):
        raise ValueError('Unknown model')
    tolerance = float(doc.get('tolerance_px', 0))
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Invalid pixel tolerance')
    return doc


def validate_points(points, shapes):
    seen = [set(), set()]
    for row in points:
        if row.get('role') not in ('fit', 'check'):
            raise ValueError('Each pair must have fit/check role')
        for i, key in enumerate(('reference', 'moving')):
            xy = np.asarray(row[key], float)
            if xy.shape != (2,) or not np.isfinite(xy).all():
                raise ValueError('Invalid point coordinates')
            h, w = shapes[i][:2]
            if not (0 <= xy[0] <= w-1 and 0 <= xy[1] <= h-1):
                raise ValueError('Point outside input image')
            # Duplicated coordinates cannot serve as independent checks.
            if any(np.linalg.norm(xy-np.asarray(old)) < .5 for old in seen[i]):
                raise ValueError('Repeated point: fit and check pairs must be distinct')
            seen[i].add(tuple(xy))


def _spread(points):
    if len(points) < 3 or np.linalg.matrix_rank(points-points.mean(axis=0)) < 2:
        raise ValueError('Points are collinear; distribute points across the shared surface')


def fit_correspondences(images, points, model='projective', tolerance_px=1.):
    if len(images) != 2 or model not in ('projective', 'affine'):
        raise ValueError('Choose two images and a valid model')
    tolerance_px = float(tolerance_px)
    if not np.isfinite(tolerance_px) or tolerance_px <= 0:
        raise ValueError('Pixel tolerance must be positive')
    validate_points(points, [im.shape for im in images])
    for row in points:
        for image, key in zip(images, ('reference', 'moving')):
            if image.ndim == 3 and image.shape[2] == 4:
                x, y = np.rint(row[key]).astype(int)
                if image[y, x, 3] == 0:
                    raise ValueError('Point lies on a transparent gap')
    fitting = [p for p in points if p['role'] == 'fit']
    checking = [p for p in points if p['role'] == 'check']
    minimum = 4 if model == 'projective' else 3
    if len(fitting) < minimum or len(checking) < 3:
        raise ValueError(f'Need at least {minimum} fit pairs and 3 separate check pairs')
    dst = np.array([p['reference'] for p in fitting], float)
    src = np.array([p['moving'] for p in fitting], float)
    for values in (src, dst, np.array([p['reference'] for p in checking]),
                   np.array([p['moving'] for p in checking])):
        _spread(values)
    if model == 'projective':
        H, _ = cv.findHomography(src, dst, 0)  # all explicitly chosen fit points
    else:
        design = np.column_stack((src, np.ones(len(src))))
        coefficients, _, rank, _ = np.linalg.lstsq(design, dst, rcond=None)
        if rank != 3:
            raise ValueError('Degenerate affine fit')
        H = np.vstack((coefficients.T, [0., 0., 1.]))
    if H is None or not sane_homography(H, images[1].shape, Options(model=model)):
        raise ValueError('Unstable transform or unsafe extrapolation')
    inv = np.linalg.inv(H)
    rows = []
    for p in points:
        a, b = np.array(p['reference']), np.array(p['moving'])
        forward = float(np.linalg.norm(transform([b], H)[0]-a))
        reverse = float(np.linalg.norm(transform([a], inv)[0]-b))
        rows.append(dict(p, forward_px=forward, reverse_px=reverse,
                         symmetric_max_px=max(forward, reverse)))
    def stats(role):
        values = [p['symmetric_max_px'] for p in rows if p['role'] == role]
        return {'count': len(values), 'median_px': float(np.median(values)),
                'max_px': float(max(values))}
    fit, check = stats('fit'), stats('check')
    passed = max(fit['max_px'], check['max_px']) <= tolerance_px
    return {'H_moving_to_reference': H.tolist(), 'model': model, 'points': rows,
            'fit': fit, 'check': check, 'tolerance_px': tolerance_px,
            'accepted': passed, 'status': 'CHECK_POINTS_PASS' if passed else 'CHECK_POINTS_FAIL',
            'metric_accuracy': 'UNVERIFIED',
            'limitation': 'Checks exclude fitting points; passing checks does not establish whole-image or multi-depth alignment.'}


def render_pair(images, report):
    if not report['accepted']:
        raise ValueError('Independent point checks failed; transform cannot be applied')
    result = compose(images, [0, 1], {0: np.eye(3), 1: np.array(report['H_moving_to_reference'])},
                     Options(model=report['model'], blend='seam'))
    if not np.any(result['coverage'] > 1):
        raise ValueError('No raster overlap')
    result['qa']['manual_registration'] = report
    return result


def export_result(destination, images, labels, points, report, result):
    """Write a new reviewable output directory; refuse replacing any prior result."""
    import tempfile
    import shutil
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Choose a new output directory')
    # Recompute from the current inputs and controls rather than trust stored status.
    verified = fit_correspondences(images, points, report['model'], report['tolerance_px'])
    if not verified['accepted']:
        raise ValueError('Independent point checks failed')
    if not np.allclose(verified['H_moving_to_reference'], report['H_moving_to_reference']):
        raise ValueError('Transform differs from current points')
    doc = session_document(images, labels, points, report['model'], report['tolerance_px'])
    doc['evaluation'] = verified
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='manual-export-') as directory:
        local = Path(directory)
        for name, image in [('mosaic.png', result['image']), ('coverage.png', result['coverage'])]:
            if not cv.imwrite(str(local/name), image, [cv.IMWRITE_PNG_COMPRESSION, 1]):
                raise OSError('Cannot encode '+name)
        np.savez_compressed(local/'source.npz', source=result['source'])
        (local/'points.json').write_text(json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        qa = dict(result['qa'], transforms=result['transforms'], origin_anchor_px=result['origin_anchor_px'])
        (local/'report.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        destination.mkdir()
        for item in local.iterdir():
            shutil.copyfile(item, destination/item.name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('moving', type=Path)
    parser.add_argument('--points', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    images = [cv.imread(str(p), cv.IMREAD_UNCHANGED) for p in (args.reference, args.moving)]
    if any(im is None for im in images):
        parser.error('Cannot decode input images')
    doc = load_session(args.points, images)
    report = fit_correspondences(images, doc['points'], doc['model'], doc['tolerance_px'])
    result = render_pair(images, report)
    export_result(args.output, images, [str(args.reference), str(args.moving)], doc['points'], report, result)
    print(json.dumps(report, indent=2))
