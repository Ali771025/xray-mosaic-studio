"""Original-file ingestion and bounded tile composition; no feature-matching dependency."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import shutil
import tempfile
from datetime import datetime
import numpy as np
import cv2 as cv
from .geometry import Calibration, Plane, GeometryError, validate_pose, projection_jacobian

LIMITATION = ('Metric coordinates apply only to the specified object reference plane. '
              'Other depths retain parallax. Sparse transmission projections do not identify a 3D volume.')


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_original(path, max_pixels):
    a = cv.imread(str(path), cv.IMREAD_UNCHANGED)
    if a is None:
        raise GeometryError(f'cannot decode original {path}', 'FAILED')
    if a.shape[0]*a.shape[1] > max_pixels:
        raise GeometryError('input exceeds configured frame pixel budget', 'FAILED')
    if a.dtype not in (np.uint8, np.uint16):
        raise GeometryError('supported originals: uint8/uint16 grayscale, BGR or BGRA', 'FAILED')
    if a.ndim == 3:
        # Preserve integer originals on disk. Defined luminance conversion in processing only.
        valid = a[..., 3] > 0 if a.shape[2] == 4 else np.ones(a.shape[:2], bool)
        if a.shape[2] not in (3, 4):
            raise GeometryError('unsupported channels')
        a = cv.cvtColor(a[..., :3], cv.COLOR_BGR2GRAY)
    else:
        valid = np.ones(a.shape, bool)
    return a, valid


def normalize_radiometry(image, valid, dark=None, flat=None, negative_log=False):
    image = image.astype(np.float32)
    valid = valid.copy() & np.isfinite(image)
    if (dark is None) != (flat is None):
        raise GeometryError('dark and flat must be supplied together')
    if flat is not None:
        if image.shape != flat.shape or image.shape != dark.shape:
            raise GeometryError('dark/flat dimensions mismatch')
        den = flat.astype(float)-dark
        num = image-dark
        valid &= np.isfinite(den) & (den > 0) & (num >= 0)
        image = np.divide(num, den, out=np.zeros_like(image), where=valid)
    elif negative_log:
        raise GeometryError('negative log requires measured dark/flat incident field')
    if negative_log:
        valid &= image > 0
        image = -np.log(np.maximum(image, np.finfo(np.float32).tiny))
    image[~valid] = 0
    return image.astype(np.float32), valid


@dataclass
class Frame:
    id: str
    path: Path
    pose: np.ndarray
    stage_sigma_mm: float | None
    timestamp: str
    dark: Path | None = None
    flat: Path | None = None


class Batch:
    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path).resolve()
        manifest_bytes = self.manifest_path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self.config = json.loads(manifest_bytes.decode('utf-8'))
        c = self.config
        try:
            if c['version'] != 1:
                raise GeometryError('unsupported batch version')
            self.calibration = Calibration(**c['calibration'])
            if c['scanner_id'] != self.calibration.scanner_id:
                raise GeometryError('scanner/profile mismatch')
            self.planes = [Plane(**p) for p in c['planes']]
            if not self.planes or len({p.id for p in self.planes}) != len(self.planes):
                raise GeometryError('nonempty uniquely named planes required')
            self.spacing = float(c['output_spacing_mm'])
            self.tile_size = int(c.get('tile_size', 256))
            self.max_frame_pixels = int(c.get('max_frame_pixels', 64_000_000))
            self.max_output_pixels = int(c.get('max_output_pixels', 200_000_000))
            if not math.isfinite(self.spacing) or self.spacing <= 0 or not 16 <= self.tile_size <= 1024:
                raise GeometryError('positive grid spacing and tile_size 16..1024 required')
            if self.max_frame_pixels <= 0 or self.max_output_pixels <= 0:
                raise GeometryError('pixel budgets must be positive')
            self.frames = []
            for f in c['frames']:
                if f['scanner_id'] != c['scanner_id']:
                    raise GeometryError('frame scanner mismatch')
                if f.get('calibration_id') != self.calibration.id:
                    raise GeometryError('frame calibration ID missing or mismatched')
                sigma = f.get('stage_sigma_mm')
                if sigma is not None and (not np.isfinite(sigma) or sigma < 0):
                    raise GeometryError('invalid stage uncertainty')
                if not f['timestamp'] or not f['id']:
                    raise GeometryError('frame ID and acquisition timestamp required')
                acquired = datetime.fromisoformat(f['timestamp'])
                if acquired.tzinfo is None or not self.calibration.created <= acquired <= self.calibration.expires:
                    raise GeometryError('acquisition timestamp outside calibration validity')
                if bool(f.get('dark')) != bool(f.get('flat')):
                    raise GeometryError('supply both dark and flat per frame')
                self.frames.append(Frame(f['id'], self.resolve(f['path']), validate_pose(f['pose']),
                                         sigma, f['timestamp'],
                                         self.resolve(f['dark']) if f.get('dark') else None,
                                         self.resolve(f['flat']) if f.get('flat') else None))
            if not self.frames or len({f.id for f in self.frames}) != len(self.frames):
                raise GeometryError('nonempty unique frame IDs required')
            if len({bool(f.flat) for f in self.frames}) != 1:
                raise GeometryError('cannot mix detector counts and normalized transmission frames')
            if len(self.frames) > 65535:
                raise GeometryError('coverage counter supports up to 65535 frames')
        except (KeyError, TypeError, ValueError) as e:
            if isinstance(e, GeometryError):
                raise
            raise GeometryError(f'invalid batch schema: {e}') from e

    def resolve(self, path):
        return (self.manifest_path.parent/Path(path)).resolve()

    def inspect(self):
        issues = self.calibration.metric_issues()
        limit = self.config.get('max_sigma_mm')
        if limit is None or not np.isfinite(limit) or limit <= 0:
            issues.append('customer maximum allowed plane/stage sigma_mm not specified')
        elif not issues:
            if any(f.stage_sigma_mm is not None and math.hypot(self.calibration.validation['sigma_plane_mm'], f.stage_sigma_mm) > limit for f in self.frames):
                issues.append('supplied plane/stage uncertainty exceeds customer sigma threshold')
        if any(f.stage_sigma_mm is None for f in self.frames):
            issues.append('stage uncertainty unknown')
        return {'status': ('UNDERCONSTRAINED' if issues else ('DEPTH_SPECIFIC_METRIC' if len(self.planes) > 1 else 'REFERENCE_PLANE_METRIC')),
                'issues': issues, 'limitation': LIMITATION,
                'calibration_id': self.calibration.id,
                'planes': [p.id for p in self.planes],
                'frames': [{'id': f.id, 'path': str(f.path), 'pose': f.pose.tolist(),
                            'stage_sigma_mm': f.stage_sigma_mm} for f in self.frames]}

    def run(self, destination, progress=None, cancelled=None):
        """Transaction: publish new directory only on success; never overwrite an export."""
        if file_hash(self.manifest_path) != self.manifest_sha256:
            raise GeometryError('batch configuration changed since loading; reload it', 'FAILED')
        destination = Path(destination).resolve()
        if destination.exists():
            raise GeometryError('choose a new output directory; exports are immutable', 'FAILED')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if any(destination == f.path or destination in f.path.parents for f in self.frames):
            raise GeometryError('output cannot contain input files')
        work = Path(tempfile.mkdtemp(prefix='.xray-', dir=destination.parent))
        try:
            report = self.inspect()
            report.update({'schema_version': 1, 'merge_policy': 'first_valid_frame_authoritative',
                           'registration': 'IMAGE_ALIGNMENT_UNVERIFIED',
                           'configuration_sha256': self.manifest_sha256,
                           'planes': [], 'frames': []})
            metadata = []
            for f in self.frames:
                image, valid = read_original(f.path, self.max_frame_pixels)
                if list(image.shape) != self.calibration.image_shape:
                    raise GeometryError(f'frame {f.id} dimensions disagree with calibration')
                m = {'id': f.id, 'path': str(f.path), 'sha256': file_hash(f.path),
                     'dtype': str(image.dtype), 'pose': f.pose.tolist(), 'timestamp': f.timestamp,
                     'stage_sigma_mm': f.stage_sigma_mm,
                     'zero_fraction': float(np.mean(image == 0)),
                     'saturated_fraction': float(np.mean(image == np.iinfo(image.dtype).max)),
                     'contrast_std': float(np.std(image)),
                     'dark_sha256': file_hash(f.dark) if f.dark else None,
                     'flat_sha256': file_hash(f.flat) if f.flat else None}
                metadata.append(m)
            report['frames'] = metadata
            # Release scan buffers before tile processing.
            del image, valid
            for index, plane in enumerate(self.planes):
                if cancelled and cancelled():
                    raise GeometryError('cancelled', 'FAILED')
                directory = work/f'plane_{index:03d}'
                directory.mkdir()
                report['planes'].append(self.compose(plane, directory, report['status'], progress, cancelled))
            report['calibration'] = self.calibration.to_dict()
            report['configuration'] = self.config
            report['multi_depth'] = ('Separate plane outputs only; coverage/overlap RMS are plane-specific. '
                                     'Intensity disagreement is not a unique depth or parallax estimate.')
            (work/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
            if any(file_hash(f.path) != m['sha256'] or
                   (f.dark and file_hash(f.dark) != m['dark_sha256']) or
                   (f.flat and file_hash(f.flat) != m['flat_sha256'])
                   for f, m in zip(self.frames, metadata)):
                raise GeometryError('input/calibration fields changed during processing', 'FAILED')
            work.rename(destination)
            return report
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise

    def compose(self, plane, directory, status, progress, cancelled):
        cal = self.calibration
        h, w = cal.image_shape
        corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        polygons = []
        for f in self.frames:
            xy, valid = cal.backproject(corners, plane, f.pose)
            if not valid.all():
                raise GeometryError('plane outside source-detector segment or crosses a projective horizon')
            # Bilinear interpolation only: refuse undersampled exports rather than silently alias.
            jac = projection_jacobian(cal, xy, plane, f.pose)
            if np.max(np.linalg.svd(jac, compute_uv=False))*self.spacing > 1.01:
                raise GeometryError('grid too coarse for bilinear remap; reduce output_spacing_mm (antialias filter unavailable)')
            polygons.append(xy)
        xy = np.concatenate(polygons)
        origin = np.floor(xy.min(axis=0)/self.spacing)*self.spacing
        extent = (xy.max(axis=0)-origin)/self.spacing
        if not np.isfinite(extent).all() or (extent > self.max_output_pixels).any():
            raise GeometryError('physical extent exceeds configured canvas budget', 'FAILED')
        shape = np.ceil(extent).astype(np.int64)[::-1]+1
        if (shape <= 0).any() or int(shape[0])*int(shape[1]) > self.max_output_pixels:
            raise GeometryError('mosaic exceeds configured pixel budget', 'FAILED')
        shape = tuple(map(int, shape))
        def output(name, dtype, fill):
            a = np.lib.format.open_memmap(directory/f'{name}.npy', mode='w+', dtype=dtype, shape=shape)
            a[:] = fill
            return a
        values = output('intensity', 'float32', np.nan)
        coverage = output('coverage', 'uint16', 0)
        source = output('source', 'int32', -1)
        uncertainty = output('sigma_mm', 'float32', np.nan)
        metric = output('metric_valid', 'uint8', 0)
        seam_sse, seam_count, valid_count, duplicate_count = 0., 0, 0, 0
        for fi, frame in enumerate(self.frames):
            raw, valid = read_original(frame.path, self.max_frame_pixels)
            dark = read_original(frame.dark, self.max_frame_pixels)[0] if frame.dark else None
            flat = read_original(frame.flat, self.max_frame_pixels)[0] if frame.flat else None
            image, valid = normalize_radiometry(raw, valid, dark, flat, self.config.get('negative_log', False))
            del raw, dark, flat
            # Conservative support: all bilinear neighbours must be valid.
            for y in range(0, shape[0], self.tile_size):
                if cancelled and cancelled():
                    raise GeometryError('cancelled', 'FAILED')
                for x in range(0, shape[1], self.tile_size):
                    ys, xs = slice(y, min(y+self.tile_size, shape[0])), slice(x, min(x+self.tile_size, shape[1]))
                    gx, gy = np.meshgrid(np.arange(x, xs.stop), np.arange(y, ys.stop))
                    object_xy = origin + np.stack((gx, gy), axis=-1)*self.spacing
                    uv, rays = cal.plane_project(object_xy, plane, frame.pose)
                    inside = rays & (uv[..., 0] >= 0) & (uv[..., 0] <= w-1) & (uv[..., 1] >= 0) & (uv[..., 1] <= h-1)
                    uv[~inside] = -2
                    mx, my = uv[..., 0].astype('float32'), uv[..., 1].astype('float32')
                    sample = cv.remap(image, mx, my, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
                    support = cv.remap(valid.astype('float32'), mx, my, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
                    good = inside & (support >= 1-1e-6)
                    c, out = coverage[ys, xs], values[ys, xs]
                    overlap = good & (c > 0)
                    diff = sample[overlap].astype(float)-out[overlap]
                    seam_sse += float(diff @ diff)
                    seam_count += int(overlap.sum())
                    first = good & (c == 0)
                    duplicate_count += int((good & (c == 1)).sum())
                    valid_count += int(first.sum())
                    out[first] = sample[first]
                    source[ys, xs][first] = fi
                    c[good] += 1
                    if status != 'UNDERCONSTRAINED':
                        sigma = math.hypot(cal.validation['sigma_plane_mm'], frame.stage_sigma_mm)
                        uncertainty[ys, xs][first] = sigma
                        metric[ys, xs][first] = 1
                if progress:
                    progress(fi+1, len(self.frames), y, shape[0])
            del image, valid
        for a in (values, coverage, source, uncertainty, metric):
            a.flush()
        total = shape[0]*shape[1]
        info = {'id': plane.id, 'directory': directory.name, 'status': ('DEPTH_SPECIFIC_METRIC' if status != 'UNDERCONSTRAINED' and len(self.planes) > 1 else status),
                'origin_plane_mm': origin.tolist(), 'spacing_mm': self.spacing, 'shape': shape,
                'plane': {'origin_mm': plane.origin_mm, 'u': plane.u, 'v': plane.v},
                'coverage_polygons_mm': [p.tolist() for p in polygons],
                'gap_fraction': 1-valid_count/total, 'overlap_fraction': duplicate_count/total,
                'overlap_intensity_rms': math.sqrt(seam_sse/seam_count) if seam_count else None,
                'overlap_samples': seam_count, 'seam_geometric_error_mm': None,
                'uncertainty_model': 'RSS supplied plane and stage isotropic 1-sigma bounds; not full covariance',
                'data_domain': ('attenuation_negative_log' if self.config.get('negative_log') else
                                ('transmission' if any(f.flat for f in self.frames) else 'detector_counts')),
                'image_alignment': 'unverified; geometry-only', 'limitation': LIMITATION}
        del values, coverage, source, uncertainty, metric
        return info
