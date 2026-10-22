"""Millimetre world geometry. Pixel coordinates refer to pixel centres."""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import numpy as np


class GeometryError(ValueError):
    def __init__(self, message, code='INSUFFICIENT_GEOMETRY'):
        self.code = code
        super().__init__(f'{code}: {message}')


def array(value, shape, label):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise GeometryError(f'{label} must be finite with shape {shape}')
    return a


def basis(u, v):
    u, v = array(u, (3,), 'axis u'), array(v, (3,), 'axis v')
    if not np.allclose([u @ u, v @ v, u @ v], [1, 1, 0], atol=1e-8):
        raise GeometryError('axes must be orthonormal')
    return u, v


@dataclass
class Plane:
    id: str
    origin_mm: list
    u: list
    v: list

    def __post_init__(self):
        array(self.origin_mm, (3,), 'plane origin')
        basis(self.u, self.v)
        if not self.id:
            raise GeometryError('plane ID required')

    def points(self, xy):
        return np.asarray(self.origin_mm) + np.asarray(xy) @ np.array([self.u, self.v])


@dataclass
class Calibration:
    version: int
    id: str
    scanner_id: str
    created_at: str
    expires_at: str
    units: str
    source_mm: list
    detector_origin_mm: list  # world point at principal point
    detector_u: list
    detector_v: list
    pitch_mm: list
    principal_px: list
    image_shape: list  # height,width
    provenance: dict
    validation: dict
    distortion: str = 'none'

    def __post_init__(self):
        if self.version != 1 or self.units != 'mm' or not self.id or not self.scanner_id:
            raise GeometryError('version=1, units=mm, calibration/scanner IDs required')
        array(self.source_mm, (3,), 'source')
        array(self.detector_origin_mm, (3,), 'detector origin')
        u, v = basis(self.detector_u, self.detector_v)
        if abs((np.array(self.detector_origin_mm)-self.source_mm) @ np.cross(u, v)) < 1e-6:
            raise GeometryError('source lies on detector plane')
        if (array(self.pitch_mm, (2,), 'pitch') <= 0).any():
            raise GeometryError('positive pixel spacing required')
        array(self.principal_px, (2,), 'principal point')
        shape = array(self.image_shape, (2,), 'image shape')
        if (shape < 2).any() or not np.equal(shape, np.floor(shape)).all():
            raise GeometryError('image dimensions must be integers >=2')
        if self.distortion != 'none':
            raise GeometryError('detector distortion requires a calibrated adapter; unsupported profile')
        if not self.provenance:
            raise GeometryError('calibration provenance required')
        try:
            self.created = datetime.fromisoformat(self.created_at)
            self.expires = datetime.fromisoformat(self.expires_at)
            if self.created.tzinfo is None or self.expires.tzinfo is None or self.expires <= self.created:
                raise ValueError()
        except (ValueError, TypeError):
            raise GeometryError('ordered timezone-aware creation/expiry timestamps required')

    def metric_issues(self, now=None):
        now = now or datetime.now(timezone.utc)
        issues = []
        if not self.created <= now <= self.expires:
            issues.append('calibration stale or not yet effective')
        r = self.validation
        for key in ('heldout_rms_px', 'max_rms_px', 'sigma_plane_mm'):
            value = r.get(key)
            if value is None or not np.isfinite(value) or value < 0:
                issues.append(f'missing/invalid validation {key}')
        if r.get('heldout_count', 0) < 3 or r.get('status') != 'validated':
            issues.append('independent held-out marker validation required')
        if not issues and r['heldout_rms_px'] > r['max_rms_px']:
            issues.append('held-out residual exceeds configured acceptance')
        return issues

    def project(self, points):
        p = np.asarray(points, float)
        s, o = np.array(self.source_mm), np.array(self.detector_origin_mm)
        u, v = np.array(self.detector_u), np.array(self.detector_v)
        n = np.cross(u, v)
        rays = p-s
        den = rays @ n
        with np.errstate(divide='ignore', invalid='ignore'):
            t = ((o-s) @ n) / den
            hit = s + rays * t[..., None]
            uv = np.stack(((hit-o) @ u, (hit-o) @ v), axis=-1)/self.pitch_mm+self.principal_px
        valid = np.isfinite(uv).all(axis=-1) & (t > 1 + 1e-10)
        return uv, valid

    def backproject(self, pixels, plane, pose):
        pose = validate_pose(pose)
        uv = np.asarray(pixels, float)
        s = np.array(self.source_mm)
        rays = (np.array(self.detector_origin_mm) +
                ((uv-self.principal_px)*self.pitch_mm) @ np.array([self.detector_u, self.detector_v]))-s
        origin = pose[:3, :3] @ plane.origin_mm + pose[:3, 3]
        axes = pose[:3, :3] @ np.array([plane.u, plane.v]).T
        n = np.cross(axes[:, 0], axes[:, 1])
        den = rays @ n
        with np.errstate(divide='ignore', invalid='ignore'):
            t = ((origin-s) @ n)/den
            q = (s + rays*t[..., None]-origin) @ axes
        valid = np.isfinite(q).all(axis=-1) & (t > 0) & (t < 1)
        return q, valid

    def plane_project(self, xy, plane, pose):
        pose = validate_pose(pose)
        return self.project(plane.points(xy) @ pose[:3, :3].T + pose[:3, 3])

    def to_dict(self):
        return asdict(self)


def validate_pose(pose):
    p = array(pose, (4, 4), 'object-to-world pose')
    if (not np.allclose(p[3], [0, 0, 0, 1]) or
            not np.allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), atol=1e-8) or
            not np.isclose(np.linalg.det(p[:3, :3]), 1)):
        raise GeometryError('stage pose must be a rigid right-handed transform in mm')
    return p


def projection_jacobian(calibration, xy, plane, pose, epsilon_mm=1e-4):
    xy = np.asarray(xy, float)
    columns = []
    for d in np.eye(2)*epsilon_mm:
        a, va = calibration.plane_project(xy+d, plane, pose)
        b, vb = calibration.plane_project(xy-d, plane, pose)
        if not np.all(va & vb):
            raise GeometryError('invalid ray for uncertainty Jacobian')
        columns.append((a-b)/(2*epsilon_mm))
    return np.stack(columns, axis=-1)
