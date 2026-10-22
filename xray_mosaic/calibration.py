"""Observable calibration only: known source, detector pose and known 3D fiducials.
Fit two pitches and principal point. Source distance and unknown depth are NOT fitted.
"""
import copy
import numpy as np
from scipy.optimize import least_squares
from .geometry import Calibration, GeometryError, projection_jacobian


def fit_detector_sampling(profile, training_world_mm, training_uv_px,
                          heldout_world_mm, heldout_uv_px, acceptance_px, sigma_plane_mm):
    cal = Calibration(**profile)
    train = np.asarray(training_world_mm, float)
    observed = np.asarray(training_uv_px, float)
    test, test_uv = np.asarray(heldout_world_mm, float), np.asarray(heldout_uv_px, float)
    if (train.ndim != 2 or train.shape[1] != 3 or observed.shape != (len(train), 2) or
            test.ndim != 2 or test.shape[1] != 3 or test_uv.shape != (len(test), 2) or
            len(train) < 6 or len(test) < 3 or
            not all(np.isfinite(a).all() for a in (train, observed, test, test_uv))):
        raise GeometryError('need >=6 training and >=3 independent held-out known 3D markers')
    if any(np.linalg.norm(train-p, axis=1).min() < 1e-8 for p in test):
        raise GeometryError('held-out points must be independent of training points')
    if not np.isfinite([acceptance_px, sigma_plane_mm]).all() or acceptance_px <= 0 or sigma_plane_mm < 0:
        raise GeometryError('measured uncertainty and positive customer residual threshold required')
    projected, ok = cal.project(train)
    if not ok.all():
        raise GeometryError('invalid calibration marker geometry')
    detector_xy = (projected-cal.principal_px)*cal.pitch_mm
    if (np.ptp(detector_xy, axis=0) < 1e-6).any():
        raise GeometryError('degenerate marker spread: both detector axes must vary')
    design = np.zeros((2*len(train), 4))
    design[0::2, 0], design[1::2, 1] = detector_xy.T
    design[0::2, 2], design[1::2, 3] = 1, 1
    condition = float(np.linalg.cond(design/np.linalg.norm(design, axis=0)))
    if not np.isfinite(condition) or condition > 1e8:
        raise GeometryError('ill-conditioned calibration')
    initial = np.r_[1/np.array(cal.pitch_mm), cal.principal_px]
    fit = least_squares(lambda p: design @ p-observed.ravel(), initial,
                        bounds=([1e-9, 1e-9, -np.inf, -np.inf], np.full(4, np.inf)),
                        loss='soft_l1', f_scale=acceptance_px)
    if not fit.success:
        raise GeometryError('calibration optimizer failed')
    out = copy.deepcopy(profile)
    out['pitch_mm'], out['principal_px'] = (1/fit.x[:2]).tolist(), fit.x[2:].tolist()
    out['validation'] = {}
    fitted = Calibration(**out)
    prediction, valid = fitted.project(test)
    if not valid.all():
        raise GeometryError('invalid held-out marker rays')
    errors = np.linalg.norm(prediction-test_uv, axis=1)
    rms = float(np.sqrt(np.mean(errors**2)))
    residual = (design @ fit.x-observed.ravel()).reshape(-1, 2)
    inlier = np.linalg.norm(residual, axis=1) <= acceptance_px
    if inlier.sum() < 6 or (np.ptp(detector_xy[inlier], axis=0) < 1e-6).any():
        raise GeometryError('insufficient distributed calibration inliers after rejection')
    # Covariance is approximate under robust loss and is labelled accordingly.
    covariance = np.linalg.pinv(fit.jac.T @ fit.jac)*float(np.mean(residual[inlier]**2))
    out['validation'] = {'status': 'validated' if rms <= acceptance_px else 'rejected',
                         'heldout_rms_px': rms, 'heldout_count': len(test),
                         'max_rms_px': acceptance_px, 'sigma_plane_mm': sigma_plane_mm,
                         'condition_number_scaled': condition, 'training_inliers': int(inlier.sum()),
                         'training_count': len(train), 'heldout_errors_px': errors.tolist(),
                         'approx_parameter_covariance': covariance.tolist(),
                         'parameter_order': ['inverse_pitch_u', 'inverse_pitch_v', 'principal_u', 'principal_v'],
                         'heldout_rms_mm': None,
                         'note': 'mm residual requires a defined target plane; source/detector pose/depth fixed independently'}
    out['provenance'] = dict(out['provenance'], sampling='estimated_from_known_3D_markers')
    return out


def validate_plane_markers(cal, plane, pose, object_xy_mm, observed_uv_px):
    xy, observed = np.asarray(object_xy_mm, float), np.asarray(observed_uv_px, float)
    predicted, ok = cal.plane_project(xy, plane, pose)
    back, valid = cal.backproject(observed, plane, pose)
    if not (ok & valid).all():
        raise GeometryError('invalid held-out plane landmark')
    return {'rms_px': float(np.sqrt(np.mean(np.sum((predicted-observed)**2, axis=-1)))),
            'rms_plane_mm': float(np.sqrt(np.mean(np.sum((back-xy)**2, axis=-1)))),
            'jacobian_px_per_mm': projection_jacobian(cal, xy, plane, pose).tolist()}
