"""Optional anchored global translation fit from externally verified plane markers.
No unrestricted homographies and no inference from zero overlap.
"""
import numpy as np
from scipy.optimize import least_squares
from .geometry import GeometryError


def refine_plane_offsets(prior_mm, sigma_mm, edges, max_deviation_mm, max_residual_mm):
    """edges: (i,j, delta_xy_mm, sigma_mm) from genuine same-plane overlap markers.
    Anchor frame 0 exactly; include independent per-frame encoder priors.
    Returns proposals only. Batch never silently applies them to stage poses.
    """
    prior, sigmas = np.asarray(prior_mm, float), np.asarray(sigma_mm, float)
    if (prior.ndim != 2 or prior.shape[1] != 2 or sigmas.shape != (len(prior),) or
            len(prior) < 2 or not np.isfinite(prior).all() or
            not np.isfinite(sigmas).all() or (sigmas <= 0).any() or
            not np.isfinite([max_deviation_mm, max_residual_mm]).all() or
            min(max_deviation_mm, max_residual_mm) <= 0):
        raise GeometryError('finite priors, positive sigma and configured gates required')
    if not edges:
        return {'accepted': False, 'reason': 'INSUFFICIENT_OVERLAP', 'offsets_mm': prior.tolist()}
    selected, rejected = [], []
    for i, j, delta, sigma in edges:
        delta = np.asarray(delta, float)
        if not (0 <= i < len(prior) and 0 <= j < len(prior)) or i == j:
            raise GeometryError('invalid marker edge indices')
        if delta.shape != (2,) or not np.isfinite(delta).all() or not np.isfinite(sigma) or sigma <= 0:
            raise GeometryError('invalid marker edge')
        if np.linalg.norm(delta-(prior[j]-prior[i])) > 2*max_deviation_mm:
            rejected.append([i, j])
        else:
            selected.append((i, j, delta, sigma))
    if not selected:
        return {'accepted': False, 'reason': 'ALL_EDGES_REJECTED', 'rejected': rejected,
                'offsets_mm': prior.tolist()}
    def unpack(x):
        return np.vstack((prior[0], x.reshape(-1, 2)))
    def residual(x):
        p = unpack(x)
        return np.r_[((p[1:]-prior[1:])/sigmas[1:, None]).ravel(),
                     np.concatenate([(p[j]-p[i]-d)/s for i, j, d, s in selected])]
    fit = least_squares(residual, prior[1:].ravel(), loss='soft_l1')
    proposed = unpack(fit.x)
    errors = np.array([np.linalg.norm(proposed[j]-proposed[i]-d) for i, j, d, s in selected])
    ok = bool(fit.success and np.max(np.linalg.norm(proposed-prior, axis=1)) <= max_deviation_mm
              and np.max(errors) <= max_residual_mm)
    return {'accepted': ok, 'reason': 'VERIFIED_MARKER_CONSTRAINTS' if ok else 'RESIDUAL_OR_STAGE_GATE',
            'offsets_mm': (proposed if ok else prior).tolist(), 'rejected': rejected,
            'residuals_mm': errors.tolist(), 'anchor_frame': 0}
