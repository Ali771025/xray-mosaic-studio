"""Image-led perspective mosaic for existing colour/X-ray desktop workflows.

This path is VISUAL_ONLY: a homography registers one dominant plane, not a 3D
transmission volume. Absolute global transforms are fitted to original frames,
then each frame is sampled once. Disconnected components are never fabricated.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
import os
import shutil
import tempfile
import math
import re
import numpy as np
import cv2 as cv
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


@dataclass
class Options:
    feature_max_dim: int = 1400
    features: int = 6000
    ratio: float = .75
    min_inliers: int = 12
    min_inlier_ratio: float = .25
    ransac_px: float = 3.
    max_output_pixels: int = 60_000_000
    min_area_ratio: float = .1
    max_area_ratio: float = 10.
    optimize: bool = True
    model: str = 'projective'  # affine is an explicit scan prior, never inferred depth
    blend: str = 'seam'  # source selects one original; feather is appearance-only


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]


def display_gray(image):
    a = image[..., :3] if image.ndim == 3 else image
    if a.ndim == 3:
        a = cv.cvtColor(a, cv.COLOR_BGR2GRAY)
    if a.dtype == np.uint8:
        return a
    finite = np.isfinite(a)
    lo, hi = np.percentile(a[finite], [.5, 99.5]) if finite.any() else (0, 1)
    return np.clip((a.astype(float)-lo)*255/max(hi-lo, 1), 0, 255).astype(np.uint8)


def transform(points, H):
    points = np.asarray(points, float)
    hom = np.c_[points.reshape(-1, 2), np.ones(points.size//2)] @ H.T
    if (np.abs(hom[:, 2]) < 1e-10).any():
        raise ValueError('projective horizon')
    return (hom[:, :2]/hom[:, 2:]).reshape(points.shape)


def corners(shape):
    h, w = shape[:2]
    return np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], float)


@dataclass
class Features:
    points: np.ndarray
    descriptors: np.ndarray | None
    scale: float
    shape: tuple
    index: object = None


def extract(image, options):
    gray = display_gray(image)
    scale = min(1., options.feature_max_dim/max(gray.shape))
    if scale < 1:
        gray = cv.resize(gray, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    gray = cv.createCLAHE(2., (8, 8)).apply(gray)
    mask = None
    if image.ndim == 3 and image.shape[2] == 4:
        mask = cv.resize((image[..., 3] > 0).astype('uint8')*255, (gray.shape[1], gray.shape[0]), interpolation=cv.INTER_NEAREST)
    keypoints, desc = cv.SIFT_create(nfeatures=options.features, contrastThreshold=.015).detectAndCompute(gray, mask)
    if desc is not None:
        desc = np.sqrt(desc/np.maximum(desc.sum(axis=1, keepdims=True), 1e-12)).astype('float32')
    index = None
    if desc is not None and len(desc) >= 2:
        cv.setRNGSeed(421)
        index = cv.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=80))
        index.add([desc])
        index.train()
    return Features(np.array([p.pt for p in keypoints], float).reshape(-1, 2)/scale, desc, scale, image.shape, index)


def sane_homography(H, shape, options):
    if H is None or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        return False
    p = corners(shape)
    den = np.c_[p, np.ones(4)] @ H[2]
    if np.min(den)*np.max(den) <= 0:
        return False
    q = transform(p, H).astype('float32')
    area = cv.contourArea(q, oriented=True)
    base = max((shape[0]-1)*(shape[1]-1), 1)
    return bool(cv.isContourConvex(q) and options.min_area_ratio < area/base < options.max_area_ratio)


def match_features(reference, moving, options=None):
    o = options or Options()
    if reference.descriptors is None or moving.descriptors is None or min(len(reference.points), len(moving.points)) < o.min_inliers:
        return None, {'reason': 'INSUFFICIENT_FEATURES'}
    forward = reference.index.knnMatch(moving.descriptors, k=2)
    reverse = moving.index.knnMatch(reference.descriptors, k=2)
    reverse_good = {m.queryIdx: m.trainIdx for pair in reverse if len(pair) == 2 for m, n in [pair] if m.distance < o.ratio*n.distance}
    good = [m for pair in forward if len(pair) == 2 for m, n in [pair]
            if m.distance < o.ratio*n.distance and reverse_good.get(m.trainIdx) == m.queryIdx]
    if len(good) < o.min_inliers:
        return None, {'reason': 'INSUFFICIENT_MUTUAL_MATCHES', 'matches': len(good)}
    src = moving.points[[m.queryIdx for m in good]]
    dst = reference.points[[m.trainIdx for m in good]]
    threshold = o.ransac_px/min(reference.scale, moving.scale)
    cv.setRNGSeed(421)
    if o.model == 'affine':
        A, mask = cv.estimateAffine2D(src, dst, method=cv.RANSAC, ransacReprojThreshold=threshold,
                                    maxIters=10000, confidence=.999, refineIters=20)
        H = np.vstack((A, [0.,0.,1.])) if A is not None else None
    elif o.model == 'projective':
        H, mask = cv.findHomography(src, dst, cv.USAC_MAGSAC, threshold, maxIters=10000, confidence=.999)
    else:
        raise ValueError('model must be projective or affine')
    if H is None:
        return None, {'reason': 'HOMOGRAPHY_FAILED', 'matches': len(good)}
    mask = mask.ravel().astype(bool)
    count = int(mask.sum())
    ratio = count/len(good)
    if count < o.min_inliers or ratio < o.min_inlier_ratio or not sane_homography(H, moving.shape, o):
        return None, {'reason': 'GEOMETRY_OR_INLIER_GATE', 'inliers': count, 'matches': len(good)}
    src, dst = src[mask], dst[mask]
    # A narrow collinear circuit trace must not determine a full projective warp.
    spread = min(abs(cv.contourArea(cv.convexHull(src.astype('float32')))), abs(cv.contourArea(cv.convexHull(dst.astype('float32')))))
    span_src = np.ptp(src, axis=0)/np.array(moving.shape[:2][::-1])
    span_dst = np.ptp(dst, axis=0)/np.array(reference.shape[:2][::-1])
    if (spread < max(100/min(reference.scale, moving.scale)**2, .003*min(np.prod(reference.shape[:2]), np.prod(moving.shape[:2])))
            or min(span_src.min(), span_dst.min()) < .04):
        return None, {'reason': 'DEGENERATE_MATCH_DISTRIBUTION', 'inliers': count}
    residual = np.linalg.norm(transform(src, H)-dst, axis=1)
    score = count*ratio/(1+np.median(residual)*min(reference.scale, moving.scale))
    return {'H': H/H[2, 2], 'src': src, 'dst': dst, 'score': float(score)}, {
        'reason': 'ACCEPTED_DOMINANT_PLANE', 'matches': len(good), 'inliers': count,
        'inlier_ratio': ratio, 'median_px': float(np.median(residual)),
        'p95_px': float(np.percentile(residual, 95)), 'score': float(score)}


def estimate_pair(reference, moving, options=None):
    o = options or Options()
    pair, qa = match_features(extract(reference, o), extract(moving, o), o)
    return (pair['H'] if pair else None), qa


def components(count, edges):
    adjacency = {i: set() for i in range(count)}
    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    groups, seen = [], set()
    for start in range(count):
        if start in seen:
            continue
        stack, group = [start], []
        seen.add(start)
        while stack:
            node = stack.pop()
            group.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor not in seen:
                    stack.append(neighbor)
                    seen.add(neighbor)
        groups.append(sorted(group))
    return sorted(groups, key=lambda g: (-len(g), g[0]))


def initialize(group, edges, return_tree=False):
    # Maximum-evidence spanning tree; anchor with greatest graph support.
    anchor = max(group, key=lambda i: sum(e['score'] for (a, b), e in edges.items() if i in (a, b)))
    matrices = {anchor: np.eye(3)}
    tree = set()
    while len(matrices) < len(group):
        candidates = [(e['score'], i, j, e) for (i, j), e in edges.items()
                      if i in group and j in group and ((i in matrices) != (j in matrices))]
        if not candidates:
            raise ValueError('disconnected transform graph')
        _, i, j, e = max(candidates, key=lambda t: t[0])
        tree.add((i, j))
        if i in matrices:
            matrices[j] = matrices[i] @ e['H']
        else:
            matrices[i] = matrices[j] @ np.linalg.inv(e['H'])
    normalized = {i: h/h[2, 2] for i, h in matrices.items()}
    return (anchor, normalized, tree) if return_tree else (anchor, normalized)


def reject_inconsistent_cycles(count, edges, tolerance_px):
    """Reject a weak geometric constraint when a cycle contradicts its placement.

    Absolute distinct inlier support prevents a high ratio from only a handful
    of repeated pins from outweighing a broader consensus. No layout is inferred
    for disconnected data. Tree-only ambiguity remains unobservable.
    """
    kept = dict(edges)
    rejected = []
    while True:
        conflict = None
        for group in components(count, kept):
            if len(group) < 3:
                continue
            anchor, poses, tree = initialize(group, kept, return_tree=True)
            for key, edge in kept.items():
                i, j = key
                if i not in poses or j not in poses or key in tree:
                    continue
                error = float(np.median(np.linalg.norm(transform(edge['dst'], poses[i])-transform(edge['src'], poses[j]), axis=1)))
                if error <= tolerance_px:
                    continue
                adjacency = {node: [] for node in group}
                for a, b in tree:
                    adjacency[a].append((b, (a,b)))
                    adjacency[b].append((a, (a,b)))
                pending = [(i, [])]
                seen = {i}
                path = None
                while pending:
                    node, route = pending.pop()
                    if node == j:
                        path = route
                        break
                    for neighbor, link in adjacency[node]:
                        if neighbor not in seen:
                            seen.add(neighbor)
                            pending.append((neighbor, route+[link]))
                cycle = path+[key]
                def support(link):
                    e = kept[link]
                    unique = len(np.unique(np.rint(np.c_[e['src'],e['dst']]),axis=0))
                    residual = np.median(np.linalg.norm(transform(e['src'],e['H'])-e['dst'],axis=1))
                    return unique/(1+residual)
                weakest = min(cycle, key=lambda link: (support(link), kept[link]['score'], link))
                conflict = {'edge': list(weakest), 'reason': 'REJECTED_CYCLE_INCONSISTENCY',
                            'cycle': [list(link) for link in cycle],
                            'cycle_median_residual_px': error, 'tolerance_px': tolerance_px,
                            'distinct_support_score': float(support(weakest))}
                break
            if conflict:
                break
        if conflict is None:
            return kept, rejected
        del kept[tuple(conflict['edge'])]
        rejected.append(conflict)


def global_refine(group, anchor, matrices, edges, shapes, affine=False):
    # Optimise absolute projective maps in normalized coordinates; anchor fixes gauge.
    scale = float(max(max(shapes[i][:2]) for i in group))
    N = np.diag([1/scale, 1/scale, 1.])
    inverse_N = np.linalg.inv(N)
    ids = [i for i in group if i != anchor]
    normalized = {i: N @ matrices[i] @ inverse_N for i in group}
    dof = 6 if affine else 8
    def pack(h):
        return (h/h[2, 2]).ravel()[:dof]
    def unpack(x):
        return {anchor: np.eye(3), **{i: np.r_[x[k*dof:(k+1)*dof], ([0.,0.,1.] if affine else [1.])].reshape(3, 3) for k, i in enumerate(ids)}}
    x0 = np.concatenate([pack(normalized[i]) for i in ids])
    relevant = []
    for (i, j), edge in edges.items():
        if i not in group or j not in group:
            continue
        # Evenly cap constraints to keep the global solve bounded.
        indices = np.linspace(0, len(edge['src'])-1, min(120, len(edge['src']))).astype(int)
        relevant.append((i, j, edge['dst'][indices]/scale, edge['src'][indices]/scale))
    def residual(x):
        hs = unpack(x)
        try:
            return np.concatenate([(transform(a, hs[i])-transform(b, hs[j])).ravel() for i, j, a, b in relevant])
        except ValueError:
            return np.full(sum(2*len(a) for _, _, a, _ in relevant), 1e6)
    rows = sum(2*len(a) for _, _, a, _ in relevant)
    sparsity = lil_matrix((rows, len(x0)), dtype=int)
    row = 0
    for i, j, a, b in relevant:
        for frame in (i, j):
            if frame in ids:
                column = ids.index(frame)*dof
                sparsity[row:row+2*len(a), column:column+dof] = 1
        row += 2*len(a)
    fit = least_squares(residual, x0, loss='soft_l1', f_scale=2/scale, max_nfev=80, jac_sparsity=sparsity.tocsr(), x_scale='jac')
    optimized = {i: inverse_N @ h @ N for i, h in unpack(fit.x).items()}
    o = Options()
    def edge_max(hs):
        return max(float(np.median(np.linalg.norm(transform(e['dst'], hs[i])-transform(e['src'], hs[j]), axis=1)))
                   for (i,j),e in edges.items() if i in hs and j in hs)
    # An optimizer must not hide a catastrophic edge behind a good median over
    # all the other correspondences.
    edge_gate = edge_max(optimized) <= max(12., edge_max(matrices)*1.05)
    if edge_gate and all(sane_homography(optimized[i], shapes[i], o) for i in group) and np.median(np.abs(residual(fit.x))) <= np.median(np.abs(residual(x0)))*1.05+1e-9:
        return optimized, {'applied': True, 'optimizer_success': bool(fit.success), 'evaluations': fit.nfev}
    return matrices, {'applied': False, 'reason': 'GLOBAL_GEOMETRY_GATE'}


def seam_masks(images, group, matrices, shape):
    """Optimize source selection at reduced scale; no geometric deformation."""
    height, width = shape
    scale = min(1., 900/max(height, width))
    S = np.diag([scale, scale, 1.])
    warped, masks, offsets = [], [], []
    for i in group:
        H = S @ matrices[i]
        q = transform(corners(images[i].shape), H)
        lo = np.floor(q.min(axis=0)).astype(int)
        hi = np.ceil(q.max(axis=0)).astype(int)+1
        w, h = map(int, hi-lo)
        shift = np.array([[1., 0, -lo[0]], [0, 1., -lo[1]], [0, 0, 1.]])
        im = images[i]
        data = im[..., :3] if im.ndim == 3 else np.repeat(im[..., None], 3, axis=2)
        maximum = np.iinfo(im.dtype).max
        data = data.astype('float32')*(255./maximum)
        valid = ((im[..., 3] > 0).astype('uint8')*255 if im.ndim == 3 and im.shape[2] == 4
                 else np.full(im.shape[:2], 255, 'uint8'))
        warped.append(cv.UMat(cv.warpPerspective(data, shift @ H, (w, h), flags=cv.INTER_LINEAR)))
        masks.append(cv.UMat(cv.warpPerspective(valid, shift @ H, (w, h), flags=cv.INTER_NEAREST)))
        offsets.append(tuple(map(int, lo)))
    # Dynamic programming changes source masks only; geometry is unchanged.
    finder = cv.detail_DpSeamFinder('COLOR_GRAD')
    finder.find(warped, offsets, masks)
    return {i: (mask.get(), offset, scale) for i, mask, offset in zip(group, masks, offsets)}


def compose(images, group, matrices, options, cancelled=None):
    if cancelled and cancelled():
        raise RuntimeError('Visual mosaic cancelled')
    if len(group) == 1 and np.array_equal(matrices[group[0]], np.eye(3)):
        i = group[0]
        if images[i].shape[0]*images[i].shape[1] > options.max_output_pixels:
            raise ValueError('Single frame exceeds configured output pixel budget')
        original = images[i]
        maximum = np.iinfo(original.dtype).max
        if original.ndim == 2:
            rgb = np.repeat(original[..., None], 3, axis=2)
            valid = np.ones(original.shape, bool)
        else:
            rgb = original[..., :3]
            valid = original[..., 3] > 0 if original.shape[2] == 4 else np.ones(original.shape[:2], bool)
        bgra = np.dstack((rgb, valid.astype(original.dtype)*maximum))
        return {'image': bgra, 'coverage': valid.astype('uint16'),
                'source': np.where(valid, i, -1).astype('int32'),
                'transforms': {str(i): np.eye(3).tolist()}, 'origin_anchor_px': [0, 0],
                'qa': {'status': 'VISUAL_ONLY', 'shape': list(original.shape[:2]),
                       'gap_fraction': float(1-valid.mean()), 'overlap_fraction': 0.,
                       'overlap_intensity_rms': None, 'blend': options.blend, 'model': options.model, 'seam_fallback': None,
                       'metric_accuracy': 'UNVERIFIED',
                       'limitation': 'Unmatched original; no attachment is verified.'}}
    projected = [transform(corners(images[i].shape), matrices[i]) for i in group]
    lo = np.floor(np.concatenate(projected).min(axis=0)).astype(int)
    hi = np.ceil(np.concatenate(projected).max(axis=0)).astype(int)+1
    width, height = map(int, hi-lo)
    if min(width, height) <= 0 or width*height > options.max_output_pixels:
        raise ValueError(f'Unsafe output extent {width}x{height}; rejected instead of cropping data')
    shift = np.array([[1., 0, -lo[0]], [0, 1., -lo[1]], [0, 0, 1.]])
    # Work in float for native uint16; retain every colour channel.
    color = any(im.ndim == 3 and im.shape[2] >= 3 for im in images)
    channels = 3 if color else 1
    shape = (height, width, channels)
    result = np.zeros(shape, np.float32)
    best = np.zeros((height, width), np.float32)
    source = np.full((height, width), -1, np.int32)
    coverage = np.zeros((height, width), np.uint16)
    overlap_sse, overlap_count = 0., 0
    transformed = {}
    selected_masks = {}
    seam_error = None
    if options.blend == 'seam' and len(group) > 1:
        try:
            selected_masks = seam_masks(images, group, {i: shift @ matrices[i] for i in group}, (height, width))
        except cv.error as error:
            seam_error = str(error)
    for i in group:
        image = images[i]
        data = image[..., :3] if image.ndim == 3 else image
        if color and data.ndim == 2:
            data = np.repeat(data[..., None], 3, axis=2)
        if not color:
            data = data[..., None]
        valid = (image[..., 3] > 0).astype('uint8') if image.ndim == 3 and image.shape[2] == 4 else np.ones(image.shape[:2], 'uint8')
        distance = cv.distanceTransform(np.pad(valid, 1), cv.DIST_L2, 3)[1:-1, 1:-1]
        H = shift @ matrices[i]
        transformed[str(i)] = H.tolist()
        # Bounded intermediate strips, one consolidated original-image warp.
        float_data = data.astype('float32')
        for y in range(0, height, 256):
            if cancelled and cancelled():
                raise RuntimeError('Visual mosaic cancelled')
            rows = min(256, height-y)
            T = np.array([[1., 0, 0], [0, 1., -y], [0, 0, 1.]]) @ H
            sample = cv.warpPerspective(float_data, T, (width, rows), flags=cv.INTER_LINEAR)
            if sample.ndim == 2:
                sample = sample[..., None]
            support = cv.warpPerspective(valid.astype('float32'), T, (width, rows), flags=cv.INTER_LINEAR) >= .999
            weight = cv.warpPerspective(distance, T, (width, rows), flags=cv.INTER_LINEAR)*support
            out, count = result[y:y+rows], coverage[y:y+rows]
            overlap = support & (count > 0)
            if overlap.any():
                diff = sample[overlap]-out[overlap]
                overlap_sse += float(np.sum(diff.astype(float)**2))
                overlap_count += int(diff.size)
            if options.blend == 'feather':
                total = best[y:y+rows]+weight
                out[:] = (out*best[y:y+rows, :, None]+sample*weight[..., None])/np.maximum(total[..., None], 1e-10)
                source[y:y+rows][weight > best[y:y+rows]] = i
                best[y:y+rows] = total
            else:
                priority = weight
                if i in selected_masks:
                    mask, (cx, cy), seam_scale = selected_masks[i]
                    mapping = np.array([[1/seam_scale, 0, cx/seam_scale], [0, 1/seam_scale, cy/seam_scale-y]], float)
                    preferred = cv.warpAffine(mask, mapping, (width, rows), flags=cv.INTER_NEAREST) > 0
                    priority = weight + (preferred & support)*1e6
                choose = priority > best[y:y+rows]
                out[choose] = sample[choose]
                best[y:y+rows][choose] = priority[choose]
                source[y:y+rows][choose] = i
            count[support] += 1
    dtype = np.result_type(*[images[i].dtype for i in group])
    if dtype not in (np.uint8, np.uint16):
        dtype = np.float32
    if np.issubdtype(dtype, np.integer):
        result = np.clip(np.rint(result), 0, np.iinfo(dtype).max).astype(dtype)
    else:
        result = result.astype(dtype)
    alpha = ((coverage > 0)*np.iinfo(dtype).max).astype(dtype) if np.issubdtype(dtype, np.integer) else (coverage > 0).astype(dtype)
    bgra = np.dstack((result if color else np.repeat(result, 3, axis=2), alpha))
    return {'image': bgra, 'coverage': coverage, 'source': source, 'transforms': transformed,
            'origin_anchor_px': lo.tolist(), 'qa': {'status': 'VISUAL_ONLY', 'shape': [height, width],
            'gap_fraction': float(np.mean(coverage == 0)), 'overlap_fraction': float(np.mean(coverage > 1)),
            'overlap_intensity_rms': math.sqrt(overlap_sse/overlap_count) if overlap_count else None,
            'blend': options.blend, 'model': options.model, 'seam_fallback': seam_error, 'metric_accuracy': 'UNVERIFIED',
            'limitation': 'Dominant-plane alignment; multi-depth parallax can remain. Blending is not geometric proof.'}}


def build_mosaic(images, options=None, progress=None, cancelled=None, checkpoint_directory=None):
    o = options or Options()
    if not images:
        raise ValueError('No images')
    cv.setNumThreads(4)
    cache = Path(checkpoint_directory) if checkpoint_directory else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    features = []
    for i, image in enumerate(images):
        if cancelled and cancelled():
            raise RuntimeError('Visual mosaic cancelled')
        if image is None or image.dtype not in (np.uint8, np.uint16):
            raise ValueError('Expected decoded uint8/uint16 images')
        cached = cache/f'features_{i:04d}.npz' if cache else None
        if cached and cached.exists():
            with np.load(cached, allow_pickle=False) as data:
                desc = data['descriptors']
                if desc.size == 0:
                    desc = None
                index = None
                if desc is not None and len(desc) >= 2:
                    cv.setRNGSeed(421)
                    index = cv.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=80))
                    index.add([desc])
                    index.train()
                feature = Features(data['points'], desc, float(data['scale']), image.shape, index)
        else:
            feature = extract(image, o)
            if cached:
                temporary = cached.with_suffix('.tmp.npz')
                np.savez_compressed(temporary, points=feature.points,
                                    descriptors=feature.descriptors if feature.descriptors is not None else np.array([], np.float32),
                                    scale=feature.scale)
                os.replace(temporary, cached)
        features.append(feature)
        if progress:
            progress('features', i+1, len(images))
    edges, pairs = {}, []
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            if cancelled and cancelled():
                raise RuntimeError('Visual mosaic cancelled')
            cached_pair = cache/f'pair_{i:04d}_{j:04d}.npz' if cache else None
            if cached_pair and cached_pair.exists():
                with np.load(cached_pair, allow_pickle=False) as data:
                    qa = json.loads(str(data['qa']))
                    pair = ({'H': data['H'], 'src': data['src'], 'dst': data['dst'], 'score': float(data['score'])}
                            if bool(data['accepted']) else None)
            else:
                pair, qa = match_features(features[i], features[j], o)
                if cached_pair:
                    temporary = cached_pair.with_suffix('.tmp.npz')
                    np.savez_compressed(temporary, qa=json.dumps(qa), accepted=pair is not None,
                                        **(pair or {}))
                    os.replace(temporary, cached_pair)
            pairs.append({'reference': i, 'moving': j, **qa})
            if pair:
                edges[i, j] = pair
        if progress:
            progress('pairs', i+1, len(images))
    cycle_tolerance = 4*o.ransac_px/min(f.scale for f in features)
    edges, rejected_cycles = reject_inconsistent_cycles(len(images), edges, cycle_tolerance)
    for rejection in rejected_cycles:
        i,j = rejection['edge']
        for pair_qa in pairs:
            if pair_qa['reference'] == i and pair_qa['moving'] == j:
                pair_qa['pair_reason'] = pair_qa['reason']
                pair_qa.update(rejection)
    groups = components(len(images), edges)
    mosaics = []
    for group in groups:
        if len(group) == 1:
            matrices, anchor, optimization = {group[0]: np.eye(3)}, group[0], {'applied': False, 'reason': 'NO_VERIFIED_OVERLAP'}
        else:
            anchor, matrices = initialize(group, edges)
            optimization = {'applied': False}
            if o.optimize and len(group) >= 3:
                matrices, optimization = global_refine(group, anchor, matrices, edges, [im.shape for im in images], affine=o.model == 'affine')
        residuals = []
        for (i, j), edge in edges.items():
            if i in group and j in group:
                error = np.linalg.norm(transform(edge['dst'], matrices[i])-transform(edge['src'], matrices[j]), axis=1)
                residuals.append({'reference': i, 'moving': j, 'median_px': float(np.median(error)),
                                  'p95_px': float(np.percentile(error, 95)), 'count': len(error)})
        rendered = compose(images, group, matrices, o, cancelled)
        rendered['qa'].update({'frames': group, 'anchor': anchor, 'optimization': optimization,
                               'global_match_residuals': residuals,
                               'cycle_rejections': [r for r in rejected_cycles if any(i in group for i in r['edge'])]})
        mosaics.append(rendered)
    return {'components': mosaics, 'pairs': pairs, 'connected': len(groups) == 1,
            'frame_count': len(images), 'feature_counts': [len(f.points) for f in features]}


def stitch_directory(directory, destination, options=None, progress=None):
    from xray_mosaic.pipeline import file_hash
    directory, destination = Path(directory), Path(destination)
    if destination.exists():
        raise ValueError('Choose a new output directory')
    files = sorted([p for p in directory.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp')], key=natural_key)
    images = [cv.imread(str(p), cv.IMREAD_UNCHANGED) for p in files]
    o = options or Options()
    input_records = [{'path': str(p.resolve()), 'sha256': file_hash(p)} for p in files]
    # Content/profile-keyed checkpoints survive host restarts without reusing stale matches.
    signature = hashlib.sha256(json.dumps({'files': input_records, 'options': vars(o),
                                           'algorithm': 'rootsift-mutual-magsac-v2'}, sort_keys=True).encode()).hexdigest()
    cache = destination.parent/'.registration-cache'/signature
    result = build_mosaic(images, o, progress, checkpoint_directory=cache)
    destination.mkdir(parents=True)
    report = {'algorithm': 'rootsift-mutual-magsac-global-homography-v3-cycle-gated', 'options': vars(options or Options()), 'status': 'VISUAL_ONLY', 'connected': result['connected'], 'frame_count': len(files),
              'files': input_records,
              'pairs': result['pairs'], 'feature_counts': result['feature_counts'], 'components': []}
    for k, component in enumerate(result['components']):
        name = f'mosaic_{k:02d}'
        # Encode locally, then copy a complete file. Mounted workspaces can make
        # libpng's many small writes much slower than a sequential file copy.
        with tempfile.TemporaryDirectory(prefix='visual-export-') as temporary:
            local = Path(temporary)
            if not cv.imwrite(str(local/(name+'.png')), component['image'], [cv.IMWRITE_PNG_COMPRESSION, 1]):
                raise OSError('Cannot write mosaic PNG')
            if not cv.imwrite(str(local/(name+'_coverage.png')), component['coverage']):
                raise OSError('Cannot write coverage PNG')
            np.savez_compressed(local/(name+'_source.npz'), source=component['source'])
            for saved in local.iterdir():
                shutil.copyfile(saved, destination/saved.name)
        qa = dict(component['qa'], output=name+'.png', transforms=component['transforms'], origin_anchor_px=component['origin_anchor_px'])
        report['components'].append(qa)
    (destination/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=['projective','affine'], default='projective')
    parser.add_argument('--blend', choices=['source', 'feather', 'seam'], default='seam')
    args = parser.parse_args()
    print(json.dumps(stitch_directory(args.input, args.output, Options(blend=args.blend, model=args.model),
                                    lambda stage, i, n: print(f'{stage}: {i}/{n}', flush=True)), indent=2))
