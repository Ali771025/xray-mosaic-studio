"""Bounded visual residual mesh with image-space holdout exclusion and promotion gates.

This experimental correction is VISUAL_ONLY. It cannot reconstruct X-ray depth.
The original pair report and original rasters are never changed.
"""
from pathlib import Path
import json
import hashlib
import shutil
import tempfile
import cv2 as cv
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from visual_mosaic import display_gray, transform, corners


def control_boxes(points, H, radius=12):
    boxes = []
    for p in points:
        if p['role'] != 'check':
            continue
        a, b = np.array(p['reference']), np.array(p['moving'])
        boxes.append(np.r_[a-radius-2, a+radius+2])
        square = b+np.array([[-radius,-radius],[radius,-radius],[radius,radius],[-radius,radius]])
        projected = transform(square, H)
        boxes.append(np.r_[projected.min(axis=0)-2, projected.max(axis=0)+2])
    return np.array(boxes, float).reshape(-1,4)


def intersects_control(point, boxes, support_radius=29):
    p = np.asarray(point)
    return bool(len(boxes) and np.any(np.all(p+support_radius >= boxes[:,:2],axis=1) &
                                         np.all(p-support_radius <= boxes[:,2:],axis=1)))


def patch_location(template, search):
    if min(template.std(), search.std()) < 4:
        return None
    response = cv.matchTemplate(search, template, cv.TM_CCOEFF_NORMED)
    _, peak, _, (x,y) = cv.minMaxLoc(response)
    if peak < .92 or min(x,y) < 1 or max(x,y) > 15:
        return None
    def offset(left,center,right):
        den = left-2*center+right
        return float(.5*(left-right)/den) if abs(den) > 1e-8 else 0.
    delta = np.array([x+offset(response[y,x-1],response[y,x],response[y,x+1])-8,
                      y+offset(response[y-1,x],response[y,x],response[y+1,x])-8])
    response[max(0,y-3):y+4,max(0,x-3):x+4] = -1
    if peak-response.max() < .01:
        return None
    return delta, float(peak)


def measure_observations(images, H, boxes, stride=8):
    a,b = [display_gray(im) for im in images]
    warped = cv.warpPerspective(b,H,a.shape[::-1])
    alpha = [(im[...,3] > 0).astype('float32') if im.ndim == 3 and im.shape[2] == 4
             else np.ones(im.shape[:2], 'float32') for im in images]
    valid = (cv.warpPerspective(alpha[1],H,a.shape[::-1]) > .999) & (alpha[0] > .999)
    support = cv.erode(valid.astype('uint8'),np.ones((59,59),'uint8')) > 0
    measured, skipped = [], 0
    for y in range(30,a.shape[0]-30,stride):
        for x in range(30,a.shape[1]-30,stride):
            if not support[y,x]:
                continue
            p = np.array([x,y],float)
            if intersects_control(p,boxes):
                skipped += 1
                continue
            hit = patch_location(a[y-12:y+13,x-12:x+13],warped[y-20:y+21,x-20:x+21])
            if hit is None:
                continue
            d, score = hit
            q = p+d
            reverse = patch_location(cv.getRectSubPix(warped,(25,25),tuple(q)),
                                     cv.getRectSubPix(a,(41,41),tuple(q)))
            if reverse is None:
                continue
            roundtrip = float(np.linalg.norm(d+reverse[0]))
            if roundtrip > .75:
                continue
            measured.append([x,y,*d,score,roundtrip])
    return np.array(measured,float).reshape(-1,6), skipped


def build_mesh(observations, shape, spacing=16):
    if len(observations) < 48:
        raise ValueError('Insufficient independently supported local patches')
    points = observations[:,:2]
    if cv.contourArea(cv.convexHull(points.astype('float32'))) < .003*np.prod(shape[:2]):
        raise ValueError('Local patch support is spatially degenerate')
    rows,cols = (int(np.ceil((n-1)/spacing))+1 for n in shape[:2])
    yy,xx = np.indices((rows,cols),dtype=float)
    query = np.stack((xx.ravel(),yy.ravel()),axis=1)*spacing
    # Fixed settings; neither smoothing nor neighbour count is tuned on check errors.
    interpolator = RBFInterpolator(points/64,observations[:,2:4],kernel='thin_plate_spline',
                                   smoothing=1.,neighbors=min(24,len(points)))
    mesh = interpolator(query/64).reshape(rows,cols,2)
    hull = cv.convexHull((points/spacing).astype('float32'))
    support = np.zeros((rows,cols),'uint8')
    cv.fillConvexPoly(support,np.rint(hull).astype('int32'),1)
    border = cv.distanceTransform(support,cv.DIST_L2,3)
    nearest,_ = cKDTree(points).query(query)
    taper = np.clip(border/2,0,1)*np.clip((96-nearest.reshape(rows,cols))/32,0,1)
    mesh *= taper[...,None]
    if not np.isfinite(mesh).all() or np.linalg.norm(mesh,axis=2).max() > 8:
        raise ValueError('Residual mesh exceeds the 8 px displacement bound')
    return mesh.astype('float32'), spacing


def sample_mesh(mesh, spacing, points):
    shape = np.shape(points)
    pts = np.asarray(points,'float32')/spacing
    # Preserve raster dimensions: a flattened native image exceeds OpenCV's 32767-column limit.
    if pts.ndim == 3:
        return cv.remap(mesh,pts[...,0],pts[...,1],cv.INTER_LINEAR,
                        borderMode=cv.BORDER_CONSTANT,borderValue=0)
    flat = pts.reshape(-1,2)
    chunks = []
    for first in range(0,len(flat),16000):
        block = flat[first:first+16000]
        chunks.append(cv.remap(mesh,block[:,0][None,:],block[:,1][None,:],cv.INTER_LINEAR,
                               borderMode=cv.BORDER_CONSTANT,borderValue=0)[0])
    return np.concatenate(chunks).reshape(shape) if chunks else np.empty(shape,'float32')



def audit_mesh(mesh, spacing, shape):
    yy,xx = np.indices(shape[:2],dtype='float32')
    field = sample_mesh(mesh,spacing,np.stack((xx,yy),axis=-1))
    ux_y,ux_x = np.gradient(field[...,0])
    uy_y,uy_x = np.gradient(field[...,1])
    determinant = (1+ux_x)*(1+uy_y)-ux_y*uy_x
    gradient = np.sqrt(ux_x**2+ux_y**2+uy_x**2+uy_y**2)
    audit = dict(max_displacement_px=float(np.linalg.norm(field,axis=2).max()),
                 min_jacobian_determinant=float(determinant.min()),
                 max_displacement_gradient_norm=float(gradient.max()),
                 finite=bool(np.isfinite(field).all()))
    audit['safe'] = (audit['finite'] and audit['max_displacement_px'] <= 8 and
                     audit['min_jacobian_determinant'] >= .5 and
                     audit['max_displacement_gradient_norm'] <= .5)
    return audit,field


def evaluate_mesh(points,H,mesh,spacing,tolerance=1.):
    checking = [p for p in points if p['role'] == 'check']
    a = np.array([p['reference'] for p in checking]);b = np.array([p['moving'] for p in checking])
    base = transform(b,H)
    predicted = base.copy()
    for _ in range(16):
        predicted = base-sample_mesh(mesh,spacing,predicted)
    inverse = transform(a+sample_mesh(mesh,spacing,a),np.linalg.inv(H))
    errors = np.maximum(np.linalg.norm(predicted-a,axis=1),np.linalg.norm(inverse-b,axis=1))
    stats = dict(count=len(errors),median_px=float(np.median(errors)),p90_px=float(np.percentile(errors,90)),
                 max_px=float(errors.max()),within_tolerance=int((errors <= tolerance).sum()),tolerance_px=tolerance)
    rows = [dict(reference=p['reference'],moving=p['moving'],before_px=p['symmetric_max_px'],after_px=float(error))
            for p,error in zip(checking,errors)]
    return stats,rows


def promotion_gate(before,after,safe,tolerance=1.):
    # A smaller median cannot hide worse tails or a newly broken check.
    return bool(safe and after['max_px'] <= tolerance and
                after['max_px'] <= before['max_px'] and
                after['p90_px'] <= before['p90_px'] and
                after['median_px'] <= before['median_px'])


def render_candidate(images,H,mesh,spacing):
    margin = 10
    bounds = np.concatenate((corners(images[0].shape),transform(corners(images[1].shape),H)))
    lo = np.floor(bounds.min(axis=0)).astype(int)-margin
    hi = np.ceil(bounds.max(axis=0)).astype(int)+margin+1
    width,height = hi-lo
    if width*height > 60_000_000:
        raise ValueError('Output exceeds native pixel budget')
    yy,xx = np.indices((height,width),dtype='float32')
    reference_coords = np.stack((xx+float(lo[0]),yy+float(lo[1])),axis=-1).astype('float32')
    target = reference_coords+sample_mesh(mesh,spacing,reference_coords)
    moving_coords = transform(target,np.linalg.inv(H)).astype('float32')
    colors,masks = [],[]
    for image,coords in zip(images,(reference_coords,moving_coords)):
        data = image[..., :3] if image.ndim == 3 else np.repeat(image[...,None],3,axis=2)
        alpha = (image[...,3] > 0).astype('float32') if image.ndim == 3 and image.shape[2] == 4 else np.ones(image.shape[:2],'float32')
        # Each original contributes through one remap, without rewarping an intermediate mosaic.
        colors.append(cv.remap(data,coords[...,0],coords[...,1],cv.INTER_LINEAR))
        masks.append(cv.remap(alpha,coords[...,0],coords[...,1],cv.INTER_LINEAR) > .999)
    scale = min(1.,900/max(height,width))
    wh = (max(1,round(width*scale)),max(1,round(height*scale)))
    seam_masks = [cv.UMat(cv.resize(mask.astype('uint8')*255,wh,interpolation=cv.INTER_NEAREST)) for mask in masks]
    seam_images = [cv.UMat(cv.resize(color.astype('float32')*(255/np.iinfo(color.dtype).max),wh)) for color in colors]
    cv.detail_DpSeamFinder('COLOR_GRAD').find(seam_images,[(0,0),(0,0)],seam_masks)
    chosen = [cv.resize(mask.get(),(width,height),interpolation=cv.INTER_NEAREST)>0 for mask in seam_masks]
    out = np.zeros((height,width,3),np.result_type(*[im.dtype for im in images]))
    owner = np.full((height,width),-1,'int32')
    best = np.full((height,width),-1,'float32')
    for i in range(2):
        priority = cv.distanceTransform(masks[i].astype('uint8'),cv.DIST_L2,3)+chosen[i]*1e6
        pick = masks[i] & (priority > best)
        out[pick] = colors[i][pick];owner[pick] = i;best[pick] = priority[pick]
    coverage = masks[0].astype('uint16')+masks[1].astype('uint16')
    image = np.dstack((out,(coverage>0).astype(out.dtype)*np.iinfo(out.dtype).max))
    return dict(image=image,coverage=coverage,source=owner,origin_reference_px=lo.tolist(),
                inverse_map_moving=moving_coords)


def run(report_path,destination):
    report_path,destination = Path(report_path),Path(destination)
    if destination.exists():
        raise ValueError('Choose a new output directory')
    source = json.loads(report_path.read_text())
    evaluation = source['evaluation']
    images = []
    for record in source['files']:
        assert hashlib.sha256(Path(record['path']).read_bytes()).hexdigest() == record['sha256']
        images.append(cv.imread(record['path'],-1))
    H = np.array(evaluation['H_moving_to_reference'])
    boxes = control_boxes(evaluation['points'],H)
    observations,skipped = measure_observations(images,H,boxes)
    mesh,spacing = build_mesh(observations,images[0].shape)
    assert not any(intersects_control(p[:2],boxes) for p in observations)
    audit,field = audit_mesh(mesh,spacing,images[0].shape)
    after,rows = evaluate_mesh(evaluation['points'],H,mesh,spacing)
    promoted = promotion_gate(evaluation['check'],after,audit['safe'])
    record = dict(status='PROMOTED_VISUAL_ONLY' if promoted else 'REJECTED_LOCAL_CORRECTION',
                  promoted=promoted,files=source['files'],parent_report=str(report_path.resolve()),
                  parent_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                  before=evaluation['check'],after=after,checks=rows,geometry_audit=audit,
                  observation_count=len(observations),excluded_observation_centers=skipped,
                  control_patch_radius_px=12,observation_footprint_radius_px=29,
                  observation_search_radius_px=8,minimum_zncc=.92,minimum_peak_margin=.01,
                  max_roundtrip_px=.75,sampling_stride_px=8,mesh_spacing_px=spacing,
                  interpolation='thin_plate_spline, smoothing=1, neighbours=24, scale=64 px',
                  model_selection='fixed parameters; no check-point fitting or filtering',
                  metric_accuracy='UNVERIFIED',
                  limitation='Visual local deformation cannot reconstruct the multiple X-ray transmission depths. These image-derived controls are not human ground truth.')
    with tempfile.TemporaryDirectory(prefix='local-pair-') as temporary:
        local = Path(temporary)
        np.savez_compressed(local/'residual_mesh.npz',mesh=mesh,spacing=spacing,observations=observations,
                            excluded_control_boxes=boxes,H_moving_to_reference=H)
        magnitude = np.linalg.norm(field,axis=2)
        display = cv.applyColorMap(np.clip(magnitude*255/8,0,255).astype('uint8'),cv.COLORMAP_TURBO)
        cv.imwrite(str(local/'residual_field.png'),display)
        if audit['safe']:
            result = render_candidate(images,H,mesh,spacing)
            for name in ('image','coverage'):
                cv.imwrite(str(local/('local_candidate.png' if name == 'image' else 'coverage.png')),result[name],[cv.IMWRITE_PNG_COMPRESSION,1])
            np.savez_compressed(local/'source.npz',source=result['source'])
            record['origin_reference_px'] = result['origin_reference_px']
            # Persist the exact source-coordinate map for review/replay; never label it metric-valid.
            np.savez_compressed(local/'inverse_map_moving.npz',coordinates=result['inverse_map_moving'])
        destination.mkdir(parents=True)
        for file in local.iterdir():
            shutil.copyfile(file,destination/file.name)
    for file in source['files']:
        assert hashlib.sha256(Path(file['path']).read_bytes()).hexdigest() == file['sha256']
    (destination/'report.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    import html
    text = '<!doctype html><html lang="fa" dir="rtl"><meta charset="utf-8"><style>body{font:16px Tahoma;max-width:1100px;margin:auto;line-height:1.9}img{max-width:100%}pre{direction:ltr;white-space:pre-wrap}</style><h1>بررسی اصلاح موضعی MOTHERBOARD</h1>'
    text += '<p>'+('اصلاح تصویری پذیرفته شد؛ اعتبار متریک تأیید نشده است.' if promoted else 'اصلاح موضعی رد شد و خروجی قبلی جایگزین نشد. تصویر نامزد در صورت نمایش صرفاً برای بازبینی است.')+'</p>'
    text += '<pre>'+html.escape(json.dumps({k:record[k] for k in ('status','before','after','geometry_audit','observation_count')},indent=2))+'</pre>'
    for filename in ('local_candidate.png','residual_field.png'):
        if (destination/filename).exists():
            text += f'<a href="{filename}"><img src="{filename}"></a>'
    text += '<p><a href="report.json">گزارش کامل و تمام نقاط کنترل</a></p></html>'
    (destination/'index.html').write_text(text,encoding='utf-8')
    return record


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('report',type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    result=run(args.report,args.output)
    print(json.dumps({k:result[k] for k in ('status','before','after','geometry_audit','observation_count')},indent=2))
