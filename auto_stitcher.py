# auto_stitcher.py
import cv2 as cv
import numpy as np
try:
    import torch
    from SuperGlue_models.matching import Matching
    from SuperGlue_models.utils import frame2tensor
    torch.set_grad_enabled(False)
except ImportError:
    torch = None
    Matching = None


def _optional_no_grad(function):
    return torch.no_grad()(function) if torch is not None else function

config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.001,
        'max_keypoints': 1024
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.1,
    }
}

device = 'cuda' if torch is not None and torch.cuda.is_available() else 'cpu'
print(f'Auto-Stitcher: Running inference on device "{device}"')

matching = None

try:
    if Matching is None:
        raise RuntimeError('Optional PyTorch/SuperGlue dependencies unavailable; geometric/manual processing remains available')
    matching = Matching(config).eval().to(device)
    for p in matching.parameters():
        p.requires_grad = False
    print("Loaded SuperPoint and SuperGlue models successfully.")
except Exception as e:
    print(f"Error loading SuperGlue models: {e}")
def update_matching_config(keypoint_thresh=None, match_thresh=None):
    """به‌روزرسانی تنظیمات مدل به صورت بلادرنگ از رابط گرافیکی"""
    global matching, config
    changed = False
    if keypoint_thresh is not None and config['superpoint']['keypoint_threshold'] != keypoint_thresh:
        config['superpoint']['keypoint_threshold'] = float(keypoint_thresh)
        changed = True
    if match_thresh is not None and config['superglue']['match_threshold'] != match_thresh:
        config['superglue']['match_threshold'] = float(match_thresh)
        changed = True
    if changed and Matching is not None:
        try:
            matching = Matching(config).eval().to(device)
            for p in matching.parameters():
                p.requires_grad = False
            print(f"Auto-Stitcher: Updated config -> Keypoint Thresh: {config['superpoint']['keypoint_threshold']}, Match Thresh: {config['superglue']['match_threshold']}")
        except Exception as e:
            print(f"Error re-initializing Matching model: {e}")

import time
import math
try:
    from skimage.metrics import structural_similarity as compare_ssim
except ImportError:
    compare_ssim = None
    print("Warning: scikit-image not found. SSIM metric will be unavailable.")

def calculate_overlap_ssim(img_ref_gray, img_warped_gray, mask_ref=None, mask_warped=None):
    """
    محاسبه شاخص SSIM فقط در ناحیه هم‌پوشانی دو تصویر.
    F-01/F-14: به جای فرض اینکه پیکسل سیاه (0) پس‌زمینه است،
    از ماسک‌های ردپای هندسی صریح (Geometric Footprint Masks) استفاده می‌شود
    تا پیکسل‌های جذب پرتو و سیاه واقعی ایکس‌ری از تحلیل هم‌پوشانی حذف نشوند.
    """
    if compare_ssim is None or img_ref_gray is None or img_warped_gray is None:
        return 0.0

    if mask_ref is None:
        mask_ref = img_ref_gray > 0
    if mask_warped is None:
        mask_warped = img_warped_gray > 0

    intersection_mask = (mask_ref > 0) & (mask_warped > 0)
    
    # اگر هم‌پوشانی وجود نداشت
    if not np.any(intersection_mask):
        return 0.0

    # 2. پیدا کردن کادر محیطی (Bounding Box) ناحیه هم‌پوشانی
    y_idxs, x_idxs = np.where(intersection_mask)
    y_min, y_max = y_idxs.min(), y_idxs.max()
    x_min, x_max = x_idxs.min(), x_idxs.max()
    
    # 3. برش تصاویر (Crop) به ناحیه مشترک
    crop_ref = img_ref_gray[y_min:y_max+1, x_min:x_max+1]
    crop_warped = img_warped_gray[y_min:y_max+1, x_min:x_max+1]
    
    # 4. محاسبه SSIM
    win_size = min(7, min(crop_ref.shape[0], crop_ref.shape[1]))
    if win_size % 2 == 0:
        win_size -= 1
    
    if win_size < 3:
        return 0.0
        
    score, _ = compare_ssim(crop_ref, crop_warped, full=True, win_size=win_size)
    return float(score)

def compute_accuracy_metrics(src_pts, dst_pts, H, mask=None):
    """
    محاسبه شاخص‌های ارزیابی برای گزارش عملکرد هم‌پوشانی
    src_pts: نقاط در تصویر مبدا (matched points)
    dst_pts: نقاط در تصویر مقصد
    H: ماتریس هموگرافی یا ماتریس تبدیل خطی
    mask: ماسک خروجی RANSAC (مشخص کننده اینلایرها)
    """
    if src_pts is None or dst_pts is None or len(src_pts) == 0:
        return 0.0, float('nan'), 0, 0

    total_matches = len(src_pts)
    if total_matches == 0:
        return 0.0, float('nan'), 0, 0

    # بررسی تطابق اندازه ماسک با نقاط؛ در صورت عدم تطابق (مثلاً فیلتر شدن نقاط در مد 3D)،
    # اینلایرها مستقیماً با اعمال H و آستانه خطای بازپخش ۵ پیکسل محاسبه می‌شوند
    if mask is not None and len(np.ravel(mask)) == total_matches:
        matches_mask = (np.ravel(mask) == 1)
    elif H is not None and total_matches > 0:
        src_reshaped = src_pts.reshape(-1, 1, 2).astype(np.float32)
        try:
            pred_pts = cv.perspectiveTransform(src_reshaped, H.astype(np.float32)).reshape(-1, 2)
            errors = np.linalg.norm(dst_pts - pred_pts, axis=1)
            matches_mask = (errors <= 5.0)
        except Exception:
            matches_mask = np.ones(total_matches, dtype=bool)
    else:
        matches_mask = np.ones(total_matches, dtype=bool)

    num_inliers = int(np.sum(matches_mask))
    inlier_ratio = (num_inliers / total_matches) * 100.0 if total_matches > 0 else 0.0

    # 2. محاسبه خطای RMSE (فقط برای نقاط Inlier)
    if num_inliers > 0 and H is not None:
        inlier_src = src_pts[matches_mask]
        inlier_dst = dst_pts[matches_mask]
        
        inlier_src_reshaped = inlier_src.reshape(-1, 1, 2).astype(np.float32)
        inlier_dst_reshaped = inlier_dst.reshape(-1, 1, 2).astype(np.float32)
        
        try:
            pred_pts = cv.perspectiveTransform(inlier_src_reshaped, H.astype(np.float32))
            error_norm = cv.norm(inlier_dst_reshaped, pred_pts, cv.NORM_L2)
            rmse = float(error_norm / math.sqrt(num_inliers))
        except Exception:
            rmse = float('nan')
    else:
        rmse = float('nan')

    return inlier_ratio, rmse, num_inliers, total_matches

@_optional_no_grad
def stitch_with_superglue(image0_bgr, image1_bgr):
    # --- شروع تایمر (برای محاسبه Latency) ---
    start_time = time.time()

    if matching is None:
        print("Auto-Stitcher Error: Models could not be loaded.")
        return None
        
    if image0_bgr is None or image1_bgr is None:
        print("Auto-Stitcher Error: One or both input images are None.")
        return None

    if image0_bgr.shape[2] == 4:
        image0_bgr = cv.cvtColor(image0_bgr, cv.COLOR_BGRA2BGR)
    if image1_bgr.shape[2] == 4:
        image1_bgr = cv.cvtColor(image1_bgr, cv.COLOR_BGRA2BGR)

    image0 = cv.cvtColor(image0_bgr, cv.COLOR_BGR2GRAY)
    image1 = cv.cvtColor(image1_bgr, cv.COLOR_BGR2GRAY)

    frame_tensor0 = frame2tensor(image0, device)
    frame_tensor1 = frame2tensor(image1, device)

    pred = matching({'image0': frame_tensor0, 'image1': frame_tensor1})
    pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
    
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches = pred['matches0']
    
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    
    print(f"Auto-Stitcher: Found {len(mkpts0)} valid matches.")

    if len(mkpts0) < 4:
        print("Auto-Stitcher Error: Not enough matches found to compute homography.")
        return None
        
    H, mask = cv.findHomography(mkpts1, mkpts0, cv.RANSAC, 5.0)

    if H is not None:
        # --- فراخوانی تابع محاسبه شاخص‌ها ---
        # توجه: mkpts1 همان نقاط src و mkpts0 همان نقاط dst هستند (مطابق ترتیب findHomography)
        inlier_r, rmse_val, n_inliers, n_total = compute_accuracy_metrics(mkpts1, mkpts0, H, mask)
        
        # --- پایان تایمر ---
        end_time = time.time()
        process_time_ms = (end_time - start_time) * 1000  # تبدیل به میلی‌ثانیه

        # --- چاپ نتایج در کنسول (برای درج در پایان‌نامه) ---
        print("-" * 30)
        print(f"--- EVALUATION REPORT FOR FRAME PAIR ---")
        print(f"Processing Time: {process_time_ms:.2f} ms")
        print(f"Total Matches:   {n_total}")
        print(f"Inliers Count:   {n_inliers}")
        print(f"Inlier Ratio:    {inlier_r:.2f} %")
        print(f"RMSE Accuracy:   {rmse_val:.4f} pixels")
        # SSIM will be printed after warping
    else:
        print("Auto-Stitcher Error: Homography matrix could not be computed.")
        return None

    h0, w0 = image0.shape[:2]
    h1, w1 = image1.shape[:2]

    corners0 = np.float32([[0, 0], [0, h0], [w0, h0], [w0, 0]]).reshape(-1, 1, 2)
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    
    corners1_transformed = cv.perspectiveTransform(corners1, H)
    
    all_corners = np.concatenate((corners0, corners1_transformed), axis=0)
    
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    final_w, final_h = x_max - x_min, y_max - y_min
    
    Ht = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
    
    warped_image0 = cv.warpPerspective(image0_bgr, Ht, (final_w, final_h))
    warped_image1 = cv.warpPerspective(image1_bgr, Ht @ H, (final_w, final_h))

    # F-01 & F-14: استفاده از ماسک ردپای هندسی بدون وابستگی به روشنایی پیکسل‌ها
    footprint0 = np.ones((h0, w0), dtype=np.uint8) * 255
    footprint1 = np.ones((h1, w1), dtype=np.uint8) * 255
    warped_mask0 = cv.warpPerspective(footprint0, Ht, (final_w, final_h), flags=cv.INTER_NEAREST)
    warped_mask1 = cv.warpPerspective(footprint1, Ht @ H, (final_w, final_h), flags=cv.INTER_NEAREST)

    mask0 = warped_mask0 > 127
    mask1 = warped_mask1 > 127
    overlap = mask0 & mask1

    # --- محاسبه SSIM بر روی ردپای هندسی هم‌پوشانی ---
    gray_w0 = cv.cvtColor(warped_image0, cv.COLOR_BGR2GRAY)
    gray_w1 = cv.cvtColor(warped_image1, cv.COLOR_BGR2GRAY)
    ssim_score = calculate_overlap_ssim(gray_w0, gray_w1, mask0, mask1)
    print(f"SSIM Score:      {ssim_score:.4f}")
    print("-" * 30)

    final_result = np.zeros_like(warped_image0)
    only0 = mask0 & (~mask1)
    only1 = mask1 & (~mask0)
    final_result[only0] = warped_image0[only0]
    final_result[only1] = warped_image1[only1]

    if np.any(overlap):
        d0 = cv.distanceTransform(mask0.astype(np.uint8), cv.DIST_L2, 5)
        d1 = cv.distanceTransform(mask1.astype(np.uint8), cv.DIST_L2, 5)
        w0 = d0[overlap].astype(np.float32)
        w1 = d1[overlap].astype(np.float32)
        sum_w = w0 + w1
        sum_w[sum_w == 0] = 1.0
        w0 = (w0 / sum_w)[:, np.newaxis]
        w1 = (w1 / sum_w)[:, np.newaxis]
        blended = warped_image0[overlap].astype(np.float32) * w0 + warped_image1[overlap].astype(np.float32) * w1
        final_result[overlap] = np.clip(blended, 0, 255).astype(np.uint8)
    else:
        final_result[mask0] = warped_image0[mask0]

    # برش دقیق به کادر اجتماع ردپاها بدون آسیب به پیکسل‌های جذب پرتو (سیاه) ایکس‌ری
    union_mask = (mask0 | mask1).astype(np.uint8) * 255
    contours, _ = cv.findContours(union_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(largest_contour)
        final_result = final_result[y:y+h, x:x+w]

    return final_result

# <<<<< تابع جدید برای محاسبه ماتریس تبدیل (برای کلید 'g') >>>>>
@_optional_no_grad
def calculate_superglue_transform(image0_bgr, image1_bgr, roi=None, stitch_mode='auto'):
    # --- شروع تایمر ---
    start_time = time.time()

    if matching is None: return None
    if image0_bgr is None or image1_bgr is None: return None

    if image0_bgr.shape[2] == 4: image0_bgr = cv.cvtColor(image0_bgr, cv.COLOR_BGRA2BGR)
    if image1_bgr.shape[2] == 4: image1_bgr = cv.cvtColor(image1_bgr, cv.COLOR_BGRA2BGR)

    image0 = cv.cvtColor(image0_bgr, cv.COLOR_BGR2GRAY)
    image1 = cv.cvtColor(image1_bgr, cv.COLOR_BGR2GRAY)

    frame_tensor0 = frame2tensor(image0, device)
    frame_tensor1 = frame2tensor(image1, device)

    pred = matching({'image0': frame_tensor0, 'image1': frame_tensor1})
    pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
    
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches = pred['matches0']
    
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    
    print(f"Auto-Aligner: Found {len(mkpts0)} valid matches.")
    if len(mkpts0) < 4: return None

    # اقدام ۴: در صورت مشخص بودن ناحیه اولویت تراز (Priority ROI)
    if roi is not None and len(roi) == 4:
        rx, ry, rw, rh = roi
        in_roi = (mkpts0[:, 0] >= rx) & (mkpts0[:, 0] <= rx + rw) & (mkpts0[:, 1] >= ry) & (mkpts0[:, 1] <= ry + rh)
        if np.sum(in_roi) >= 4:
            print(f"Auto-Aligner: Focusing Homography strictly on Priority ROI ({np.sum(in_roi)} matches).")
            mkpts0_roi = mkpts0[in_roi]
            mkpts1_roi = mkpts1[in_roi]
            H_roi, mask_roi = cv.findHomography(mkpts1_roi, mkpts0_roi, cv.RANSAC, 3.0)
            if H_roi is not None:
                if abs(H_roi[2, 2]) > 1e-7:
                    H_roi = H_roi / H_roi[2, 2]
                return H_roi
        else:
            # F-11: گزارش صریح عقب‌نشینی از ROI به تصویر سراسری
            print(f"Auto-Aligner: Notice: Priority ROI has insufficient matches ({np.sum(in_roi)} < 4). Falling back to full image.")

    # Image offsets alone cannot distinguish planar perspective/scale from depth.
    # This is a visual alignment API: retain the robust full dominant-plane fit.
    # Calibrated metric correction is provided by the separate xray_mosaic path.
    mkpts0_fit = mkpts0
    mkpts1_fit = mkpts1

    H, mask = cv.findHomography(mkpts1_fit, mkpts0_fit, cv.RANSAC, 5.0)

    if H is not None:
        # F-04: نرمال‌سازی هموگرافی برای پایداری مولفه‌های انتقال
        if abs(H[2, 2]) > 1e-7:
            H = H / H[2, 2]

        if stitch_mode == '3d':
            print('Auto-Aligner: VISUAL_ONLY dominant-plane homography; unknown-depth structures may retain parallax.')

        # --- گزارش ارزیابی برای حالت Auto-Align (g) ---
        mask_eval = mask if (mask is not None and len(mask) == len(mkpts1)) else None
        inlier_r, rmse_val, n_inliers, n_total = compute_accuracy_metrics(mkpts1, mkpts0, H, mask_eval)
        
        # محاسبه SSIM مجازی
        h, w = image0.shape[:2]
        img_warped = cv.warpPerspective(image1, H, (w, h))
        ssim_score = calculate_overlap_ssim(image0, img_warped)

        end_time = time.time()
        process_time_ms = (end_time - start_time) * 1000

        print("-" * 30)
        print(f"--- EVALUATION REPORT (Auto-Align 'g') ---")
        print(f"Processing Time: {process_time_ms:.2f} ms")
        print(f"Total Matches:   {n_total}")
        print(f"Inliers Count:   {n_inliers}")
        print(f"Inlier Ratio:    {inlier_r:.2f} %")
        print(f"RMSE Accuracy:   {rmse_val:.4f} pixels")
        print(f"SSIM Score:      {ssim_score:.4f}")
        print("-" * 30)

    return H


# <<<<< تابع جدید برای ارزیابی میزان تطابق (برای کلید 'f') >>>>>
@_optional_no_grad
def get_match_confidence(image0_bgr, image1_bgr):
    """
    F-12: برآورد ساختاریافته شاخص‌های انطباق (تعداد انطباق، درصد اعتبار، امتیاز میانگین).
    برای سازگاری رو به عقب، مقدار بازگشتی در تبدیل به int همان تعداد انطباق‌های معتبر است.
    """
    if matching is None or image0_bgr is None or image1_bgr is None:
        return 0

    if image0_bgr.shape[2] == 4: image0_bgr = cv.cvtColor(image0_bgr, cv.COLOR_BGRA2BGR)
    if image1_bgr.shape[2] == 4: image1_bgr = cv.cvtColor(image1_bgr, cv.COLOR_BGRA2BGR)

    image0 = cv.cvtColor(image0_bgr, cv.COLOR_BGR2GRAY)
    image1 = cv.cvtColor(image1_bgr, cv.COLOR_BGR2GRAY)

    frame_tensor0 = frame2tensor(image0, device)
    frame_tensor1 = frame2tensor(image1, device)

    pred = matching({'image0': frame_tensor0, 'image1': frame_tensor1})
    matches = pred['matches0'][0]
    
    valid_matches = matches > -1
    num_valid_matches = int(np.sum(valid_matches.detach().cpu().numpy()))
    
    return num_valid_matches


def cluster_depth_disparities(offsets, threshold=3.5, return_info=False):
    """
    خوشه‌بندی بردار‌های اختلاف منظر (Depth Parallax Clustering).
    تشخیص لایه‌های مختلف عمق سه‌بعدی و پیدا کردن آفست پایدار برای صفحه مرجع.
    در قطعات دو بعدی (تک‌عمق): رفتار ۱۰۰٪ صلب استاندارد بدون هیچ تغییری حفظ می‌شود.
    در قطعات سه‌بعدی (چندعمق): لایه‌های مختلف تفکیک شده و تراز بر روی صفحه مرجع غالب قفل می‌گردد.
    """
    if len(offsets) < 4:
        dom = np.median(offsets, axis=0) if len(offsets) > 0 else np.array([0.0, 0.0], dtype=np.float32)
        p_info = {
            'is_2d_planar': True,
            'has_parallax': False,
            'dominant_offset': dom,
            'std': np.array([0.0, 0.0], dtype=np.float32),
            'peaks': [float(dom[0])],
            'disparity_spread': 0.0,
            'num_clusters': 1,
            'dominant_ratio': 1.0
        }
        return (dom, False, p_info) if return_info else (dom, False)

    offsets = np.asarray(offsets, dtype=np.float32)
    std_disp = np.std(offsets, axis=0)
    norm_std = float(np.linalg.norm(std_disp))

    # تشخیص محور اصلی حرکت (معمولاً محور X یا محوری با واریانس/میانگین بالاتر)
    med_all = np.median(offsets, axis=0)
    primary_axis = 0 if abs(med_all[0]) >= abs(med_all[1]) or std_disp[0] >= std_disp[1] else 1

    axis_vals = offsets[:, primary_axis]

    # تحلیل چندقله‌ای ۱ بعدی با هیستوگرام ریزدانه‌ای (Bin Width = 1.5px)
    val_min, val_max = float(np.min(axis_vals)), float(np.max(axis_vals))
    val_range = val_max - val_min

    peaks = []
    has_parallax = False
    is_2d_planar = True

    if val_range > threshold and len(offsets) >= 6:
        bin_width = 1.5
        num_bins = max(5, int(math.ceil(val_range / bin_width)))
        hist, bin_edges = np.histogram(axis_vals, bins=num_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        min_peak_pts = max(3, int(len(offsets) * 0.12))  # حداقل ۱۲٪ نقاط برای تشکیل لایه عمقی
        candidate_peaks = []
        for i in range(len(hist)):
            left_ok = (i == 0) or (hist[i] >= hist[i - 1])
            right_ok = (i == len(hist) - 1) or (hist[i] >= hist[i + 1])
            if left_ok and right_ok and hist[i] >= min_peak_pts:
                candidate_peaks.append((bin_centers[i], hist[i]))

        # ادغام قله‌های نزدیک به هم (کمتر از آستانه)
        filtered_peaks = []
        for center, count in sorted(candidate_peaks, key=lambda x: x[1], reverse=True):
            if not any(abs(center - p[0]) < threshold for p in filtered_peaks):
                filtered_peaks.append((center, count))

        if len(filtered_peaks) > 1:
            has_parallax = True
            is_2d_planar = False
            peaks = [float(p[0]) for p in filtered_peaks]
        elif norm_std > threshold:
            has_parallax = True
            is_2d_planar = False
            peaks = [float(med_all[primary_axis])]
        else:
            peaks = [float(med_all[primary_axis])]
    else:
        peaks = [float(med_all[primary_axis])]

    # انتخاب خوشه غالب (Dominant Reference Plane Cluster)
    if has_parallax and len(peaks) > 1:
        dom_peak_val = peaks[0]
        in_dom_mask = np.abs(axis_vals - dom_peak_val) <= (threshold * 1.2)
        if np.sum(in_dom_mask) >= 3:
            dominant_offset = np.median(offsets[in_dom_mask], axis=0)
            dom_count = int(np.sum(in_dom_mask))
        else:
            dominant_offset = med_all
            dom_count = len(offsets)
        disparity_spread = float(max(peaks) - min(peaks))
    else:
        # حالت تک‌صفحه‌ای دوبعدی (Planar 2D Mode)
        dists = np.linalg.norm(offsets - med_all, axis=1)
        in_dom_mask = dists < (threshold * 1.5)
        if np.sum(in_dom_mask) >= 3:
            dominant_offset = np.median(offsets[in_dom_mask], axis=0)
            dom_count = int(np.sum(in_dom_mask))
        else:
            dominant_offset = med_all
            dom_count = len(offsets)
        disparity_spread = 0.0

    dominant_offset = np.array([float(dominant_offset[0]), float(dominant_offset[1])], dtype=np.float32)

    p_info = {
        'is_2d_planar': is_2d_planar,
        'has_parallax': has_parallax,
        'dominant_offset': dominant_offset,
        'std': std_disp,
        'peaks': peaks,
        'disparity_spread': disparity_spread,
        'num_clusters': len(peaks),
        'dominant_ratio': float(dom_count / max(1, len(offsets)))
    }

    if return_info:
        return dominant_offset, has_parallax, p_info
    return dominant_offset, has_parallax


@_optional_no_grad
def calculate_translation_offset(image0_bgr, image1_bgr, roi=None, stitch_mode='auto'):
    """
    محاسبه فاصله خطی (X, Y) بین دو تصویر با تفکیک هوشمند عمق (Depth-Aware).
    در صورت تعریف ROI، تراز با اولویت ۱۰۰٪ روی همان عمق متمرکز می‌شود.
    در حالت دوبعدی، عملکرد صلب پایدار دقیقاً مطابق قبل است.
    """
    start_time = time.time()

    if matching is None or image0_bgr is None or image1_bgr is None:
        return None

    if image0_bgr.shape[2] == 4: image0_bgr = cv.cvtColor(image0_bgr, cv.COLOR_BGRA2BGR)
    if image1_bgr.shape[2] == 4: image1_bgr = cv.cvtColor(image1_bgr, cv.COLOR_BGRA2BGR)

    image0 = cv.cvtColor(image0_bgr, cv.COLOR_BGR2GRAY)
    image1 = cv.cvtColor(image1_bgr, cv.COLOR_BGR2GRAY)

    frame_tensor0 = frame2tensor(image0, device)
    frame_tensor1 = frame2tensor(image1, device)

    pred = matching({'image0': frame_tensor0, 'image1': frame_tensor1})

    kpts0, kpts1 = pred['keypoints0'][0].detach().cpu().numpy(), pred['keypoints1'][0].detach().cpu().numpy()
    matches = pred['matches0'][0].detach().cpu().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    if len(mkpts0) < 4:
        print("Auto-Snap: Not enough matches found.")
        return None

    offsets = mkpts0 - mkpts1

    # اقدام ۴: در صورت وجود ROI اولویت تراز
    used_roi = False
    if roi is not None and len(roi) == 4:
        rx, ry, rw, rh = roi
        in_roi = (mkpts0[:, 0] >= rx) & (mkpts0[:, 0] <= rx + rw) & (mkpts0[:, 1] >= ry) & (mkpts0[:, 1] <= ry + rh)
        if np.sum(in_roi) >= 3:
            dx_res = float(np.median(offsets[in_roi, 0]))
            dy_res = float(np.median(offsets[in_roi, 1]))
            used_roi = True
            print(f"Auto-Snap: Locked alignment to Priority ROI depth ({np.sum(in_roi)} matches) -> dx: {dx_res:.2f}, dy: {dy_res:.2f}")
        else:
            # F-11: گزارش صریح عقب‌نشینی به الگوی فراگیر
            print(f"Auto-Snap: Notice: Priority ROI had insufficient matches ({np.sum(in_roi)} < 3); falling back to dominant plane.")

    if not used_roi:
        dom_offset, has_parallax, p_info = cluster_depth_disparities(offsets, return_info=True)
        dx_res, dy_res = float(dom_offset[0]), float(dom_offset[1])
        
        is_3d = (stitch_mode == '3d') or (stitch_mode == 'auto' and has_parallax)
        if is_3d and not (stitch_mode == '2d'):
            print(f"Auto-Snap: [3D Cone-Beam Mode] Multi-depth parallax detected ({p_info['num_clusters']} layers, spread: {p_info['disparity_spread']:.1f}px). Locked to Reference Plane -> dx: {dx_res:.2f}, dy: {dy_res:.2f}")
        else:
            print(f"Auto-Snap: [2D Planar Mode] Single depth plane confirmed (std: {float(np.linalg.norm(p_info['std'])):.2f}px). Offset -> dx: {dx_res:.2f}, dy: {dy_res:.2f} (Matches: {len(mkpts0)})")

    # گزارش ارزیابی
    dist_errors = np.linalg.norm(offsets - np.array([dx_res, dy_res]), axis=1)
    custom_mask = (dist_errors < 5.0).astype(np.uint8).reshape(-1, 1)

    H_translation = np.array([
        [1, 0, dx_res],
        [0, 1, dy_res],
        [0, 0, 1]
    ], dtype=np.float32)

    inlier_r, rmse_val, n_inliers, n_total = compute_accuracy_metrics(mkpts1, mkpts0, H_translation, custom_mask)

    h, w = image0.shape[:2]
    img1_translated = cv.warpPerspective(image1, H_translation, (w, h))
    ssim_score = calculate_overlap_ssim(image0, img1_translated)

    end_time = time.time()
    process_time_ms = (end_time - start_time) * 1000

    print("-" * 30)
    print(f"--- EVALUATION REPORT (Smart-Snap 'n') ---")
    print(f"Processing Time: {process_time_ms:.2f} ms")
    print(f"Total Matches:   {n_total}")
    print(f"Inliers Count:   {n_inliers}")
    print(f"Inlier Ratio:    {inlier_r:.2f} %")
    print(f"RMSE Accuracy:   {rmse_val:.4f} pixels")
    print(f"SSIM Score:      {ssim_score:.4f}")
    print("-" * 30)

    return dx_res, dy_res


@_optional_no_grad
def calculate_elastic_mesh_warp(image0_bgr, image1_bgr, grid_cells=(16, 16), roi=None):
    """
    اقدام ۲: کشش منعطف محلی (Local Mesh Warping / APAP).
    حل قطعی مشکل ناهمگونی دو طرف کادر ناشی از پرتوهای مایل مخروطی اشعه ایکس.
    به جای یک ماتریس صلب واحد، تصویر با توری محلی تغییر شکل داده و هم‌زمان در هر دو سمت تراز می‌شود.
    """
    if matching is None or image0_bgr is None or image1_bgr is None:
        return None, None

    img0_src = image0_bgr[:, :, :3] if image0_bgr.shape[2] >= 3 else image0_bgr
    img1_src = image1_bgr[:, :, :3] if image1_bgr.shape[2] >= 3 else image1_bgr

    image0_gray = cv.cvtColor(img0_src, cv.COLOR_BGR2GRAY)
    image1_gray = cv.cvtColor(img1_src, cv.COLOR_BGR2GRAY)

    frame_tensor0 = frame2tensor(image0_gray, device)
    frame_tensor1 = frame2tensor(image1_gray, device)

    pred = matching({'image0': frame_tensor0, 'image1': frame_tensor1})
    kpts0 = pred['keypoints0'][0].detach().cpu().numpy()
    kpts1 = pred['keypoints1'][0].detach().cpu().numpy()
    matches = pred['matches0'][0].detach().cpu().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    if len(mkpts0) < 6:
        print("Elastic Warp: Insufficient match points.")
        return None, None

    h1, w1 = image1_bgr.shape[:2]
    gw, gh = grid_cells
    xs = np.linspace(0, w1, gw, dtype=np.float32)
    ys = np.linspace(0, h1, gh, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)

    # محاسبه جابه‌جایی‌های محلی با Moving Least Squares (MLS)
    offsets = mkpts0 - mkpts1
    global_dx = np.median(offsets[:, 0])
    global_dy = np.median(offsets[:, 1])

    # اقدام ۴: در صورت وجود محدوده اولویت، نقاط درون ROI وزن و نفوذ ۵ برابر می‌گیرند
    pt_importance = np.ones(len(mkpts0), dtype=np.float32)
    if roi is not None and len(roi) == 4:
        rx, ry, rw, rh = roi
        in_roi = (mkpts0[:, 0] >= rx) & (mkpts0[:, 0] <= rx + rw) & (mkpts0[:, 1] >= ry) & (mkpts0[:, 1] <= ry + rh)
        if np.any(in_roi):
            pt_importance[in_roi] = 5.0
            print(f"Elastic Warp: Prioritizing {np.sum(in_roi)} matches inside Priority ROI.")

    disp_x = np.full_like(gx, global_dx)
    disp_y = np.full_like(gy, global_dy)

    sigma = max(w1, h1) * 0.15  # کاهش شعاع تأثیر برای جلوگیری از سرایت به مرکز
    gamma = 2.0

    # فاصله هر نقطه توری تا مرکز تایل جهت حفاظت ۱۰۰٪ از هسته داخلی
    cx, cy = w1 / 2.0, h1 / 2.0
    max_r = math.hypot(cx, cy)

    for i in range(gh):
        for j in range(gw):
            px, py = gx[i, j], gy[i, j]
            sq_dists = (mkpts1[:, 0] - px) ** 2 + (mkpts1[:, 1] - py) ** 2
            min_dist_to_overlap = math.sqrt(float(np.min(sq_dists)))

            # میرایی نمایی: اگر گره توری از ناحیه هم‌پوشانی دور باشد (در دل موزاییک)،
            # تغییر شکل محلی به صفر میل می‌کند و فقط جابه‌جایی صلب (global_dx, global_dy) حفظ می‌شود
            overlap_decay = math.exp(-(min_dist_to_overlap ** 2) / (2.0 * (sigma ** 2)))

            # محافظت از هسته مرکزی تایل (شعاع ۳۵٪ مرکزی کاملاً بدون دفرماسیون)
            r_norm = math.hypot(px - cx, py - cy) / max_r
            center_protection = float(np.clip((r_norm - 0.35) / 0.45, 0.0, 1.0))

            damping = overlap_decay * center_protection

            weights = (1.0 / ((sq_dists / (sigma ** 2)) ** gamma + 1e-4)) * pt_importance
            sum_w = np.sum(weights)
            if sum_w > 0:
                local_ox = float(np.sum(weights * offsets[:, 0]) / sum_w)
                local_oy = float(np.sum(weights * offsets[:, 1]) / sum_w)
                disp_x[i, j] = global_dx + damping * (local_ox - global_dx)
                disp_y[i, j] = global_dy + damping * (local_oy - global_dy)

    # ایجاد نقشه متراکم بازنگاشت (Dense Remap Field)
    dense_dx = cv.resize(disp_x, (w1, h1), interpolation=cv.INTER_CUBIC)
    dense_dy = cv.resize(disp_y, (w1, h1), interpolation=cv.INTER_CUBIC)

    grid_x, grid_y = np.meshgrid(np.arange(w1, dtype=np.float32), np.arange(h1, dtype=np.float32))
    # منبع پیکسل‌ها در تصویر اصلی ۱
    map_x = grid_x - (dense_dx - global_dx)
    map_y = grid_y - (dense_dy - global_dy)

    warped_img1 = cv.remap(image1_bgr, map_x, map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    print(f"Elastic Mesh Warp: Successfully adapted 3D parallax field using {len(mkpts0)} guide points.")
    return warped_img1, (float(global_dx), float(global_dy))


# ==============================================================================
# موتورهای محاسباتی یدکی و پیشرفته (Alternative / Neural Backup Engines)
# ==============================================================================

from neural_stitcher_engine import (
    dense_matcher_instance,
    deformable_engine_instance,
    parallax_engine_instance,
    global_graph_instance
)

def calculate_dense_feature_offset(image0_bgr, image1_bgr, roi=None):
    """
    روش یدکی ۱: محاسبه آفست با تطبیق متراکم پیکسلی (Dense Feature Matching)
    مناسب برای تصاویری که نقاط گوشه‌ای با کنتراست شدید ندارند و لایه‌های ظریف داخلی مهم است.
    """
    return dense_matcher_instance.calculate_dense_translation(image0_bgr, image1_bgr, roi=roi)

def calculate_dense_feature_homography(image0_bgr, image1_bgr, roi=None):
    """
    روش یدکی ۲: محاسبه ماتریس هموگرافی با برازش متراکم جریان محلی و MAGSAC++
    """
    return dense_matcher_instance.calculate_dense_homography(image0_bgr, image1_bgr, roi=roi)

def calculate_deformable_neural_warp(image0_bgr, image1_bgr, grid_cells=(24, 24), roi=None):
    """
    روش یدکی ۳: دفرماسیون غیرصلب دیفئومورفیک (Diffeomorphic Deformable Registration)
    انطباق لایه‌های داخلی بدون پاره شدن یا چین‌خوردگی ترک‌های مدار چاپی (با قید ژاکوبین مثبت).
    """
    return deformable_engine_instance.calculate_deformable_warp(image0_bgr, image1_bgr, grid_cells=grid_cells, roi=roi)

def calculate_parallax_mesh_warp(image0_bgr, image1_bgr, roi=None):
    """
    روش یدکی ۴: دوخت با تحمل اختلاف منظر (Parallax-Tolerant Mesh Warp - UDIS2)
    """
    return parallax_engine_instance.calculate_mesh_warp(image0_bgr, image1_bgr, roi=roi)

def optimize_global_grid_offsets(grid_cells, pairwise_offsets, anchor_gid=None):
    """
    روش یدکی ۵: بهینه‌ساز سراسری گراف برای تراز هم‌زمان کل تایل‌های ماتریس (بدون رانش تجمعی)
    """
    return global_graph_instance.solve_grid_translations(grid_cells, pairwise_offsets, anchor_gid=anchor_gid)

# Dispatcher functions (انتخاب هوشمند بین روش اصلی و روش‌های یدکی)
def calculate_smart_snap_offset(image0_bgr, image1_bgr, engine='superglue', roi=None, stitch_mode='auto'):
    if engine == 'dense':
        res = calculate_dense_feature_offset(image0_bgr, image1_bgr, roi=roi)
        if res is not None:
            return res
        print("Dense Matcher fallback to SuperGlue...")
    return calculate_translation_offset(image0_bgr, image1_bgr, roi=roi, stitch_mode=stitch_mode)

def calculate_auto_align_transform(image0_bgr, image1_bgr, engine='superglue', roi=None, stitch_mode='auto'):
    if engine == 'sift' or (engine == 'superglue' and matching is None):
        from visual_mosaic import estimate_pair
        return estimate_pair(image0_bgr, image1_bgr)[0]
    if engine == 'dense':
        res = calculate_dense_feature_homography(image0_bgr, image1_bgr, roi=roi)
        if res is not None:
            return res
        print("Dense Homography fallback to SuperGlue...")
    return calculate_superglue_transform(image0_bgr, image1_bgr, roi=roi, stitch_mode=stitch_mode)

def calculate_warp_deformation(image0_bgr, image1_bgr, engine='elastic_mls', roi=None):
    if engine == 'deformable_dir':
        res = calculate_deformable_neural_warp(image0_bgr, image1_bgr, roi=roi)
        if res and res[0] is not None:
            return res
        print("Deformable DIR fallback to MLS Elastic Warp...")
    elif engine == 'parallax_udis':
        res = calculate_parallax_mesh_warp(image0_bgr, image1_bgr, roi=roi)
        if res and res[0] is not None:
            return res
        print("Parallax UDIS fallback to MLS Elastic Warp...")
    return calculate_elastic_mesh_warp(image0_bgr, image1_bgr, roi=roi)

