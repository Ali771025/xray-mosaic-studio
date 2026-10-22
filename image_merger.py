# image_merger.py
import numpy as np
import cv2 as cv

def compute_center_weighted_mask(mask, max_feather_dist=None):
    """
    محاسبه نقشه فاصله اقلیدسی از لبه‌های کادر برای وزن‌دهی به مرکز تصویر.
    در تصاویر ایکس‌ری، مرکز تصویر عمودترین تابش پرتو و کمترین اعوجاج مخروطی را دارد.
    """
    if not np.any(mask):
        return np.zeros_like(mask, dtype=np.float32)
    
    # محاسبه فاصله از نزدیک‌ترین حاشیه بیرونی (Distance Transform)
    dist = cv.distanceTransform(mask.astype(np.uint8), cv.DIST_L2, 5)
    
    if max_feather_dist is not None and max_feather_dist > 0:
        dist = np.clip(dist / float(max_feather_dist), 0.0, 1.0)
    else:
        max_val = np.max(dist)
        if max_val > 0:
            dist = dist / max_val
            
    return dist.astype(np.float32)

def find_optimal_seamline_mask(crop_base, crop_top, crop_mask, d_base, d_top):
    """
    برش بهینه خط درز (Optimal Seamline Cut) بر پایه برنامه‌نویسی پویا (Dynamic Programming).
    مسیر برش از میان مناطقی با حداقل اختلاف و کمترین لبه‌ها عبور می‌کند تا اجزای سه‌بعدی
    مانند پیچ و سیم‌ها به هیچ وجه دوبل و شبح‌وار نشوند.
    """
    oh, ow = crop_mask.shape
    if oh < 4 or ow < 4:
        return (d_top >= d_base).astype(np.float32)

    # 1. نقشه انرژی اختلاف شدت نور
    diff = np.mean(np.abs(crop_top[:, :, :3].astype(np.float32) - crop_base[:, :, :3].astype(np.float32)), axis=2)

    # 2. جریمه لبه‌ها (گرادیان سوبل): خط برش ترجیح می‌دهد از میان اجسام متراکم عبور نکند
    gray_b = cv.cvtColor(crop_base[:, :, :3], cv.COLOR_BGR2GRAY).astype(np.float32)
    gray_t = cv.cvtColor(crop_top[:, :, :3], cv.COLOR_BGR2GRAY).astype(np.float32)
    grad_b = np.abs(cv.Sobel(gray_b, cv.CV_32F, 1, 0)) + np.abs(cv.Sobel(gray_b, cv.CV_32F, 0, 1))
    grad_t = np.abs(cv.Sobel(gray_t, cv.CV_32F, 1, 0)) + np.abs(cv.Sobel(gray_t, cv.CV_32F, 0, 1))
    edge_penalty = (grad_b + grad_t) * 0.4

    # 3. اولویت فاصله مرکزی (Voronoi Prior): تمایل به باقی ماندن نزدیک مرز متعادل
    d_top_f = d_top.astype(np.float32)
    d_base_f = d_base.astype(np.float32)
    max_d = max(float(np.max(d_top_f)), float(np.max(d_base_f)), 1.0)
    if max_d > 1.0:
        d_top_f = d_top_f / max_d
        d_base_f = d_base_f / max_d
    dist_diff = np.abs(d_top_f - d_base_f) * 20.0

    energy = np.nan_to_num(diff + edge_penalty + dist_diff, nan=1e4, posinf=1e5, neginf=0.0).astype(np.float32)
    # خارج از ناحیه هم‌پوشانی هزینه بالایی می‌گیرد
    energy[~crop_mask] = 1e5

    seam_top_mask = np.zeros((oh, ow), dtype=np.bool_)

    if ow >= oh:
        # برش افقی بهینه‌سازی‌شده برداری با شتاب C-level NumPy (بیش از ۱۰۰ برابر سریع‌تر)
        dp = np.copy(energy)
        trace = np.zeros((oh, ow), dtype=np.int32)
        for c in range(1, ow):
            prev = dp[:, c - 1]
            prev_up = np.pad(prev[:-1], (1, 0), constant_values=1e5)
            prev_mid = prev
            prev_dn = np.pad(prev[1:], (0, 1), constant_values=1e5)
            stacked = np.stack([prev_up, prev_mid, prev_dn], axis=0)
            best_shift = np.argmin(stacked, axis=0) - 1
            dp[:, c] += np.min(stacked, axis=0)
            trace[:, c] = np.clip(np.arange(oh) + best_shift, 0, oh - 1)

        # ردیابی به عقب (Backtracking)
        seam_r = np.zeros(ow, dtype=np.int32)
        seam_r[-1] = int(np.argmin(dp[:, -1]))
        for c in range(ow - 2, -1, -1):
            seam_r[c] = trace[seam_r[c + 1], c + 1]

        # تعیین اینکه کدام طرف خط درز متعلق به تصویر بالایی (top) است
        mean_top_d = np.mean(d_top[:oh // 2, :])
        mean_base_d = np.mean(d_base[:oh // 2, :])
        top_is_above = mean_top_d > mean_base_d

        row_indices = np.arange(oh)[:, np.newaxis]
        if top_is_above:
            seam_top_mask = row_indices < seam_r[np.newaxis, :]
        else:
            seam_top_mask = row_indices >= seam_r[np.newaxis, :]
    else:
        # برش عمودی بهینه‌سازی‌شده برداری با شتاب C-level NumPy (بیش از ۱۰۰ برابر سریع‌تر)
        dp = np.copy(energy)
        trace = np.zeros((oh, ow), dtype=np.int32)
        for r in range(1, oh):
            prev = dp[r - 1, :]
            prev_left = np.pad(prev[:-1], (1, 0), constant_values=1e5)
            prev_mid = prev
            prev_right = np.pad(prev[1:], (0, 1), constant_values=1e5)
            stacked = np.stack([prev_left, prev_mid, prev_right], axis=0)
            best_shift = np.argmin(stacked, axis=0) - 1
            dp[r, :] += np.min(stacked, axis=0)
            trace[r, :] = np.clip(np.arange(ow) + best_shift, 0, ow - 1)

        # ردیابی به عقب
        seam_c = np.zeros(oh, dtype=np.int32)
        seam_c[-1] = int(np.argmin(dp[-1, :]))
        for r in range(oh - 2, -1, -1):
            seam_c[r] = trace[r + 1, seam_c[r + 1]]

        mean_top_d = np.mean(d_top[:, :ow // 2])
        mean_base_d = np.mean(d_base[:, :ow // 2])
        top_is_left = mean_top_d > mean_base_d

        col_indices = np.arange(ow)[np.newaxis, :]
        if top_is_left:
            seam_top_mask = col_indices < seam_c[:, np.newaxis]
        else:
            seam_top_mask = col_indices >= seam_c[:, np.newaxis]

    # اعمال میکرومحوشدگی درز (تنها ۱ تا ۲ پیکسل نرم برای جلوگیری از پله‌پله شدن بدون دوبل شدن محتوا)
    weight_map = seam_top_mask.astype(np.float32)
    weight_map = cv.GaussianBlur(weight_map, (3, 3), 0.8)
    return weight_map


def merge_images_with_offset(base_img, top_img, top_img_pos_relative, feather_power=2.0, enable_feathering=True, edge_brightness=0.0, blend_mode='seamline', precomputed_dist_base=None, precomputed_dist_top=None):
    """
    ترکیب هوشمند دو تصویر با خنثی‌سازی اعوجاج پرسپکتیو و سایه‌های لبه ایکس‌ری.
    پشتیبانی از دو حالت:
    1. 'seamline': برش بهینه خط درز بدون شبح‌وارگی و بدون دوبل شدن پیچ/سیم‌ها (پیش‌فرض).
    2. 'feather': ترکیب تدریجی وزن‌دار با نرمی فاصله مرکزی.
    """
    if base_img.shape[2] == 3:
        base_img = cv.cvtColor(base_img, cv.COLOR_BGR2BGRA)
    if top_img.shape[2] == 3:
        top_img = cv.cvtColor(top_img, cv.COLOR_BGR2BGRA)

    h_base, w_base = base_img.shape[:2]
    h_top, w_top = top_img.shape[:2]
    x_top_rel, y_top_rel = top_img_pos_relative

    min_x = min(0, x_top_rel)
    min_y = min(0, y_top_rel)
    max_x = max(w_base, x_top_rel + w_top)
    max_y = max(h_base, y_top_rel + h_top)

    final_w, final_h = int(max_x - min_x), int(max_y - min_y)
    final_canvas = np.zeros((final_h, final_w, 4), dtype=np.uint8)

    base_pos_new = (int(-min_x), int(-min_y))
    top_pos_new = (int(x_top_rel - min_x), int(y_top_rel - min_y))

    # قرار دادن تصویر پایه در بوم
    bx, by = base_pos_new
    final_canvas[by:by+h_base, bx:bx+w_base] = base_img

    tx, ty = top_pos_new
    roi = final_canvas[ty:ty+h_top, tx:tx+w_top]

    mask_top = top_img[:, :, 3] > 0
    mask_base = roi[:, :, 3] > 0
    overlap_region = mask_top & mask_base
    only_top_region = mask_top & (~mask_base)

    if not enable_feathering or not np.any(overlap_region):
        # F-17: Standard Porter-Duff Over for straight alpha to prevent dark edge fringes
        top_pixels = top_img[mask_top]
        roi_base_part = roi[mask_top]
        a_top = top_pixels[:, 3].astype(np.float32) / 255.0
        a_base = roi_base_part[:, 3].astype(np.float32) / 255.0
        a_out = a_top + a_base * (1.0 - a_top)
        a_out_safe = np.where(a_out > 0, a_out, 1.0)[:, np.newaxis]
        rgb_top = top_pixels[:, :3].astype(np.float32)
        rgb_base = roi_base_part[:, :3].astype(np.float32)
        blended_rgb = (rgb_top * a_top[:, np.newaxis] + rgb_base * a_base[:, np.newaxis] * (1.0 - a_top[:, np.newaxis])) / a_out_safe
        roi[mask_top, :3] = np.clip(blended_rgb, 0, 255).astype(np.uint8)
        roi[mask_top, 3] = np.clip(a_out * 255.0, 0, 255).astype(np.uint8)
    else:
        dist_top = precomputed_dist_top if precomputed_dist_top is not None else compute_center_weighted_mask(mask_top)
        dist_base = precomputed_dist_base if precomputed_dist_base is not None else compute_center_weighted_mask(mask_base)

        if blend_mode == 'multiband':
            # اقدام ۵: ترکیب چندبانده هرمی لاپلاسین (حفظ حداکثر تیزی خطوط مدار و ترک‌های مس)
            from neural_stitcher_engine import merge_images_multiband_pyramid
            return merge_images_multiband_pyramid(base_img, top_img, top_img_pos_relative)
        elif blend_mode == 'seamline':
            # اقدام ۱: برش بهینه خط درز (Optimal Seamline Cut با برنامه‌نویسی پویا)
            y_idxs, x_idxs = np.where(overlap_region)
            ymin, ymax = y_idxs.min(), y_idxs.max()
            xmin, xmax = x_idxs.min(), x_idxs.max()

            crop_base = roi[ymin:ymax+1, xmin:xmax+1]
            crop_top = top_img[ymin:ymax+1, xmin:xmax+1]
            crop_mask = overlap_region[ymin:ymax+1, xmin:xmax+1]
            d_b = dist_base[ymin:ymax+1, xmin:xmax+1]
            d_t = dist_top[ymin:ymax+1, xmin:xmax+1]

            crop_weight_top = find_optimal_seamline_mask(crop_base, crop_top, crop_mask, d_b, d_t)
            weight_top = np.zeros_like(dist_top, dtype=np.float32)
            weight_top[ymin:ymax+1, xmin:xmax+1] = crop_weight_top
            weight_base = 1.0 - weight_top
        else:
            # حالت قبلی: سیگموئید و محوشدگی فاصله مرکزی
            p = max(1.0, float(feather_power))
            diff = dist_top - dist_base
            weight_top = 1.0 / (1.0 + np.exp(np.clip(-p * 15.0 * diff, -30.0, 30.0)))
            weight_base = 1.0 - weight_top

        # 1. ترکیب بخش هم‌پوشانی
        if np.any(overlap_region):
            w_top_3d = weight_top[overlap_region, np.newaxis]
            w_base_3d = weight_base[overlap_region, np.newaxis]

            top_rgb = top_img[overlap_region, :3].astype(np.float32)
            base_rgb = roi[overlap_region, :3].astype(np.float32)

            if edge_brightness != 0.0:
                edge_factor = (1.0 - dist_top[overlap_region, np.newaxis])
                top_rgb = np.clip(top_rgb + edge_brightness * edge_factor, 0.0, 255.0)

            blended_overlap = (top_rgb * w_top_3d + base_rgb * w_base_3d)
            roi[overlap_region, :3] = np.clip(blended_overlap, 0, 255).astype(np.uint8)
            roi[overlap_region, 3] = np.maximum(top_img[overlap_region, 3], roi[overlap_region, 3])

        # 2. قرار دادن بخش اختصاصی تصویر رویی (F-17: انتقال مستقیم بدون تیره شدن ضرب در آلفا)
        if np.any(only_top_region):
            roi[only_top_region] = top_img[only_top_region]

    new_origin_offset = (min_x, min_y)
    return final_canvas, new_origin_offset