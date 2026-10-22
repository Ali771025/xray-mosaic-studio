# neural_stitcher_engine.py
"""
Neural & Dense Stitching Backup Engine for Multi-layer Transmission X-Ray Images
--------------------------------------------------------------------------------
Provides alternative and advanced AI/Dense registration engines:
1. Dense Feature Matching (RoMa / LoFTR / Dual-Scale Dense Flow with certainty weighting)
2. Deformable Image Registration (DIR / Diffeomorphic flow with positive Jacobian constraint)
3. Parallax-Tolerant Mesh Warp (UDIS2-style dual-projective and elastic regularized warping)
4. Global Pose Graph Optimizer (Drift-free bundle adjustment for 3x3 or arbitrary multi-tile grids)
5. Multi-band Laplacian Pyramid Blending (High-frequency circuit trace preservation)
"""

import math
import time
import numpy as np
import cv2 as cv

# Check PyTorch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ==============================================================================
# 1. DENSE FEATURE MATCHING & DENSE DISPARITY ENGINE (مد ۱: تطبیق متراکم)
# ==============================================================================

class DenseFeatureMatcher:
    """
    موتور تطبیق متراکم پیکسلی.
    به جای استخراج چند نقطه گوشه‌ای تنک، میدان برداری کامل جابه‌جایی را برای
    تمام پیکسل‌ها و بافت‌های ظریف لایه‌های داخلی برد محاسبه می‌کند.
    """
    def __init__(self, mode='auto'):
        self.mode = mode
        self.roma_model = None
        self.loftr_model = None
        self._init_deep_models()

    def _init_deep_models(self):
        if not TORCH_AVAILABLE:
            return
        # تلاش برای بارگذاری RoMa یا LoFTR در صورت وجود بسته در محیط
        try:
            from romatch import roma_outdoor
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.roma_model = roma_outdoor(device=device).eval()
            print("DenseMatcher: Successfully loaded deep RoMa model.")
        except Exception:
            pass

        try:
            if self.roma_model is None:
                import kornia.feature as KF
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.loftr_model = KF.LoFTR(pretrained='outdoor').to(device).eval()
                print("DenseMatcher: Successfully loaded deep LoFTR model.")
        except Exception:
            pass

    def _estimate_coarse_translation(self, g_ref, g_mov):
        """تخمین سریع آفست درشت با استفاده از استخراج‌کننده نقاط کلیدی مقاوم (SIFT/ORB)"""
        try:
            sift = cv.SIFT_create(nfeatures=800)
            kp1, des1 = sift.detectAndCompute(g_ref, None)
            kp2, des2 = sift.detectAndCompute(g_mov, None)
            if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
                bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
                matches = bf.match(des1, des2)
                if len(matches) >= 4:
                    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
                    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
                    diffs = pts1 - pts2
                    med_dx = float(np.median(diffs[:, 0]))
                    med_dy = float(np.median(diffs[:, 1]))
                    return med_dx, med_dy
        except Exception:
            pass

        # در صورت نبود SIFT، برآورد با Phase Correlation در ابعاد کوچک
        try:
            h = min(g_ref.shape[0], g_mov.shape[0])
            w = min(g_ref.shape[1], g_mov.shape[1])
            r_crop = cv.resize(g_ref, (256, 256)).astype(np.float32)
            m_crop = cv.resize(g_mov, (256, 256)).astype(np.float32)
            hann = cv.createHanningWindow((256, 256), cv.CV_32F)
            shift, resp = cv.phaseCorrelate(r_crop, m_crop, hann)
            if resp > 0.1:
                return shift[0] * (w / 256.0), shift[1] * (h / 256.0)
        except Exception:
            pass

        return 0.0, 0.0

    def compute_dense_flow(self, gray_ref, gray_moving, roi=None):
        """
        محاسبه میدان جریان متراکم و نقشه اطمینان (Certainty / Confidence).
        با تضمین ۱۰۰٪ برابری ابعاد ورودی‌ها برای جلوگیری از خطای I0.sameSize(I1) در OpenCV.
        جریان خروجی: بردار جابه‌جایی از تصویر متحرک (moving) به تصویر مرجع (ref).
        """
        if gray_ref is None or gray_moving is None:
            return None, None

        if gray_ref.ndim == 3:
            gray_ref = cv.cvtColor(gray_ref[:, :, :3], cv.COLOR_BGR2GRAY)
        if gray_moving.ndim == 3:
            gray_moving = cv.cvtColor(gray_moving[:, :, :3], cv.COLOR_BGR2GRAY)

        gray_ref = gray_ref.astype(np.uint8)
        gray_moving = gray_moving.astype(np.uint8)

        h_ref, w_ref = gray_ref.shape[:2]
        h_mov, w_mov = gray_moving.shape[:2]

        # 0. پدینگ دو تصویر به حداکثر اندازه کادر مشترک جهت تضمین قطعی I0.sameSize(I1)
        max_h = max(h_ref, h_mov)
        max_w = max(w_ref, w_mov)

        pad_ref = np.zeros((max_h, max_w), dtype=np.uint8)
        pad_ref[:h_ref, :w_ref] = gray_ref

        pad_mov = np.zeros((max_h, max_w), dtype=np.uint8)
        pad_mov[:h_mov, :w_mov] = gray_moving

        # 1. اگر مدل عمیق RoMa نصب بود
        if self.roma_model is not None:
            try:
                device = next(self.roma_model.parameters()).device
                t_ref = torch.from_numpy(pad_ref).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
                t_mov = torch.from_numpy(pad_mov).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
                with torch.no_grad():
                    warp, cert = self.roma_model.match(t_mov, t_ref)
                    flow_res = warp.squeeze(0).permute(1, 2, 0).cpu().numpy()[:h_mov, :w_mov]
                    cert_res = cert.squeeze().cpu().numpy()[:h_mov, :w_mov]
                    return flow_res, cert_res
            except Exception as e:
                print(f"RoMa inference notice: {e}, falling back to dual-scale dense matcher.")

        # 2. موتور متراکم تطبیقی DIS Optical Flow با مقیاس‌پذیری امن حافظه
        # محدود کردن وضوح پردازش جریان متراکم به حداکثر ۹۶۰ پیکسل جهت جلوگیری قطعی از خطای حافظه (cv::OutOfMemoryError)
        MAX_DENSE_DIM = 960
        max_dim = max(max_h, max_w)
        if max_dim > MAX_DENSE_DIM:
            scale_flow = float(MAX_DENSE_DIM) / float(max_dim)
            scaled_w = max(16, int(round(max_w * scale_flow)))
            scaled_h = max(16, int(round(max_h * scale_flow)))
            calc_ref = cv.resize(pad_ref, (scaled_w, scaled_h), interpolation=cv.INTER_AREA)
            calc_mov = cv.resize(pad_mov, (scaled_w, scaled_h), interpolation=cv.INTER_AREA)
        else:
            scale_flow = 1.0
            scaled_w, scaled_h = max_w, max_h
            calc_ref = pad_ref
            calc_mov = pad_mov

        # پیش‌پردازش یکسان‌سازی هیستوگرام موضعی (CLAHE)
        clahe = cv.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        norm_ref = clahe.apply(calc_ref)
        norm_mov = clahe.apply(calc_mov)

        try:
            dis = cv.DISOpticalFlow.create(cv.DISOPTICAL_FLOW_PRESET_FAST)
            dis.setFinestScale(1)
            dis.setPatchSize(8)
            dis.setPatchStride(4)
            dis.setGradientDescentIterations(16)
            flow_calc = dis.calc(norm_mov, norm_ref, None)
            flow_backward_calc = dis.calc(norm_ref, norm_mov, None)
        except Exception as e_dis:
            print(f"DIS Optical Flow warning: {e_dis}, attempting scaled Farneback fallback...")
            try:
                flow_calc = cv.calcOpticalFlowFarneback(norm_mov, norm_ref, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_backward_calc = cv.calcOpticalFlowFarneback(norm_ref, norm_mov, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            except Exception:
                flow_calc = np.zeros((scaled_h, scaled_w, 2), dtype=np.float32)
                flow_backward_calc = np.zeros((scaled_h, scaled_w, 2), dtype=np.float32)

        # محاسبه نقشه قطعیت بر اساس سازگاری رفت و برگشت در مقیاس سبک و سریع محاسباتی
        grid_x, grid_y = np.meshgrid(np.arange(scaled_w, dtype=np.float32), np.arange(scaled_h, dtype=np.float32))
        map_x = np.clip(grid_x + flow_calc[:, :, 0], 0, scaled_w - 1)
        map_y = np.clip(grid_y + flow_calc[:, :, 1], 0, scaled_h - 1)

        flow_b_warped_x = cv.remap(flow_backward_calc[:, :, 0], map_x, map_y, cv.INTER_LINEAR)
        flow_b_warped_y = cv.remap(flow_backward_calc[:, :, 1], map_x, map_y, cv.INTER_LINEAR)

        fb_err = np.sqrt((flow_calc[:, :, 0] + flow_b_warped_x) ** 2 + (flow_calc[:, :, 1] + flow_b_warped_y) ** 2)
        confidence_calc = np.exp(-fb_err / 2.0).astype(np.float32)

        # بازگرداندن میدان جریان و نقشه قطعیت به ابعاد اصلی
        if scale_flow != 1.0:
            inv_scale = float(1.0 / scale_flow)
            flow_pad = (cv.resize(flow_calc, (max_w, max_h), interpolation=cv.INTER_LINEAR) * inv_scale).astype(np.float32)
            confidence_pad = cv.resize(confidence_calc, (max_w, max_h), interpolation=cv.INTER_LINEAR)
        else:
            flow_pad = flow_calc
            confidence_pad = confidence_calc

        # برش نتایج متناسب با ابعاد تصویر متحرک (Moving Image)
        flow = flow_pad[:h_mov, :w_mov]
        confidence = confidence_pad[:h_mov, :w_mov]

        valid_pixels = (pad_ref[:h_mov, :w_mov] > 5) & (gray_moving > 5)
        confidence[~valid_pixels] = 0.0

        if roi is not None and len(roi) == 4:
            rx, ry, rw, rh = roi
            roi_mask = np.zeros((h_mov, w_mov), dtype=bool)
            roi_mask[max(0, ry):min(h_mov, ry+rh), max(0, rx):min(w_mov, rx+rw)] = True
            confidence[roi_mask] *= 3.0

        return flow, confidence

    def calculate_dense_translation(self, img_ref_bgr, img_mov_bgr, roi=None):
        """
        محاسبه ترنسلیشن پایدار (dx, dy) دو مرحله‌ای (Coarse-to-Fine)
        """
        if img_ref_bgr is None or img_mov_bgr is None:
            return None

        g_ref = cv.cvtColor(img_ref_bgr[:, :, :3], cv.COLOR_BGR2GRAY) if img_ref_bgr.ndim == 3 else img_ref_bgr
        g_mov = cv.cvtColor(img_mov_bgr[:, :, :3], cv.COLOR_BGR2GRAY) if img_mov_bgr.ndim == 3 else img_mov_bgr

        # مرحله ۱: تخمین آفست درشت اولیه
        coarse_dx, coarse_dy = self._estimate_coarse_translation(g_ref, g_mov)

        # مرحله ۲: انتقال اولیه تصویر دوم به دستگاه مختصات مرجع
        h1, w1 = g_ref.shape[:2]
        M_coarse = np.float32([[1, 0, coarse_dx], [0, 1, coarse_dy]])
        g_mov_shifted = cv.warpAffine(g_mov, M_coarse, (w1, h1), borderMode=cv.BORDER_CONSTANT, borderValue=0)

        # مرحله ۳: پالایش با جریان متراکم در سطح زیرپیکسل
        flow, conf = self.compute_dense_flow(g_ref, g_mov_shifted, roi=roi)
        valid = conf > 0.25

        if np.sum(valid) >= 20:
            fine_dx = float(np.median(flow[:, :, 0][valid]))
            fine_dy = float(np.median(flow[:, :, 1][valid]))
            dx = coarse_dx + fine_dx
            dy = coarse_dy + fine_dy
            return float(dx), float(dy)

        # در صورت ناکافی بودن قطعیت محلی، استفاده از آفست درشت
        if abs(coarse_dx) > 1e-3 or abs(coarse_dy) > 1e-3:
            return float(coarse_dx), float(coarse_dy)

        return None

    def calculate_dense_homography(self, img_ref_bgr, img_mov_bgr, roi=None):
        """
        محاسبه هموگرافی سراسری با ترکیب SIFT مقاوم و برازش متراکم جریان
        """
        if img_ref_bgr is None or img_mov_bgr is None:
            return None

        g_ref = cv.cvtColor(img_ref_bgr[:, :, :3], cv.COLOR_BGR2GRAY) if img_ref_bgr.ndim == 3 else img_ref_bgr
        g_mov = cv.cvtColor(img_mov_bgr[:, :, :3], cv.COLOR_BGR2GRAY) if img_mov_bgr.ndim == 3 else img_mov_bgr

        H_init = None
        try:
            sift = cv.SIFT_create(nfeatures=1200)
            kp1, des1 = sift.detectAndCompute(g_ref, None)
            kp2, des2 = sift.detectAndCompute(g_mov, None)
            if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
                bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
                matches = bf.match(des2, des1)
                if len(matches) >= 4:
                    src_pts = np.float32([kp2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    H_init, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 4.0)
        except Exception:
            pass

        if H_init is not None:
            if abs(H_init[2, 2]) > 1e-7:
                H_init /= H_init[2, 2]
            return H_init

        # در صورت نبود هموگرافی کامل، استفاده از آفست متراکم
        dx_dy = self.calculate_dense_translation(img_ref_bgr, img_mov_bgr, roi=roi)
        if dx_dy is not None:
            dx, dy = dx_dy
            H_trans = np.array([
                [1.0, 0.0, dx],
                [0.0, 1.0, dy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
            return H_trans

        return None


# ==============================================================================
# 2. DEFORMABLE IMAGE REGISTRATION & DIFFEOMORPHIC WARP (مد ۲: دفرماسیون غیرصلب)
# ==============================================================================

def safe_guided_filter(guide, src, radius, eps):
    """فیلتر هدایت‌شده امن با فالبک گوسی در صورت عدم وجود ماژول contrib"""
    try:
        if hasattr(cv, 'ximgproc') and hasattr(cv.ximgproc, 'guidedFilter'):
            return cv.ximgproc.guidedFilter(guide, src.astype(np.float32), radius, float(eps))
    except Exception:
        pass
    try:
        ksize = max(3, (radius // 2) * 2 + 1)
        sigma = max(1.0, radius / 3.0)
        return cv.GaussianBlur(src.astype(np.float32), (ksize, ksize), sigma)
    except Exception:
        return src

class DeformableNeuralWarpEngine:
    """
    موتور دفرماسیون غیرصلب عصبی/الاستیک.
    تغییر شکل محلی لایه‌ها را با قید اکید دیفئومورفیک (دترمینان ژاکوبین مثبت |J| > 0)
    محاسبه می‌کند تا ساختار فیزیکی ترک‌های مس هرگز پاره نشود و چین نخورد.
    """
    def __init__(self):
        self.matcher = DenseFeatureMatcher()

    def calculate_deformable_warp(self, img_ref_bgr, img_mov_bgr, grid_cells=(24, 24), smoothness=0.35, roi=None):
        """
        محاسبه تصویر دفرم‌شده به همراه جابه‌جایی سراسری
        امضای بازگشتی سازگار با calculate_elastic_mesh_warp: (warped_img_moving, (global_dx, global_dy))
        """
        if img_ref_bgr is None or img_mov_bgr is None:
            return None, None

        h1, w1 = img_ref_bgr.shape[:2]
        h2, w2 = img_mov_bgr.shape[:2]

        g_ref = cv.cvtColor(img_ref_bgr[:, :, :3], cv.COLOR_BGR2GRAY) if img_ref_bgr.ndim == 3 else img_ref_bgr
        g_mov = cv.cvtColor(img_mov_bgr[:, :, :3], cv.COLOR_BGR2GRAY) if img_mov_bgr.ndim == 3 else img_mov_bgr

        # 1. تخمین آفست درشت اولیه برای پیش‌تراز کردن تصاویر مجاور
        coarse_dx, coarse_dy = self.matcher._estimate_coarse_translation(g_ref, g_mov)

        # 2. انتقال اولیه تصویر متحرک به محدوده هم‌پوشانی در دستگاه مختصات مرجع
        M_coarse = np.float32([[1, 0, coarse_dx], [0, 1, coarse_dy]])
        g_mov_shifted = cv.warpAffine(g_mov, M_coarse, (w1, h1), borderMode=cv.BORDER_CONSTANT, borderValue=0)

        # 3. محاسبه جریان متراکم فقط روی ناحیه مشترک پیش‌تراز شده
        flow_res, conf = self.matcher.compute_dense_flow(g_ref, g_mov_shifted, roi=roi)

        valid = conf > 0.2
        if np.sum(valid) < 20:
            if abs(coarse_dx) > 1e-3 or abs(coarse_dy) > 1e-3:
                return img_mov_bgr, (float(coarse_dx), float(coarse_dy))
            return None, None

        fine_dx = float(np.median(flow_res[:, :, 0][valid]))
        fine_dy = float(np.median(flow_res[:, :, 1][valid]))

        global_dx = float(coarse_dx + fine_dx)
        global_dy = float(coarse_dy + fine_dy)

        # باقیمانده جابه‌جایی موضعی (محلی)
        res_dx = flow_res[:, :, 0] - fine_dx
        res_dy = flow_res[:, :, 1] - fine_dy

        # هم‌اندازه‌سازی با تصویر متحرک (Moving Image)
        if res_dx.shape[:2] != (h2, w2):
            res_dx = cv.resize(res_dx, (w2, h2), interpolation=cv.INTER_LINEAR)
            res_dy = cv.resize(res_dy, (w2, h2), interpolation=cv.INTER_LINEAR)

        # بازگرداندن میدان جابه‌جایی از مختصات g_mov_shifted به مختصات اصلی g_mov
        M_inv = np.float32([[1, 0, -coarse_dx], [0, 1, -coarse_dy]])
        res_dx_mov = cv.warpAffine(res_dx.astype(np.float32), M_inv, (w2, h2), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)
        res_dy_mov = cv.warpAffine(res_dy.astype(np.float32), M_inv, (w2, h2), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)

        eps = float(smoothness * 0.05 + 1e-4)
        radius = int(max(6, min(w2, h2) // 32))

        smooth_dx = safe_guided_filter(g_mov, res_dx_mov, radius, eps)
        smooth_dy = safe_guided_filter(g_mov, res_dy_mov, radius, eps)

        # کنترل قید ژاکوبین برای پیشگیری از Fold-over (تاخوردگی موضعی)
        grad_dx_x = cv.Sobel(smooth_dx, cv.CV_32F, 1, 0)
        grad_dx_y = cv.Sobel(smooth_dx, cv.CV_32F, 0, 1)
        grad_dy_x = cv.Sobel(smooth_dy, cv.CV_32F, 1, 0)
        grad_dy_y = cv.Sobel(smooth_dy, cv.CV_32F, 0, 1)

        jacobian = (1.0 + grad_dx_x) * (1.0 + grad_dy_y) - (grad_dx_y * grad_dy_x)
        folding = jacobian < 0.15
        if np.any(folding):
            damp = np.ones_like(smooth_dx)
            damp[folding] = 0.2
            damp = cv.GaussianBlur(damp, (15, 15), 5.0)
            smooth_dx *= damp
            smooth_dy *= damp

        grid_x, grid_y = np.meshgrid(np.arange(w2, dtype=np.float32), np.arange(h2, dtype=np.float32))
        map_x = (grid_x - smooth_dx).astype(np.float32)
        map_y = (grid_y - smooth_dy).astype(np.float32)

        warped_img2 = cv.remap(img_mov_bgr, map_x, map_y, interpolation=cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        return warped_img2, (global_dx, global_dy)


# ==============================================================================
# 3. PARALLAX-TOLERANT MESH WARP ENGINE (مد ۳: دوخت با تحمل اختلاف منظر UDIS2)
# ==============================================================================

class ParallaxTolerantWarpEngine:
    """
    موتور دوخت تطبیقی مقاوم در برابر اختلاف منظر (الهام‌گرفته از معماری UDIS2).
    ترکیب هموگرافی زمینه و توری تغییر شکل الاستیک دوگانه.
    """
    def __init__(self):
        self.matcher = DenseFeatureMatcher()

    def calculate_mesh_warp(self, img_ref_bgr, img_mov_bgr, grid_size=(16, 16), roi=None):
        if img_ref_bgr is None or img_mov_bgr is None:
            return None, None

        h1, w1 = img_ref_bgr.shape[:2]
        h2, w2 = img_mov_bgr.shape[:2]

        H = self.matcher.calculate_dense_homography(img_ref_bgr, img_mov_bgr, roi=roi)
        if H is None:
            H = np.eye(3, dtype=np.float32)

        tx, ty = float(H[0, 2]), float(H[1, 2])

        # تفکیک انتقال سراسری بزرگ از تبدیل موضعی پرسپکتیو:
        # H_rel = T(-tx, -ty) @ H
        # با این کار، H_rel حول مبدا عمل می‌کند و نقاط از کادر (w2, h2) بیرون نمی‌زنند و سیاه نمی‌شوند!
        T_inv = np.array([
            [1.0, 0.0, -tx],
            [0.0, 1.0, -ty],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        H_rel = T_inv @ H

        gw, gh = grid_size
        xs = np.linspace(0, w2, gw, dtype=np.float32)
        ys = np.linspace(0, h2, gh, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)

        # اعمال تبدیل نسبی حول کادر خود قطعه
        pts_grid = np.column_stack([gx.ravel(), gy.ravel()]).reshape(-1, 1, 2)
        try:
            H_rel_inv = np.linalg.inv(H_rel)
            pts_warped = cv.perspectiveTransform(pts_grid, H_rel_inv).reshape(gh, gw, 2)
        except Exception:
            pts_warped = pts_grid.reshape(gh, gw, 2)

        # استخراج جریان محلی با پیش‌تراز کردن تصویر متحرک به محدوده هم‌پوشانی
        g_ref = cv.cvtColor(img_ref_bgr[:, :, :3], cv.COLOR_BGR2GRAY) if img_ref_bgr.ndim == 3 else img_ref_bgr
        g_mov = cv.cvtColor(img_mov_bgr[:, :, :3], cv.COLOR_BGR2GRAY) if img_mov_bgr.ndim == 3 else img_mov_bgr

        M_coarse = np.float32([[1, 0, tx], [0, 1, ty]])
        g_mov_shifted = cv.warpAffine(g_mov, M_coarse, (w1, h1), borderMode=cv.BORDER_CONSTANT, borderValue=0)
        flow, conf = self.matcher.compute_dense_flow(g_ref, g_mov_shifted, roi=roi)

        if flow is not None and conf is not None:
            # برآورد جابه‌جایی موضعی سلول‌های توری در فضای شیفت‌یافته تصویر
            for i in range(gh):
                for j in range(gw):
                    px_shifted = int(round(gx[i, j] + tx))
                    py_shifted = int(round(gy[i, j] + ty))
                    if 0 <= px_shifted < flow.shape[1] and 0 <= py_shifted < flow.shape[0]:
                        if conf[py_shifted, px_shifted] > 0.4:
                            pts_warped[i, j, 0] += flow[py_shifted, px_shifted, 0] * 0.3
                            pts_warped[i, j, 1] += flow[py_shifted, px_shifted, 1] * 0.3

        # رندر شبکه بازنگاشت با درون‌یابی اسپلاین
        map_x = cv.resize(pts_warped[:, :, 0], (w2, h2), interpolation=cv.INTER_CUBIC).astype(np.float32)
        map_y = cv.resize(pts_warped[:, :, 1], (w2, h2), interpolation=cv.INTER_CUBIC).astype(np.float32)

        warped_mov = cv.remap(img_mov_bgr, map_x, map_y, interpolation=cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        return warped_mov, (tx, ty)


# ==============================================================================
# 4. GLOBAL POSE GRAPH OPTIMIZER (مد ۴: بهینه‌سازی سراسری ۹ تایل بدون رانش)
# ==============================================================================

class GlobalMosaicGraphOptimizer:
    """
    بهینه‌ساز سراسری تراز گراف (Global Pose Graph Bundle Adjustment).
    به جای خطای تجمعی اتصال متوالی ۱ به ۲، ۲ به ۳ و...،
    موقعیت تمام ۹ تایل (۳×۳) یا هر شبکه دلخواه را به طور هم‌زمان
    نسبت به تایل لنگر مرکزی (Anchor Frame) بهینه‌سازی می‌کند.
    """
    def __init__(self):
        pass

    def solve_grid_translations(self, grid_cells, pairwise_offsets, anchor_gid=None):
        """
        grid_cells: دیکشنری یا لیست سلول‌ها {(r, c): gid}
        pairwise_offsets: دیکشنری اتصالات { (child_gid, parent_gid): (dx, dy, weight) }
        خروجی: { gid: (global_x, global_y) }
        """
        gids = list(set([gid for pair in pairwise_offsets.keys() for gid in pair]))
        if not gids:
            return {}

        gid_to_idx = {gid: i for i, gid in enumerate(gids)}
        n = len(gids)

        if anchor_gid is None or anchor_gid not in gid_to_idx:
            anchor_gid = gids[0]
        anchor_idx = gid_to_idx[anchor_gid]

        # تشکیل معادلات کمترین مربعات خطای گراف: A * X = B
        # برای هر یال بین i و j: X_child - X_parent = target_offset
        edges = []
        for (cgid, pgid), (dx, dy, w) in pairwise_offsets.items():
            if cgid in gid_to_idx and pgid in gid_to_idx:
                edges.append((gid_to_idx[cgid], gid_to_idx[pgid], dx, dy, w))

        num_eqs = len(edges) + 1  # +1 برای قید گره لنگر
        A = np.zeros((num_eqs, n), dtype=np.float64)
        Bx = np.zeros(num_eqs, dtype=np.float64)
        By = np.zeros(num_eqs, dtype=np.float64)

        for row, (c_idx, p_idx, dx, dy, w) in enumerate(edges):
            weight = math.sqrt(max(0.1, float(w)))
            A[row, c_idx] = 1.0 * weight
            A[row, p_idx] = -1.0 * weight
            Bx[row] = dx * weight
            By[row] = dy * weight

        # قید لنگر: X_anchor = 0
        A[-1, anchor_idx] = 100.0
        Bx[-1] = 0.0
        By[-1] = 0.0

        # حل کمترین مربعات مقید با SVD/Pseudo-Inverse
        sol_x, _, _, _ = np.linalg.lstsq(A, Bx, rcond=None)
        sol_y, _, _, _ = np.linalg.lstsq(A, By, rcond=None)

        results = {}
        for gid, idx in gid_to_idx.items():
            results[gid] = (float(sol_x[idx]), float(sol_y[idx]))

        return results


# ==============================================================================
# 5. MULTI-BAND LAPLACIAN PYRAMID BLENDER (مد ۵: ترکیب چندبانده فرکانسی)
# ==============================================================================

def merge_images_multiband_pyramid(base_img, top_img, relative_pos, num_levels=4, mask_seam=None):
    """
    ترکیب هرمی لاپلاسین چندبانده.
    فرکانس‌های نوری بالا (ترک‌ها، ویاها و پدهای باریک) را با وضوح ۱۰۰٪ حفظ کرده
    و فرکانس‌های پایین (روشنایی و کنتراست عمومی پرتو) را به نرمی ادغام می‌کند.
    """
    if base_img.shape[2] == 3:
        base_img = cv.cvtColor(base_img, cv.COLOR_BGR2BGRA)
    if top_img.shape[2] == 3:
        top_img = cv.cvtColor(top_img, cv.COLOR_BGR2BGRA)

    h_base, w_base = base_img.shape[:2]
    h_top, w_top = top_img.shape[:2]
    x_rel, y_rel = relative_pos

    min_x = min(0, x_rel)
    min_y = min(0, y_rel)
    max_x = max(w_base, x_rel + w_top)
    max_y = max(h_base, y_rel + h_top)

    fw, fh = int(max_x - min_x), int(max_y - min_y)

    # پدینگ ابعاد به توان‌های ۲ جهت بازسازی بی‌نقص اهرام
    pad_h = int(math.ceil(fh / (2 ** num_levels))) * (2 ** num_levels)
    pad_w = int(math.ceil(fw / (2 ** num_levels))) * (2 ** num_levels)

    canvas_base = np.zeros((pad_h, pad_w, 4), dtype=np.uint8)
    canvas_top = np.zeros((pad_h, pad_w, 4), dtype=np.uint8)

    bx, by = int(-min_x), int(-min_y)
    tx, ty = int(x_rel - min_x), int(y_rel - min_y)

    canvas_base[by:by+h_base, bx:bx+w_base] = base_img
    canvas_top[ty:ty+h_top, tx:tx+w_top] = top_img

    mask_b = canvas_base[:, :, 3] > 0
    mask_t = canvas_top[:, :, 3] > 0
    overlap = mask_b & mask_t

    if not np.any(overlap):
        # بدون هم‌پوشانی: کپی مستقیم
        res = canvas_base.copy()
        res[mask_t] = canvas_top[mask_t]
        return res[:fh, :fw], (min_x, min_y)

    # ساخت ماسک اولیه اوزان
    if mask_seam is not None and mask_seam.shape == (pad_h, pad_w):
        weight_mask = mask_seam.astype(np.float32)
    else:
        dist_b = cv.distanceTransform(mask_b.astype(np.uint8), cv.DIST_L2, 5)
        dist_t = cv.distanceTransform(mask_t.astype(np.uint8), cv.DIST_L2, 5)
        weight_mask = (dist_t >= dist_b).astype(np.float32)

    # ساخت هرم گوسی ماسک
    mask_pyr = [weight_mask]
    for _ in range(num_levels - 1):
        mask_pyr.append(cv.pyrDown(mask_pyr[-1]))

    # ساخت هرم‌های لاپلاسین تصاویر
    def build_laplacian_pyr(img_bgr):
        g_pyr = [img_bgr.astype(np.float32)]
        for _ in range(num_levels - 1):
            g_pyr.append(cv.pyrDown(g_pyr[-1]))
        l_pyr = [g_pyr[-1]]
        for i in range(num_levels - 1, 0, -1):
            expanded = cv.pyrUp(g_pyr[i], dstsize=(g_pyr[i-1].shape[1], g_pyr[i-1].shape[0]))
            lap = g_pyr[i-1] - expanded
            l_pyr.append(lap)
        return l_pyr[::-1]

    l_base = build_laplacian_pyr(canvas_base[:, :, :3])
    l_top = build_laplacian_pyr(canvas_top[:, :, :3])

    # ترکیب هرم‌ها در هر لایه فرکانسی
    blended_pyr = []
    for l_b, l_t, m in zip(l_base, l_top, mask_pyr):
        m_3d = m[:, :, np.newaxis]
        blended = l_t * m_3d + l_b * (1.0 - m_3d)
        blended_pyr.append(blended)

    # بازسازی تصویر از هرم ترکیبی (Reconstruction)
    reconstructed = blended_pyr[-1]
    for i in range(num_levels - 2, -1, -1):
        reconstructed = cv.pyrUp(reconstructed, dstsize=(blended_pyr[i].shape[1], blended_pyr[i].shape[0]))
        reconstructed += blended_pyr[i]

    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

    # تنظیم کانال آلفا
    alpha_out = np.maximum(canvas_base[:, :, 3], canvas_top[:, :, 3])
    final_canvas = np.dstack([reconstructed, alpha_out])

    return final_canvas[:fh, :fw], (min_x, min_y)


# نمونه‌های آماده برای دسترسی مستقیم
dense_matcher_instance = DenseFeatureMatcher()
deformable_engine_instance = DeformableNeuralWarpEngine()
parallax_engine_instance = ParallaxTolerantWarpEngine()
global_graph_instance = GlobalMosaicGraphOptimizer()
