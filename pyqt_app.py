# pyqt_app.py
"""
Picture Puzzle | پازل تصاویر
Modern PyQt6 Graphic UI/UX with Light / White Theme, Colorful Icons & Canvas Hotkey HUD
"""

import sys
import os
import math
import copy
import json
import datetime
import numpy as np
import cv2 as cv

try:
    from PyQt6.QtCore import Qt, QPointF, QRectF, QTimer, pyqtSignal, QSize, QThread, QEvent, QObject
    from PyQt6.QtGui import (
        QPainter, QColor, QPen, QBrush, QFont, QImage, QPixmap,
        QKeySequence, QShortcut, QLinearGradient, QRadialGradient, QPolygonF, QIcon,
        QPainterPath
    )
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QPushButton, QLabel, QSlider, QCheckBox, QGroupBox,
        QFileDialog, QMessageBox, QStatusBar, QDockWidget, QScrollArea,
        QFrame, QSplitter, QProgressBar, QSizePolicy, QToolBar, QToolButton,
        QSpinBox, QComboBox, QDoubleSpinBox, QLineEdit
    )
except ImportError:
    print("Warning: PyQt6 is not installed. Please run: pip install PyQt6")

from manual_stitcher import perform_manual_snap
from image_merger import merge_images_with_offset, compute_center_weighted_mask
from auto_stitcher import (
    calculate_superglue_transform,
    get_match_confidence,
    calculate_translation_offset,
    calculate_overlap_ssim,
    compute_accuracy_metrics,
    update_matching_config,
    calculate_elastic_mesh_warp,
    calculate_smart_snap_offset,
    calculate_auto_align_transform,
    calculate_warp_deformation,
    optimize_global_grid_offsets
)

INPUT_DIRECTORY = ''
OUTPUT_SAVE_DIRECTORY = 'puzzle_outputs'
PROJECT_SAVE_DIRECTORY = 'puzzle_saves'
MANUAL_SNAP_THRESHOLD = 20

# ================== Natural Sorting & Helpers ==================
def natural_sort_key(s):
    """
    Key for natural / alphanumeric sorting (e.g. 1, 2, 3 ... 9, 10, 11).
    Splits digits from text and converts digits to integers.
    """
    import re
    if not isinstance(s, str):
        s = str(s)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def detect_mosaic_grid_from_files(filenames):
    """
    Checks if filenames have 2D grid coordinates like (row, col) or (r, c) or (y, x).
    Examples:
      'tile_r01_c02.png' -> (1, 2)
      'img_1_2.jpg' -> (1, 2)
      'row2_col3.tif' -> (2, 3)
      'slice_y0_x1.png' -> (0, 1)
    Returns a dict {filename: (r, c)} if consistent coordinates found for all files, else None.
    """
    import re
    if not filenames or len(filenames) < 2:
        return None

    patterns = [
        r'(?:r|row)[_\-\s]*(\d+)[_\-\s]*(?:c|col)[_\-\s]*(\d+)',
        r'(?:y)[_\-\s]*(\d+)[_\-\s]*(?:x)[_\-\s]*(\d+)',
        r'^[^\d]*(\d+)[_\-\s]+(\d+)[^\d]*$'
    ]

    for pat in patterns:
        coords = {}
        matched_all = True
        for f in filenames:
            base = os.path.splitext(os.path.basename(f))[0]
            m = re.search(pat, base, re.IGNORECASE)
            if m:
                coords[f] = (int(m.group(1)), int(m.group(2)))
            else:
                matched_all = False
                break
        if matched_all and len(coords) == len(filenames):
            rows = {c[0] for c in coords.values()}
            cols = {c[1] for c in coords.values()}
            if len(rows) > 1 or len(cols) > 1:
                return coords

    return None


def get_optimal_matrix_cols(n):
    """
    Calculates the optimal number of columns for an n-tile 2D mosaic matrix.
    Prefers balanced landscape aspect ratio.
    """
    if n <= 1:
        return 1
    root = int(round(math.sqrt(n)))
    best_c = int(math.ceil(math.sqrt(n)))
    for c in range(root, 1, -1):
        if n % c == 0:
            r = n // c
            return max(c, r)
    return best_c


def detect_2d_grid_cells(pieces, cols_hint=0, mode='matrix'):
    """
    Intelligently assigns each piece to a 2D grid cell (row, col).
    1. Checks if filenames contain 2D coordinates (e.g. r01_c02 or row1_col2).
    2. If not, checks if pieces are already spatially arranged on canvas:
       Clusters pieces by Y (rows) and sorts each row by X (columns).
    3. Fallback: sorts naturally and divides into cols_hint (or optimal cols).
    Returns (num_rows, num_cols, piece_cells) where piece_cells is list of (row, col, piece).
    """
    n = len(pieces)
    if n == 0:
        return 0, 0, []
    if n == 1:
        return 1, 1, [(0, 0, pieces[0])]

    # 1. Try filename-based grid
    filenames = [p.filename for p in pieces if hasattr(p, 'filename') and p.filename]
    if len(filenames) == n:
        grid_coords = detect_mosaic_grid_from_files(filenames)
        if grid_coords:
            min_r = min(grid_coords[p.filename][0] for p in pieces)
            min_c = min(grid_coords[p.filename][1] for p in pieces)
            piece_cells = []
            for p in pieces:
                r, c = grid_coords[p.filename]
                piece_cells.append((r - min_r, c - min_c, p))
            num_rows = max(item[0] for item in piece_cells) + 1
            num_cols = max(item[1] for item in piece_cells) + 1
            return num_rows, num_cols, piece_cells

    # 2. Modes: pure single row or single column
    if mode == 'row':
        sorted_by_x = sorted(pieces, key=lambda p: (p.x, natural_sort_key(getattr(p, 'filename', '') or str(p.id))))
        return 1, n, [(0, c, p) for c, p in enumerate(sorted_by_x)]
    elif mode == 'col':
        sorted_by_y = sorted(pieces, key=lambda p: (p.y, natural_sort_key(getattr(p, 'filename', '') or str(p.id))))
        return n, 1, [(r, 0, p) for r, p in enumerate(sorted_by_y)]

    # 3. Spatial clustering by (X, Y) on canvas (if pieces were moved into approximate rows/cols)
    avg_h = sum(p.h for p in pieces) / float(n)
    y_span = max(p.y for p in pieces) - min(p.y for p in pieces)

    if y_span > avg_h * 0.4 and cols_hint <= 0:
        sorted_by_y = sorted(pieces, key=lambda p: p.y)
        rows_list = []
        cur_row = [sorted_by_y[0]]
        cur_row_y = sorted_by_y[0].y

        for p in sorted_by_y[1:]:
            if abs(p.y - cur_row_y) > avg_h * 0.45:
                rows_list.append(cur_row)
                cur_row = [p]
                cur_row_y = p.y
            else:
                cur_row.append(p)
                cur_row_y = sum(item.y for item in cur_row) / len(cur_row)
        if cur_row:
            rows_list.append(cur_row)

        if len(rows_list) > 1:
            piece_cells = []
            max_c = 0
            for r, row_pieces in enumerate(rows_list):
                row_sorted = sorted(row_pieces, key=lambda p: p.x)
                for c, p in enumerate(row_sorted):
                    piece_cells.append((r, c, p))
                    if c + 1 > max_c:
                        max_c = c + 1
            return len(rows_list), max_c, piece_cells

    # 4. Fallback: Sequential order using natural sort
    sorted_pieces = sorted(pieces, key=lambda p: natural_sort_key(getattr(p, 'filename', '') or str(p.id)))
    if cols_hint > 0:
        num_cols = max(1, min(cols_hint, n))
    else:
        num_cols = get_optimal_matrix_cols(n)

    num_rows = int(math.ceil(n / float(num_cols)))
    piece_cells = []
    for i, p in enumerate(sorted_pieces):
        r = i // num_cols
        c = i % num_cols
        piece_cells.append((r, c, p))
    return num_rows, num_cols, piece_cells


# ================== Data Structure ==================
class PuzzlePiece:
    def __init__(self, id, image, x=0, y=0, original_ref=None, filename="", group_id=None):
        self.id = int(id)
        self.image = image
        self.x = float(x)
        self.y = float(y)
        self.w = image.shape[1]
        self.h = image.shape[0]
        self.is_selected = False
        self.group_id = int(group_id) if group_id is not None else int(id)
        self.original_image_ref = original_ref if original_ref is not None else image
        self.filename = filename
        self.boundary_world = None
        self.boundary_src = None
        self.corner_indices = [0, 20, 40, 60]
        self.corners_world = None
        self.sync_corners_to_rect()

    @staticmethod
    def _calc_rect_boundary(x, y, w, h, num_per_edge=20):
        """
        Generates 4 * num_per_edge boundary points (corners at 0, N, 2N, 3N):
        Top edge: [x + (i/N)*w, y] for i in 0..N-1
        Right edge: [x + w, y + (i/N)*h] for i in 0..N-1
        Bottom edge: [x + (1 - i/N)*w, y + h] for i in 0..N-1
        Left edge: [x, y + (1 - i/N)*h] for i in 0..N-1
        """
        N = int(num_per_edge)
        pts = np.zeros((4 * N, 2), dtype=np.float32)
        for i in range(N):
            frac = float(i) / float(N)
            # Top edge: TL -> TR
            pts[i] = [x + frac * w, y]
            # Right edge: TR -> BR
            pts[N + i] = [x + w, y + frac * h]
            # Bottom edge: BR -> BL
            pts[2 * N + i] = [x + (1.0 - frac) * w, y + h]
            # Left edge: BL -> TL
            pts[3 * N + i] = [x, y + (1.0 - frac) * h]
        return pts

    def sync_corners_to_rect(self):
        orig = self.original_image_ref
        orig_h, orig_w = orig.shape[:2]
        self.corner_indices = [0, 20, 40, 60]
        self.boundary_src = self._calc_rect_boundary(0, 0, orig_w, orig_h, num_per_edge=20)
        self.boundary_world = self._calc_rect_boundary(self.x, self.y, self.w, self.h, num_per_edge=20)
        self.corners_world = self.boundary_world[self.corner_indices].copy()

    def get_boundary_world(self):
        if not hasattr(self, 'boundary_world') or self.boundary_world is None:
            self.sync_corners_to_rect()
        return self.boundary_world

    def get_corners_world(self):
        if not hasattr(self, 'boundary_world') or self.boundary_world is None:
            self.sync_corners_to_rect()
        c_idx = getattr(self, 'corner_indices', [0, 20, 40, 60])
        return self.boundary_world[c_idx]

    def translate(self, dx, dy):
        self.x += float(dx)
        self.y += float(dy)
        if hasattr(self, 'boundary_world') and self.boundary_world is not None:
            self.boundary_world[:, 0] += float(dx)
            self.boundary_world[:, 1] += float(dy)
            c_idx = getattr(self, 'corner_indices', [0, 20, 40, 60])
            self.corners_world = self.boundary_world[c_idx].copy()
        elif hasattr(self, 'corners_world') and self.corners_world is not None:
            self.corners_world[:, 0] += float(dx)
            self.corners_world[:, 1] += float(dy)



# ================== Background Worker Thread ==================
class ProcessingWorker(QThread):
    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, task_fn, *args, **kwargs):
        super().__init__()
        self.task_fn = task_fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            try:
                import torch
                torch.set_grad_enabled(False)
            except ImportError:
                pass
            res = self.task_fn(*self.args, **self.kwargs)
            self.result_ready.emit(res)
        except Exception as e:
            self.error_occurred.emit(str(e))


# ================== OpenCV Mat to QPixmap ==================
def cv_mat_to_qpixmap(mat):
    if mat is None or mat.size == 0:
        return QPixmap()
    # F-02: Ensure uint8 data type for Qt QImage buffer compatibility
    if mat.dtype == np.uint16:
        mat_8u = (mat / 256.0).astype(np.uint8)
    elif mat.dtype != np.uint8:
        mat_8u = cv.normalize(mat, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    else:
        mat_8u = mat

    h, w = mat_8u.shape[:2]
    if mat_8u.ndim == 2:
        qimg = QImage(mat_8u.data, w, h, w, QImage.Format.Format_Grayscale8)
    elif mat_8u.shape[2] == 4:
        qimg = QImage(mat_8u.data, w, h, w * 4, QImage.Format.Format_ARGB32)
    elif mat_8u.shape[2] == 3:
        rgb = cv.cvtColor(mat_8u, cv.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    else:
        return QPixmap()
    return QPixmap.fromImage(qimg.copy())


# ================== 2D Moving Least Squares (MLS) Elastic Mesh Warping ==================
def compute_mls_affine_warp(src_pts, dst_pts, orig_img, out_shape, boundary_dst):
    """
    Computes real-time 2D Moving Least Squares (MLS) Affine deformation.
    Deforms orig_img so that src_pts smoothly and elastically map to dst_pts,
    and cleanly clips the result to boundary_dst polygon with antialiased alpha.
    Zero rectangular bleed outside the dashed boundary.
    """
    out_h, out_w = int(out_shape[0]), int(out_shape[1])
    out_h = max(16, out_h)
    out_w = max(16, out_w)

    if orig_img.ndim == 2:
        orig_rgba = cv.cvtColor(orig_img, cv.COLOR_GRAY2BGRA)
    elif orig_img.shape[2] == 3:
        orig_rgba = cv.cvtColor(orig_img, cv.COLOR_BGR2BGRA)
    else:
        orig_rgba = orig_img.copy()

    # Coarse grid for ultra-fast MLS computation (computes in < 6ms)
    grid_step = 16
    gw = max(8, int(math.ceil(out_w / grid_step)) + 1)
    gh = max(8, int(math.ceil(out_h / grid_step)) + 1)
    xs = np.linspace(0.0, float(out_w), gw, dtype=np.float32)
    ys = np.linspace(0.0, float(out_h), gh, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    grid_q = np.stack([gx, gy], axis=-1)  # shape (gh, gw, 2)

    # Vectorized MLS Affine: map from grid_q (destination) back to source
    M = len(dst_pts)
    dst_pts_np = np.asarray(dst_pts, dtype=np.float32)
    src_pts_np = np.asarray(src_pts, dtype=np.float32)

    diff = grid_q[:, :, np.newaxis, :] - dst_pts_np[np.newaxis, np.newaxis, :, :]  # (gh, gw, M, 2)
    sq_dist = np.sum(diff ** 2, axis=-1) + 1.0  # (gh, gw, M)
    weights = 1.0 / sq_dist  # (gh, gw, M)
    sum_w = np.sum(weights, axis=-1, keepdims=True) + 1e-6  # (gh, gw, 1)

    q_star = np.sum(weights[:, :, :, np.newaxis] * dst_pts_np[np.newaxis, np.newaxis, :, :], axis=2) / sum_w  # (gh, gw, 2)
    p_star = np.sum(weights[:, :, :, np.newaxis] * src_pts_np[np.newaxis, np.newaxis, :, :], axis=2) / sum_w  # (gh, gw, 2)

    q_hat = dst_pts_np[np.newaxis, np.newaxis, :, :] - q_star[:, :, np.newaxis, :]  # (gh, gw, M, 2)
    p_hat = src_pts_np[np.newaxis, np.newaxis, :, :] - p_star[:, :, np.newaxis, :]  # (gh, gw, M, 2)

    c_xx = np.sum(weights * (q_hat[:, :, :, 0] ** 2), axis=-1)
    c_yy = np.sum(weights * (q_hat[:, :, :, 1] ** 2), axis=-1)
    c_xy = np.sum(weights * (q_hat[:, :, :, 0] * q_hat[:, :, :, 1]), axis=-1)

    det = c_xx * c_yy - c_xy ** 2 + 1e-4
    inv_c_00 = c_yy / det
    inv_c_01 = -c_xy / det
    inv_c_10 = -c_xy / det
    inv_c_11 = c_xx / det

    m_00 = np.sum(weights * p_hat[:, :, :, 0] * q_hat[:, :, :, 0], axis=-1)
    m_01 = np.sum(weights * p_hat[:, :, :, 0] * q_hat[:, :, :, 1], axis=-1)
    m_10 = np.sum(weights * p_hat[:, :, :, 1] * q_hat[:, :, :, 0], axis=-1)
    m_11 = np.sum(weights * p_hat[:, :, :, 1] * q_hat[:, :, :, 1], axis=-1)

    a_00 = m_00 * inv_c_00 + m_01 * inv_c_10
    a_01 = m_00 * inv_c_01 + m_01 * inv_c_11
    a_10 = m_10 * inv_c_00 + m_11 * inv_c_10
    a_11 = m_10 * inv_c_01 + m_11 * inv_c_11

    dq_x = grid_q[:, :, 0] - q_star[:, :, 0]
    dq_y = grid_q[:, :, 1] - q_star[:, :, 1]

    grid_u = a_00 * dq_x + a_01 * dq_y + p_star[:, :, 0]
    grid_v = a_10 * dq_x + a_11 * dq_y + p_star[:, :, 1]

    dense_map_x = cv.resize(grid_u.astype(np.float32), (out_w, out_h), interpolation=cv.INTER_CUBIC)
    dense_map_y = cv.resize(grid_v.astype(np.float32), (out_w, out_h), interpolation=cv.INTER_CUBIC)

    warped = cv.remap(
        orig_rgba,
        dense_map_x,
        dense_map_y,
        interpolation=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    # Exact polygon clipping with subtle antialiased feathering
    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    b_clipped = np.asarray(boundary_dst, dtype=np.float32).copy()
    b_clipped[:, 0] = np.clip(b_clipped[:, 0], 0.0, float(out_w - 1))
    b_clipped[:, 1] = np.clip(b_clipped[:, 1], 0.0, float(out_h - 1))
    poly_pts = np.round(b_clipped).astype(np.int32)
    cv.fillPoly(mask, [poly_pts], 255)
    cv.GaussianBlur(mask, (3, 3), 0.5, dst=mask)
    warped[:, :, 3] = np.minimum(warped[:, :, 3], mask)

    return warped


# ================== Elemental Vector Icon Generators ==================
def create_dual_blocks_icon(translucent=True, opacity=0.55, size=48):
    """
    Elemental vector icon of two overlapping image blocks:
    - Base block (Block #1): Solid technical tile with grid lines.
    - Top block (Block #2): Diagonally offset, translucent see-through layer
      with vivid accent border and transparency indicator.
    """
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    b_size = size * 0.56
    # Base Block (bottom-left)
    b1_x = size * 0.08
    b1_y = size * 0.36
    b1_rect = QRectF(b1_x, b1_y, b_size, b_size)
    painter.setBrush(QBrush(QColor("#F1F5F9")))
    painter.setPen(QPen(QColor("#64748B"), 2.0))
    painter.drawRoundedRect(b1_rect, 4, 4)

    # Micro grid lines inside Base Block
    painter.setPen(QPen(QColor("#CBD5E1"), 1.2, Qt.PenStyle.DotLine))
    painter.drawLine(QPointF(b1_x + 5, b1_y + b_size / 2), QPointF(b1_x + b_size - 5, b1_y + b_size / 2))
    painter.drawLine(QPointF(b1_x + b_size / 2, b1_y + 5), QPointF(b1_x + b_size / 2, b1_y + b_size - 5))

    # Top Block (top-right, overlapping)
    b2_x = size * 0.36
    b2_y = size * 0.08
    b2_rect = QRectF(b2_x, b2_y, b_size, b_size)

    if translucent:
        alpha = int(255 * max(0.25, min(0.9, opacity)))
        painter.setBrush(QBrush(QColor(37, 99, 235, alpha)))
        painter.setPen(QPen(QColor("#2563EB"), 2.4))
        painter.drawRoundedRect(b2_rect, 4, 4)

        # Diagonal transparency indicator hatches
        painter.setPen(QPen(QColor(255, 255, 255, 220), 1.8))
        painter.drawLine(QPointF(b2_x + 6, b2_y + 12), QPointF(b2_x + 12, b2_y + 6))
        painter.drawLine(QPointF(b2_x + b_size - 12, b2_y + b_size - 6), QPointF(b2_x + b_size - 6, b2_y + b_size - 12))
    else:
        painter.setBrush(QBrush(QColor("#FFFFFF")))
        painter.setPen(QPen(QColor("#334155"), 2.2))
        painter.drawRoundedRect(b2_rect, 4, 4)

    painter.end()
    return QIcon(pix)


def create_toggle_element_icon(is_on=True, size=48):
    """
    Elemental vector icon of an interactive toggle switch element.
    """
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    tw = size * 0.88
    th = size * 0.48
    tx = (size - tw) / 2
    ty = (size - th) / 2
    track_rect = QRectF(tx, ty, tw, th)
    radius = th / 2.0

    if is_on:
        grad = QLinearGradient(tx, ty, tx + tw, ty)
        grad.setColorAt(0.0, QColor("#2563EB"))
        grad.setColorAt(1.0, QColor("#4F46E5"))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor("#1D4ED8"), 1.2))
        painter.drawRoundedRect(track_rect, radius, radius)

        thumb_diam = th - 6
        thumb_x = tx + tw - thumb_diam - 3
        thumb_y = ty + 3
        painter.setBrush(QBrush(QColor("#FFFFFF")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(thumb_x, thumb_y, thumb_diam, thumb_diam))
    else:
        painter.setBrush(QBrush(QColor("#CBD5E1")))
        painter.setPen(QPen(QColor("#94A3B8"), 1.2))
        painter.drawRoundedRect(track_rect, radius, radius)

        thumb_diam = th - 6
        thumb_x = tx + 3
        thumb_y = ty + 3
        painter.setBrush(QBrush(QColor("#FFFFFF")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(thumb_x, thumb_y, thumb_diam, thumb_diam))

    painter.end()
    return QIcon(pix)


def create_opacity_level_icon(level=100, size=48):
    """
    Elemental vector icon for layer opacity level indicator.
    """
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    margin = size * 0.12
    r = size - 2 * margin
    rect = QRectF(margin, margin, r, r)

    painter.setPen(QPen(QColor("#2563EB"), 2.2))
    painter.setBrush(QBrush(QColor("#F8FAFC")))
    painter.drawEllipse(rect)

    fill_alpha = int(255 * max(0.2, min(1.0, level / 100.0)))
    painter.setBrush(QBrush(QColor(37, 99, 235, fill_alpha)))
    painter.setPen(Qt.PenStyle.NoPen)
    span = int(360 * 16 * (level / 100.0))
    painter.drawPie(rect, 90 * 16, -span)

    painter.end()
    return QIcon(pix)


def create_smart_snap_icon(size=48):
    """Elemental icon for Smart Snap (Pair Align)"""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    b_size = size * 0.40
    painter.setBrush(QBrush(QColor("#E2E8F0")))
    painter.setPen(QPen(QColor("#059669"), 2.0))
    painter.drawRoundedRect(QRectF(size * 0.08, size * 0.30, b_size, b_size), 3, 3)

    painter.setBrush(QBrush(QColor("#D1FAE5")))
    painter.setPen(QPen(QColor("#059669"), 2.0))
    painter.drawRoundedRect(QRectF(size * 0.52, size * 0.30, b_size, b_size), 3, 3)

    painter.setBrush(QBrush(QColor("#F59E0B")))
    painter.setPen(Qt.PenStyle.NoPen)
    poly = QPolygonF([
        QPointF(size * 0.52, size * 0.18),
        QPointF(size * 0.44, size * 0.50),
        QPointF(size * 0.50, size * 0.50),
        QPointF(size * 0.46, size * 0.82),
        QPointF(size * 0.58, size * 0.44),
        QPointF(size * 0.50, size * 0.44)
    ])
    painter.drawPolygon(poly)
    painter.end()
    return QIcon(pix)


def create_homography_icon(size=48):
    """Elemental icon for Homography Auto-Align"""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    p1 = QPointF(size * 0.22, size * 0.20)
    p2 = QPointF(size * 0.82, size * 0.14)
    p3 = QPointF(size * 0.74, size * 0.84)
    p4 = QPointF(size * 0.14, size * 0.76)
    painter.setPen(QPen(QColor("#7C3AED"), 2.2))
    painter.setBrush(QBrush(QColor(124, 58, 237, 35)))
    painter.drawPolygon(QPolygonF([p1, p2, p3, p4]))

    painter.setPen(QPen(QColor("#A78BFA"), 1.4, Qt.PenStyle.DashLine))
    painter.drawLine((p1 + p4) / 2, (p2 + p3) / 2)
    painter.drawLine((p1 + p2) / 2, (p4 + p3) / 2)
    painter.end()
    return QIcon(pix)


def create_overlap_adjust_icon(size=48):
    """Elemental icon for Manual Overlap Adjustment"""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    b_size = size * 0.44
    painter.setBrush(QBrush(QColor(234, 88, 12, 40)))
    painter.setPen(QPen(QColor("#EA580C"), 2.0))
    painter.drawRoundedRect(QRectF(size * 0.12, size * 0.28, b_size, b_size), 3, 3)
    painter.setBrush(QBrush(QColor(234, 88, 12, 90)))
    painter.drawRoundedRect(QRectF(size * 0.44, size * 0.28, b_size, b_size), 3, 3)

    painter.setPen(QPen(QColor("#EA580C"), 2.2))
    painter.drawLine(QPointF(size * 0.28, size * 0.16), QPointF(size * 0.72, size * 0.16))
    painter.drawLine(QPointF(size * 0.28, size * 0.16), QPointF(size * 0.36, size * 0.10))
    painter.drawLine(QPointF(size * 0.28, size * 0.16), QPointF(size * 0.36, size * 0.22))
    painter.drawLine(QPointF(size * 0.72, size * 0.16), QPointF(size * 0.64, size * 0.10))
    painter.drawLine(QPointF(size * 0.72, size * 0.16), QPointF(size * 0.64, size * 0.22))
    painter.end()
    return QIcon(pix)


def create_merge_icon(size=48):
    """Elemental vector icon for Merge / Batch Blending"""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    b_size = size * 0.42
    painter.setBrush(QBrush(QColor(16, 185, 129, 70)))
    painter.setPen(QPen(QColor("#10B981"), 2.0))
    painter.drawRoundedRect(QRectF(size * 0.12, size * 0.28, b_size, b_size), 3, 3)

    painter.setBrush(QBrush(QColor(16, 185, 129, 150)))
    painter.drawRoundedRect(QRectF(size * 0.46, size * 0.28, b_size, b_size), 3, 3)

    painter.setPen(QPen(QColor("#059669"), 2.2))
    painter.drawLine(QPointF(size * 0.24, size * 0.16), QPointF(size * 0.44, size * 0.16))
    painter.drawLine(QPointF(size * 0.38, size * 0.10), QPointF(size * 0.44, size * 0.16))
    painter.drawLine(QPointF(size * 0.38, size * 0.22), QPointF(size * 0.44, size * 0.16))

    painter.drawLine(QPointF(size * 0.76, size * 0.16), QPointF(size * 0.56, size * 0.16))
    painter.drawLine(QPointF(size * 0.62, size * 0.10), QPointF(size * 0.56, size * 0.16))
    painter.drawLine(QPointF(size * 0.62, size * 0.22), QPointF(size * 0.56, size * 0.16))
    painter.end()
    return QIcon(pix)


def create_elastic_warp_icon(size=48):
    """Elemental vector icon for Elastic Mesh Warping (APAP / MLS)"""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    margin = size * 0.12
    p1 = QPointF(margin, margin + 4)
    p2 = QPointF(size - margin, margin - 2)
    p3 = QPointF(size - margin - 5, size - margin)
    p4 = QPointF(margin + 5, size - margin - 3)

    painter.setBrush(QBrush(QColor(124, 58, 237, 28)))
    painter.setPen(QPen(QColor("#7C3AED"), 2.0))
    painter.drawPolygon(QPolygonF([p1, p2, p3, p4]))

    painter.setPen(QPen(QColor("#A78BFA"), 1.4, Qt.PenStyle.DashLine))
    painter.drawLine((p1 + p4) / 2, (p2 + p3) / 2)
    painter.drawLine((p1 + p2) / 2, (p4 + p3) / 2)

    painter.setBrush(QBrush(QColor("#7C3AED")))
    painter.setPen(QPen(QColor("#FFFFFF"), 1.2))
    for pt in [p1, p2, p3, p4, (p1 + p2 + p3 + p4) / 4]:
        painter.drawEllipse(pt, 2.8, 2.8)

    painter.end()
    return QIcon(pix)


def create_stretch_warp_icon(size=48):
    """Elemental vector icon for Edge & Corner Stretch / Boundary Fit Tool"""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    margin = size * 0.16
    box_s = size - 2 * margin
    rect = QRectF(margin, margin, box_s, box_s)

    # Quad background & outline
    painter.setBrush(QBrush(QColor(99, 102, 241, 35)))
    painter.setPen(QPen(QColor("#6366F1"), 2.0, Qt.PenStyle.DashLine))
    painter.drawRoundedRect(rect, 4, 4)

    # 4 Corner nodes
    painter.setBrush(QBrush(QColor("#FFFFFF")))
    painter.setPen(QPen(QColor("#4F46E5"), 2.0))
    cr = size * 0.08
    painter.drawEllipse(QPointF(margin, margin), cr, cr)
    painter.drawEllipse(QPointF(size - margin, margin), cr, cr)
    painter.drawEllipse(QPointF(size - margin, size - margin), cr, cr)
    painter.drawEllipse(QPointF(margin, size - margin), cr, cr)

    # 4 Directional stretch arrows/pills on edge centers
    painter.setBrush(QBrush(QColor("#0284C7")))
    painter.setPen(QPen(QColor("#FFFFFF"), 1.2))
    er = size * 0.06
    # Top & Bottom
    painter.drawRoundedRect(QRectF(size / 2.0 - er * 1.5, margin - er, er * 3.0, er * 2.0), 2, 2)
    painter.drawRoundedRect(QRectF(size / 2.0 - er * 1.5, size - margin - er, er * 3.0, er * 2.0), 2, 2)
    # Left & Right
    painter.drawRoundedRect(QRectF(margin - er, size / 2.0 - er * 1.5, er * 2.0, er * 3.0), 2, 2)
    painter.drawRoundedRect(QRectF(size - margin - er, size / 2.0 - er * 1.5, er * 2.0, er * 3.0), 2, 2)

    painter.end()
    return QIcon(pix)



def create_globe_language_icon(size=32):
    """Elemental vector icon for Language / Translation Globe"""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    margin = size * 0.12
    r = size - 2 * margin
    rect = QRectF(margin, margin, r, r)

    # Outer circle
    painter.setPen(QPen(QColor("#2563EB"), 2.0))
    painter.setBrush(QBrush(QColor("#EFF6FF")))
    painter.drawEllipse(rect)

    # Latitudes & Longitudes
    painter.setPen(QPen(QColor("#3B82F6"), 1.4))
    # Equator line
    painter.drawLine(QPointF(margin, size / 2.0), QPointF(margin + r, size / 2.0))
    # Vertical meridian
    painter.drawLine(QPointF(size / 2.0, margin), QPointF(size / 2.0, margin + r))
    # Elliptical meridian
    painter.drawEllipse(QRectF(margin + r * 0.22, margin, r * 0.56, r))

    painter.end()
    return QIcon(pix)


def create_arrange_spacing_icon(size=48):
    """
    Vector icon representing an aligned 2D Mosaic Matrix (2x2 grid) with pixel spacing:
    - 4 image tiles (Top-Left Blue, Top-Right Emerald, Bottom-Left Violet, Bottom-Right Amber)
    - Cross spacing gap with horizontal and vertical pixel distance indicators
    """
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    pad = size * 0.08
    gap = size * 0.16
    tw = (size - 2 * pad - gap) / 2.0
    th = (size - 2 * pad - gap) / 2.0

    x0 = pad
    x1 = pad + tw + gap
    y0 = pad
    y1 = pad + th + gap

    tiles = [
        (x0, y0, "#EFF6FF", "#2563EB", "1"),
        (x1, y0, "#ECFDF5", "#059669", "2"),
        (x0, y1, "#EEF2FF", "#4F46E5", "3"),
        (x1, y1, "#FFFBEB", "#D97706", "4"),
    ]

    font_sz = max(7, int(tw * 0.45))
    painter.setFont(QFont("Segoe UI", font_sz, QFont.Weight.Bold))

    for tx, ty, bg, border, label in tiles:
        rect = QRectF(tx, ty, tw, th)
        painter.setBrush(QBrush(QColor(bg)))
        painter.setPen(QPen(QColor(border), 1.8))
        painter.drawRoundedRect(rect, 3, 3)
        painter.setPen(QColor(border))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

    # Horizontal and vertical cross gap indicator
    painter.setPen(QPen(QColor("#EA580C"), 1.8))
    mid_y = y0 + th + gap / 2.0
    painter.drawLine(QPointF(x0 + tw * 0.3, mid_y), QPointF(x1 + tw * 0.7, mid_y))
    mid_x = x0 + tw + gap / 2.0
    painter.drawLine(QPointF(mid_x, y0 + th * 0.3), QPointF(mid_x, y1 + th * 0.7))

    # Center dot
    painter.setBrush(QBrush(QColor("#EA580C")))
    painter.drawEllipse(QPointF(mid_x, mid_y), 2.2, 2.2)

    painter.end()
    return QIcon(pix)


# ================== Internationalization & Translations ==================
TRANSLATIONS = {
    'en': {
        # Window & Docks
        'window_title': "Picture Puzzle",
        'dock_title': "Control Center",
        'toolbar_title': "Main Toolbar",
        'lang_label': "Language:",
        
        # Toolbar Actions (Compact, Fixed Height, Never Hidden)
        'btn_open_tb': "Folder",
        'btn_load_tb': "Project",
        'btn_save_tb': "Save",
        'btn_export_tb': "Export",
        'btn_fit_tb': "Fit View",
        'btn_arrange_tb': "Arrange",
        'btn_smart_snap_tb': "Snap [N]",
        'btn_auto_align_tb': "Align [G]",
        'btn_elastic_tb': "Elastic [E]",
        'btn_stretch_tb': "Stretch [W]",
        'btn_merge_tb': "Merge [M]",
        'btn_overlap_tb': "Overlap",
        'btn_undo_tb': "Undo",
        'btn_redo_tb': "Redo",
        'btn_ghost_tb': "Ghost [O]",

        # Group 1: Project & File Operations
        'group_file': "📁 Project & File Operations",
        'btn_open': "📂 Open Folder",
        'btn_fit': "🔍 Fit View",
        'btn_arrange': "📐 Mosaic Matrix Layout (Rows & Cols) [ L ]",
        'tip_arrange': "Arrange mosaic tiles in an aligned 2D Matrix (rows and columns) with exact pixel spacing",
        'lbl_spacing': "Spacing (px):",
        'lbl_cols': "Columns:",
        'cols_auto': "Auto",
        'lbl_arrange_mode': "Mode:",
        'mode_matrix': "2D Mosaic Matrix",
        'mode_row': "Single Row",
        'mode_col': "Single Column",
        'btn_save': "💾 Save Project (S)",
        'btn_export': "🖼️ Export HQ (P)",

        # Group 2: Smart Stitching Tools
        'group_tools': "⚡ Smart Stitching Tools",
        'lbl_ai_engines': "🤖 AI & Backup Computing Engines:",
        'lbl_stitch_mode': "Processing Mode:",
        'stitch_mode_auto': "Auto (2D/3D Smart)",
        'stitch_mode_2d': "2D Planar (Strict Rigid)",
        'stitch_mode_3d': "3D Cone-Beam (Parallax-Aware)",
        'lbl_align_engine': "Align Engine:",
        'engine_superglue': "SuperGlue (Sparse - Default)",
        'engine_dense': "Dense Feature Matcher (Backup AI)",
        'lbl_warp_engine': "Deform Engine:",
        'engine_mls': "MLS Elastic Mesh (Default)",
        'engine_deformable': "Deformable DIR (Diffeomorphic AI)",
        'engine_parallax': "Parallax-Tolerant Warp (UDIS2)",
        'chk_global_graph': "Global Graph Solve (Drift-Free Matrix)",
        'btn_smart_snap': " Smart-Snap Pair [ N ]",
        'tip_smart_snap': "Align two adjacent image pieces without perspective warping",
        'btn_auto_align': " Homography Auto-Align [ G ]",
        'tip_auto_align': "Align using SuperGlue homography matrix (Pair or Chain)",
        'btn_elastic': " Elastic Mesh Warp [ E ]",
        'tip_elastic': "Local adaptive mesh deformation (MLS) to resolve divergent cone-beam distortion on both sides",
        'btn_stretch': " Edge & Corner Stretch [ W ]",
        'tip_stretch': "Drag tile corners and edges to freely stretch and fit boundaries against adjacent mosaic pieces",
        'btn_reset_stretch': "Reset Stretch",
        'tip_reset_stretch': "Restore selected piece to original shape and rectangular dimensions",
        'btn_merge': "🪢 Merge Selected Pieces [ M ]",
        'tip_merge': "Permanently blend and merge selected pieces using active seamline / multi-band algorithm [ M ]",
        'btn_overlap': " Manual Overlap Adjust",
        'tip_overlap': "Interactive live drag & adjust overlap slider between 2 pieces",

        # Group 3: Layer Transparency & Overlay Inspection
        'group_inspect': "Layer Transparency & Overlay Inspection",
        'lbl_opacity': "Top Layer Opacity: {val}%",
        'btn_toggle_opacity_off': " Toggle See-Through [ O ]",
        'btn_toggle_opacity_on': " See-Through Active ({val}%)  [ O ]",
        'tip_toggle_opacity': "Toggle top layer opacity between 55% (translucent) and 100% (solid) for see-through inspection",

        # Group 4: X-Ray Cone-Beam Distortion Fix
        'group_xray': "🛡️ X-Ray Cone-Beam Distortion Fix",
        'lbl_blend_mode': "Merge Seam Mode:",
        'btn_mode_seamline': "Seamline Cut (Zero Ghosting)",
        'btn_mode_feather': "Feathering",
        'btn_mode_multiband': "Multi-Band Pyramid",
        'tip_blend_mode': "Optimal Seamline Cut avoids doubling discrete screws and wires; Feathering provides smooth transition",
        'chk_feather': "Enable Center-Weighted Blending",
        'tip_feather': "Favor perpendicular ray rays from tile center over distorted edge rays",
        'lbl_power': "Feather Softness (Power): {val:.1f}",
        'lbl_edge_bright': "Edge Seam Brightness: {sign}{val}",
        'tip_edge_bright': "Compensate for edge vignetting / falloff darkness at the merge seam",

        # Group 5: X-Ray Geometric & Physical Calibration
        'group_calib': "📐 X-Ray Geometric Calibration",
        'lbl_calib_status_uncal': "Scale: 2D Pixel Domain (Uncalibrated)",
        'lbl_calib_status_cal': 'Manual geometry: M={mag:.3f}, pitch={pitch_mm:.5f} mm; metric accuracy unverified',
        'lbl_sdd': "SDD (mm):",
        'tip_sdd': "Source-to-Detector Distance in mm. Must be greater than SOD.",
        'lbl_sod': "SOD (mm):",
        'tip_sod': "Source-to-Object Distance in mm.",
        'lbl_pitch': "Pixel Pitch (µm):",
        'tip_pitch': "Detector physical pixel pitch in micrometers (e.g. 100 or 150 µm).",
        'lbl_odd_display': "ODD: {val}",
        'lbl_mag_display': "Magnification M₀: {val}",
        'val_unspecified': "Unspecified (Null)",
        'btn_clear_calib': "Reset Calibration to Pixel Domain",
        'btn_cone_rectify': 'Reference-plane mosaic | موزاییک صفحه مرجع',
        'tip_cone_rectify': 'Import measured calibration and absolute per-shot poses; process originals and export coverage. Other depths retain parallax.',
        'dialog_cone_rectify_title': "Cone-Beam Orthorectification",
        'dialog_cone_rectify_req': "Please specify valid SDD (mm) and Pixel Pitch (µm) in the Calibration panel first.",
        'status_cone_rectify_done': "Cone-beam radial orthorectification applied to {count} tiles.",

        # Group 6: Edit & Canvas Selection
        'group_edit': "🎨 Edit & Canvas Selection",
        'btn_undo': "↩ Undo [ Z ]",
        'btn_redo': "↪ Redo [ Y ]",
        'btn_select_all': "⊞ Select All [ A ]",
        'btn_del': "🗑 Delete [ D ]",

        # Group 6: Keyboard Shortcuts Reference
        'group_shortcuts': "⌨️ Keyboard Shortcuts Reference",
        'sc_smart_snap': "Smart Snap (Pair Align)",
        'sc_auto_align': "Auto-Align Batch / Pair",
        'sc_elastic': "Elastic Mesh Warp (Local Parallax)",
        'sc_stretch': "Edge & Corner Stretch / Fit",
        'sc_transparency': "Toggle Layer Transparency",
        'sc_overlap': "Manual Overlap Mode",
        'sc_nudge': "Pixel Micro-Nudge (1px)",
        'sc_fast_nudge': "Fast Nudge Step (Shift = 5px)",
        'sc_finalize': "Finalize Overlap",
        'sc_arrange': "2D Mosaic Matrix Layout (L)",
        'sc_undo_redo': "Undo / Redo Action",
        'sc_zoom': "Smooth Zoom Viewport",
        'sc_pan': "Pan Viewport",

        # Status Bar & Dialog Messages
        'status_ready': "Picture Puzzle is Ready.",
        'status_arranged': "Arranged {count} image(s) sequentially with {spacing}px spacing.",
        'status_arranged_matrix': "Arranged {count} mosaic tiles in {rows}x{cols} matrix with {spacing}px gap.",
        'status_loaded': "Loaded {count} X-ray image tiles successfully.",
        'status_dir_not_found': "Directory '{folder}' not found.",
        'status_no_images': "No images found in '{folder}'.",
        'status_undone': "Action Undone.",
        'status_nothing_undo': "Nothing to undo.",
        'status_redone': "Action Redone.",
        'status_nothing_redo': "Nothing to redo.",
        'status_deleted': "Deleted {count} group(s).",
        'status_no_del_selection': "No groups selected to delete.",
        'status_selected_all': "Selected all {count} groups.",
        'status_smart_snap_req': "Select at least 2 groups for Smart Snap.",
        'status_smart_snap_success': "Smart Snap aligned pair successfully.",
        'status_smart_snap_fail': "Smart Snap failed: Insufficient match points.",
        'status_smart_snap_chain': "Smart Snap chain completed ({count} pairs aligned).",
        'status_auto_align_req': "Select at least 2 groups for Auto-Align.",
        'status_auto_align_success': "Auto-Align applied with SuperGlue Homography.",
        'status_auto_align_fail': "Auto-Align failed: Insufficient feature matches.",
        'status_auto_align_chain': "Batch Auto-Align chain completed ({count} pairs aligned).",
        'status_elastic_req': "Select exactly 2 groups for Elastic Mesh Warp.",
        'status_elastic_success': "Elastic Mesh Warp applied successfully with local adaptive deformation.",
        'status_elastic_fail': "Elastic Mesh Warp failed: Insufficient feature matches.",
        'status_stretch_activated': "Stretch Mode Active: Drag any corner or edge handle to fit adjacent mosaic boundaries. Press [Esc] to exit.",
        'status_stretch_done': "Tile boundary stretched and fitted successfully.",
        'status_stretch_reset': "Tile shape restored to original rectangular dimensions.",
        'status_select_one_stretch': "Select exactly 1 tile to stretch corners and edges.",
        'status_overlap_req': "Select exactly 2 groups to adjust overlap.",
        'status_overlap_enter': "Overlap Mode: Drag with mouse to adjust. Press Enter to finalize.",
        'status_overlap_finalized': "Overlap finalized successfully.",
        'status_merge_no_sel': "No pieces selected for merge. Select at least 2 pieces.",
        'status_merge_min2': "Select at least 2 pieces to merge.",
        'status_merge_success': "Pieces merged successfully with {mode} blending.",
        'status_merging': "Merging image pieces in background... Please wait.",
        'status_aligning': "Calculating AI alignment... Please wait.",
        'status_exporting': "Exporting high-quality image and metadata in background... Please wait.",
        'status_export_saved': "High-quality output saved to: {path}",
        'status_project_saving': "Saving project and images in background... Please wait.",
        'dialog_export_title': "Export Successful",
        'dialog_export_msg': "High-quality mosaic and audit package saved to:\n{path}\n\nGenerated Artifacts:\n• mosaic.png (Stitched Mosaic)\n• validity_mask.png (Binary Data Mask)\n• source_index_map.png (Tile Provenance Map)\n• transforms.json (Global Coordinates)\n• report.json (QA & Calibration Manifest)",
        'dialog_select_dir': "Select X-Ray Images Directory",
        'status_lang_changed': "Language switched to English.",
        'status_project_saved': "Project successfully saved to: {path}",
        'status_project_not_found': "Project file not found: {path}",
        'status_project_loaded': "Loaded project with {count} tiles from: {path}",
        'dialog_save_project': "Save Puzzle Project",
        'dialog_load_project': "Open Puzzle Project Manifest",
        'dialog_project_saved_title': "Project Saved",
        'dialog_project_saved_msg': "Puzzle project saved successfully to:\n{path}",
        'btn_load_tb': "Open Project",
        'btn_load_project': "📂 Open Project",

        # Hotkey HUD Overlay
        'hud_header': "HOTKEY TRIGGERED",
        'hud_recent': "Recent Keys: ",
        'hud_see_through_off': "See-Through Overlay: OFF (100% Solid)",
        'hud_see_through_on': "See-Through Overlay: ON (55% Translucent)",
        'hud_undo': "Undo Action",
        'hud_redo': "Redo Action",
        'hud_delete': "Delete Selected",
        'hud_select_all': "Select All",
        'hud_smart_snap': "Smart Snap (Align Pair)",
        'hud_auto_align': "Auto-Align (Homography)",
        'hud_elastic': "Elastic Mesh Warp",
        'hud_stretch_mode': "Edge & Corner Stretch Tool",
        'hud_stretch_applied': "Edge Fit Applied",
        'hud_stretch_reset': "Tile Shape Reset",
        'hud_overlap': "Overlap Adjustment",
        'hud_finalize': "Finalize Overlap",
        'hud_merge': "Merge Pieces (Blend)",
        'hud_reset': "Reset Modes",
        'hud_arrange': "Arrange Images With Gap",
        'hud_save': "Save Project",
        'hud_export': "HQ Export",
        'hud_capture_mode': "Canvas Frame Capture Active",
        'status_capture_mode': "Frame Capture: Drag handles or box to select area. Press [Enter] to Save, [Esc] to Cancel.",
        'dialog_capture_save': "Save Canvas Region Snapshot",
        'hud_micro_nudge': "Micro-Nudge ({dx:+d}px, {dy:+d}px)",
        'hud_piece_nudge': "Nudge Piece ({dx:+d}px, {dy:+d}px)",
    },
    'fa': {
        # Window & Docks
        'window_title': "پازل تصاویر",
        'dock_title': "مرکز کنترل",
        'toolbar_title': "نوار ابزار اصلی",
        'lang_label': "زبان:",
        
        # Toolbar Actions (Compact, Fixed Height, Never Hidden)
        'btn_open_tb': "پوشه",
        'btn_load_tb': "پروژه",
        'btn_save_tb': "ذخیره",
        'btn_export_tb': "خروجی",
        'btn_fit_tb': "تنظیم نما",
        'btn_arrange_tb': "چیدمان [L]",
        'btn_smart_snap_tb': "اسنپ [N]",
        'btn_auto_align_tb': "تراز [G]",
        'btn_elastic_tb': "کشسان [E]",
        'btn_stretch_tb': "کشش [W]",
        'btn_merge_tb': "ادغام [M]",
        'btn_overlap_tb': "هم‌پوشانی دستی",
        'btn_undo_tb': "بازگشت",
        'btn_redo_tb': "مجدد",
        'btn_ghost_tb': "دید شفاف [O]",

        # Group 1: Project & File Operations
        'group_file': "📁 مدیریت پروژه و فایل‌ها",
        'btn_open': "📂 باز کردن پوشه",
        'btn_fit': "🔍 تنظیم اندازه تصویر",
        'btn_arrange': "📐 چیدمان ماتریسی موزاییک (سطر و ستون) [ L ]",
        'tip_arrange': "فیت کردن و چیدمان ماتریسی قطعات موزاییک در سطرها و ستون‌های دقیق با فاصله پیکسلی مشخص بدون چسبیدن بهم",
        'lbl_spacing': "فاصله پیکسلی:",
        'lbl_cols': "ستون‌های ماتریس:",
        'cols_auto': "خودکار",
        'lbl_arrange_mode': "حالت چیدمان:",
        'mode_matrix': "ماتریس موزاییک (سطر و ستون)",
        'mode_row': "یک ردیف افقی",
        'mode_col': "یک ستون عمودی",
        'btn_save': "💾 ذخیره پروژه (S)",
        'btn_export': "🖼️ خروجی باکیفیت (P)",

        # Group 2: Smart Stitching Tools
        'group_tools': "⚡ ابزارهای هوشمند چیدمان",
        'lbl_ai_engines': "🤖 موتورهای هوش مصنوعی و مدل‌های یدکی:",
        'lbl_stitch_mode': "مد پردازش موزاییک:",
        'stitch_mode_auto': "هوشمند خودکار (2D/3D)",
        'stitch_mode_2d': "دوبعدی مسطح (صلب بدون تغییر)",
        'stitch_mode_3d': "سه‌بعدی پرتو مخروطی (آگاه از پارالاکس)",
        'lbl_align_engine': "موتور تراز تصویر:",
        'engine_superglue': "SuperGlue (نقطه‌ای - پیش‌فرض)",
        'engine_dense': "تطبیق متراکم پیکسلی (یدکی هوشمند)",
        'lbl_warp_engine': "موتور دفرماسیون و تغییرشکل:",
        'engine_mls': "توری کشسان MLS (پیش‌فرض)",
        'engine_deformable': "دفرماسیون عصبی دیفئومورفیک (یدکی)",
        'engine_parallax': "دوخت با تحمل اختلاف منظر UDIS2 (یدکی)",
        'chk_global_graph': "بهینه‌سازی سراسری گراف (حذف رانش ماتریس)",
        'btn_smart_snap': " اتصال هوشمند دوتایی [ N ]",
        'tip_smart_snap': "تراز هوشمند دو قطعه مجاور بدون تغییر پرسپکتیو",
        'btn_auto_align': " تراز خودکار هوموگرافی [ G ]",
        'tip_auto_align': "تراز هوشمند با ماتریس هوموگرافی SuperGlue (جفت یا زنجیره‌ای)",
        'btn_elastic': " تراز کشسان محلی [ E ]",
        'tip_elastic': "تغییر شکل توری منعطف (MLS) برای تراز همزمان دو طرف تصویر در پرتوهای مایل",
        'btn_stretch': " دفرمه و کشش اضلاع و خط‌چین [ W ]",
        'tip_stretch': "کشیدن آزادانه هر نقطه از خط‌چین، گوشه‌ها و اضلاع بلوک تصویر برای دفرمه و جفت کردن دقیق روی لبه‌های موزاییک مجاور",
        'btn_reset_stretch': "بازنشانی دفرمه و کشش",
        'tip_reset_stretch': "بازگرداندن قطعه انتخابی به ابعاد و فرم مستطیلی اولیه",
        'btn_merge': "🪢 ادغام قطعات انتخابی [ M ]",
        'tip_merge': "ادغام یکپارچه قطعات انتخاب‌شده با الگوریتم فعال خط‌درز هوشمند یا هرمی فرکانسی [ M ]",
        'btn_overlap': " تنظیم دستی هم‌پوشانی",
        'tip_overlap': "جابجایی و تنظیم زنده هم‌پوشانی قطعات با ماوس قبل از ادغام نهایی",

        # Group 3: Layer Transparency & Overlay Inspection
        'group_inspect': "شفافیت لایه‌ها و بازرسی هم‌پوشانی",
        'lbl_opacity': "شفافیت لایه بالایی: {val}%",
        'btn_toggle_opacity_off': " بازرسی شفاف لایه‌ها [ O ]",
        'btn_toggle_opacity_on': " حالت شفاف فعال است ({val}%)  [ O ]",
        'tip_toggle_opacity': "تغییر شفافیت لایه بالا بین ۵۵٪ (شفاف) و ۱۰۰٪ (کامل) برای بررسی انطباق لایه‌ها",

        # Group 4: X-Ray Cone-Beam Distortion Fix
        'group_xray': "🛡️ تصحیح اعوجاج مخروطی اشعه ایکس",
        'lbl_blend_mode': "الگوریتم خط درز ادغام:",
        'btn_mode_seamline': "برش خط درز هوشمند (بدون شبح)",
        'btn_mode_feather': "محوشدگی نرم",
        'btn_mode_multiband': "ترکیب چندبانده فرکانسی",
        'tip_blend_mode': "برش بهینه خط درز مانع دوبل شدن پیچ و سیم‌ها می‌شود؛ محوشدگی نرم انتقال تدریجی ایجاد می‌کند",
        'chk_feather': "فعال‌سازی ترکیب وزن‌دار مرکزی",
        'tip_feather': "اولویت‌دهی به پرتوهای عمودی مرکز تصویر نسبت به پرتوهای مایل حاشیه",
        'lbl_power': "نرمی محوشدگی حاشیه: {val:.1f}",
        'lbl_edge_bright': "روشنایی درز اتصال: {sign}{val}",
        'tip_edge_bright': "جبران تیرگی و افت نور در لبه‌های درز اتصال قطعات",

        # Group 5: X-Ray Geometric Calibration
        'group_calib': "📐 کالیبراسیون هندسی پرتو ایکس",
        'lbl_calib_status_uncal': "مقیاس: پیکسلی دوبعدی (کالیبره‌نشده)",
        'lbl_calib_status_cal': 'Manual geometry: M={mag:.3f}, pitch={pitch_mm:.5f} mm; metric accuracy unverified',
        'lbl_sdd': "فاصله چشمه تا دتکتور SDD (mm):",
        'tip_sdd': "فاصله منبع پرتو ایکس تا دتکتور بر حسب میلی‌متر (باید بزرگتر از SOD باشد).",
        'lbl_sod': "فاصله چشمه تا شیء SOD (mm):",
        'tip_sod': "فاصله منبع پرتو ایکس تا صفحه شیء بر حسب میلی‌متر.",
        'lbl_pitch': "گام پیکسل دتکتور Pixel Pitch (µm):",
        'tip_pitch': "اندازه فیزیکی هر پیکسل آشکارساز بر حسب میکرومتر (مثلاً 100 یا 150 میکرومتر).",
        'lbl_odd_display': "فاصله شیء تا دتکتور ODD: {val}",
        'lbl_mag_display': "بزرگ‌نمایی صفحه مرجع M₀: {val}",
        'val_unspecified': "نامشخص / خالی",
        'btn_clear_calib': "بازنشانی کالیبراسیون به حالت پیکسلی",
        'btn_cone_rectify': 'Reference-plane mosaic | موزاییک صفحه مرجع',
        'tip_cone_rectify': 'Import measured calibration and absolute per-shot poses; process originals and export coverage. Other depths retain parallax.',
        'dialog_cone_rectify_title': "تصحیح پرتو مخروطی اشعه ایکس",
        'dialog_cone_rectify_req': "لطفاً ابتدا مقادیر معتبر SDD (فاصله چشمه تا دتکتور) و Pixel Pitch (گام پیکسل) را در پنل کالیبراسیون وارد نمایید.",
        'status_cone_rectify_done': "تصحیح شعاعی پرتو مخروطی روی {count} تایل اعمال گردید.",

        # Group 6: Edit & Canvas Selection
        'group_edit': "🎨 ویرایش و انتخاب قطعات",
        'btn_undo': "↩ بازگشت [ Z ]",
        'btn_redo': "↪ تکرار [ Y ]",
        'btn_select_all': "⊞ انتخاب همه [ A ]",
        'btn_del': "🗑 حذف قطعات [ D ]",

        # Group 6: Keyboard Shortcuts Reference
        'group_shortcuts': "⌨️ راهنمای کلیدهای میانبر",
        'sc_smart_snap': "اتصال هوشمند دوتایی",
        'sc_auto_align': "تراز خودکار دسته‌ای / جفتی",
        'sc_elastic': "تراز کشسان محلی (رفع اختلاف منظر)",
        'sc_stretch': "کشش و تنظیم مرزهای بلوک تصویر",
        'sc_transparency': "تغییر شفافیت لایه برای بازرسی",
        'sc_overlap': "حالت تنظیم دستی هم‌پوشانی",
        'sc_nudge': "جابجایی دقیق پیکسلی (۱ پیکسل)",
        'sc_fast_nudge': "جابجایی سریع با شیفت (۵ پیکسل)",
        'sc_finalize': "تأیید نهایی هم‌پوشانی",
        'sc_arrange': "چیدمان ماتریسی موزاییک (سطر و ستون)",
        'sc_undo_redo': "بازگشت / انجام مجدد عملیات",
        'sc_zoom': "بزرگ‌نمایی نمای کاربری",
        'sc_pan': "جابجایی آزاد نمای بوم",

        # Status Bar & Dialog Messages
        'status_ready': "پازل تصاویر آماده به کار است.",
        'status_arranged': "تعداد {count} تصویر به ترتیب و با فاصله {spacing} پیکسل کنار هم چیده شدند.",
        'status_arranged_matrix': "تعداد {count} تصویر موزاییک در ماتریس {rows}×{cols} با فاصله {spacing} پیکسل چیده شدند.",
        'status_loaded': "تعداد {count} تصویر قطعه اشعه ایکس با موفقیت بارگذاری شد.",
        'status_dir_not_found': "پوشه '{folder}' یافت نشد.",
        'status_no_images': "هیچ تصویری در پوشه '{folder}' پیدا نشد.",
        'status_undone': "عملیات به عقب برگشت.",
        'status_nothing_undo': "عملیاتی برای بازگشت وجود ندارد.",
        'status_redone': "عملیات دوباره انجام شد.",
        'status_nothing_redo': "عملیاتی برای انجام مجدد وجود ندارد.",
        'status_deleted': "{count} گروه حذف شد.",
        'status_no_del_selection': "هیچ گروهی برای حذف انتخاب نشده است.",
        'status_selected_all': "تمامی {count} گروه انتخاب شدند.",
        'status_smart_snap_req': "حداقل ۲ گروه را برای اتصال هوشمند انتخاب کنید.",
        'status_smart_snap_success': "اتصال هوشمند جفت با موفقیت انجام شد.",
        'status_smart_snap_fail': "اتصال ناموفق: نقاط تطابق کافی یافت نشد.",
        'status_smart_snap_chain': "زنجیره اتصال هوشمند کامل شد ({count} جفت متصل شدند).",
        'status_auto_align_req': "حداقل ۲ گروه را برای تراز خودکار انتخاب کنید.",
        'status_auto_align_success': "تراز خودکار با ماتریس هوموگرافی اعمال شد.",
        'status_auto_align_fail': "تراز خودکار ناموفق: ویژگی‌های مشترک کافی پیدا نشد.",
        'status_auto_align_chain': "زنجیره تراز خودکار کامل شد ({count} جفت تراز شدند).",
        'status_elastic_req': "دقیقاً ۲ گروه را برای تراز کشسان انتخاب کنید.",
        'status_elastic_success': "تراز کشسان محلی با موفقیت اعمال شد.",
        'status_elastic_fail': "تراز کشسان ناموفق: نقاط تطابق کافی یافت نشد.",
        'status_stretch_activated': "حالت دفرمه و کشش اضلاع فعال شد: هر نقطه از خط‌چین یا گوشه‌ها را بکشید تا روی لبه قطعه مجاور جفت شود. [Esc] برای خروج.",
        'status_stretch_done': "لبه و مرزهای قطعه با موفقیت دفرمه و منطبق شد.",
        'status_stretch_reset': "فرم قطعه به ابعاد اولیه بازنشانی شد.",
        'status_select_one_stretch': "دقیقاً ۱ قطعه را برای دفرمه و کشیدن خط‌چین و اضلاع انتخاب کنید.",
        'status_overlap_req': "دقیقاً ۲ گروه را برای تنظیم هم‌پوشانی انتخاب کنید.",
        'status_overlap_enter': "حالت هم‌پوشانی: با ماوس جابجا کنید. اینتر را برای ثبت بزنید.",
        'status_overlap_finalized': "هم‌پوشانی با موفقیت ثبت شد.",
        'status_merge_no_sel': "هیچ قطعه‌ای برای ادغام انتخاب نشده است. لطفاً حداقل ۲ قطعه را انتخاب کنید.",
        'status_merge_min2': "برای ادغام باید حداقل ۲ قطعه انتخاب شوند.",
        'status_merge_success': "قطعات با موفقیت با الگوریتم {mode} ادغام شدند.",
        'status_merging': "در حال ادغام قطعات تصویر در پس‌زمینه... لطفاً صبر کنید.",
        'status_aligning': "در حال تراز هوشمند قطعات تصویر... لطفاً صبر کنید.",
        'status_exporting': "در حال تهیه و ذخیره خروجی باکیفیت در پس‌زمینه... لطفاً صبر کنید.",
        'status_export_saved': "تصویر خروجی باکیفیت ذخیره شد در: {path}",
        'status_project_saving': "در حال ذخیره پروژه و تصاویر در پس‌زمینه... لطفاً صبر کنید.",
        'dialog_export_title': "خروجی موفقیت‌آمیز",
        'dialog_export_msg': "بسته خروجی استاندارد و مانیفست تحلیلی در مسیر زیر ذخیره شد:\n{path}\n\nفایل‌های تولیدشده:\n• mosaic.png (تصویر نهایی دوخته‌شده)\n• validity_mask.png (ماسک باینری داده معتبر)\n• source_index_map.png (نقشه منشأ قطعات)\n• transforms.json (مختصات و تبدیل‌ها)\n• report.json (گزارش جامع اعتبارسنجی)",
        'dialog_select_dir': "انتخاب پوشه تصاویر اشعه ایکس",
        'status_lang_changed': "زبان نرم‌افزار به فارسی تغییر کرد.",
        'status_project_saved': "پروژه با موفقیت در مسیر زیر ذخیره شد:\n{path}",
        'status_project_not_found': "فایل پروژه یافت نشد: {path}",
        'status_project_loaded': "پروژه با {count} قطعه از مسیر زیر بارگذاری شد:\n{path}",
        'dialog_save_project': "ذخیره پروژه پازل",
        'dialog_load_project': "باز کردن مانیفست پروژه پازل",
        'dialog_project_saved_title': "ذخیره موفق پروژه",
        'dialog_project_saved_msg': "پروژه پازل و موقعیت قطعات با موفقیت ذخیره شد در:\n{path}",
        'btn_load_tb': "باز کردن پروژه",
        'btn_load_project': "📂 باز کردن پروژه",

        # Hotkey HUD Overlay
        'hud_header': "کلید میانبر فعال شد",
        'hud_recent': "کلیدهای اخیر: ",
        'hud_see_through_off': "دید شفاف: غیرفعال (۱۰۰٪ کدر)",
        'hud_see_through_on': "دید شفاف: فعال (۵۵٪ شفاف)",
        'hud_undo': "بازگشت عملیات",
        'hud_redo': "تکرار عملیات",
        'hud_delete': "حذف انتخاب‌شده‌ها",
        'hud_select_all': "انتخاب همه",
        'hud_smart_snap': "اتصال هوشمند (جفت)",
        'hud_auto_align': "تراز خودکار (هوموگرافی)",
        'hud_elastic': "تراز کشسان محلی",
        'hud_stretch_mode': "دفرمه و کشش اضلاع",
        'hud_stretch_applied': "دفرمه و انطباق لبه اعمال شد",
        'hud_stretch_reset': "فرم اولیه بازنشانی شد",
        'hud_overlap': "تنظیم هم‌پوشانی",
        'hud_finalize': "تأیید نهایی هم‌پوشانی",
        'hud_merge': "ادغام قطعات (ترکیب لایه‌ها)",
        'hud_reset': "بازنشانی حالت‌ها",
        'hud_arrange': "چیدمان ماتریسی موزاییک اعمال شد",
        'hud_save': "ذخیره پروژه",
        'hud_export': "خروجی باکیفیت",
        'hud_capture_mode': "حالت کادر عکس‌برداری فعال شد",
        'status_capture_mode': "کادر عکس‌برداری: کادر را روی ناحیه مورد نظر تنظیم کنید و کلید Enter را برای ذخیره بزنید (Esc برای انصراف).",
        'dialog_capture_save': "ذخیره عکس ناحیه انتخابی بوم",
        'hud_micro_nudge': "جابجایی دقیق ({dx:+d}px, {dy:+d}px)",
        'hud_piece_nudge': "جابجایی قطعه ({dx:+d}px, {dy:+d}px)",
    }
}


# ================== Interactive Canvas Widget ==================
class InteractivePuzzleCanvas(QWidget):
    statusChanged = pyqtSignal(str)
    selectionChanged = pyqtSignal(list)
    actionTriggered = pyqtSignal(str, str)
    opacityChanged = pyqtSignal(float)

    STATE_NORMAL = 0
    STATE_MANUAL_SNAP_SELECT_EDGE1 = 1
    STATE_MANUAL_SNAP_SELECT_EDGE2 = 2
    STATE_OVERLAP_ADJUST = 3
    STATE_STRETCH_WARP = 4
    STATE_CAPTURE_REGION = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #F1F4F9;")

        self.pieces = []
        self.next_piece_id = 0
        self.selection = []
        self.history = []
        self.redo_history = []
        self.current_state = self.STATE_NORMAL

        # Viewport parameters
        self.scale = 1.0
        self.view_offset_x = 0.0
        self.view_offset_y = 0.0

        # Mouse interaction
        self.is_dragging = False
        self.is_panning = False
        self.drag_start_pos = QPointF(0, 0)
        self.last_mouse_pos = QPointF(0, 0)
        self.hovered_piece_id = -1
        self.hovered_edge_info = None
        self.manual_snap_selection_1 = None

        # Overlap Adjustment parameters
        self.overlap_info = None
        self.overlap_chain = None
        self.temp_preview = None
        self.overlap_alpha = 1.0
        self.temp_preview_pixmap = None
        self._is_updating_preview = False
        self.overlap_preview_timer = QTimer(self)
        self.overlap_preview_timer.setSingleShot(True)
        self.overlap_preview_timer.setInterval(70)
        self.overlap_preview_timer.timeout.connect(self._update_preview_from_current_pos)

        # Interactive Multi-Point Elastic Boundary Stretch & Fit Tool
        self.stretch_active_piece = None
        self.stretch_active_handle = -1  # 0..19: 20 perimeter control handles
        self.stretch_drag_start = None
        self.stretch_orig_boundary = None
        self.stretch_cur_boundary = None
        self.stretch_orig_corners = None
        self.stretch_cur_corners = None
        self.stretch_snap_info = None

        # Action 1: Seamline Cut vs Feathering Blend Mode
        self.blend_mode = 'seamline'

        # Interactive Region Crop & Frame Snapshot Tool
        self.capture_rect_world = None
        self.capture_active_handle = None
        self.capture_drag_start = None

        # X-ray perspective & blending settings
        self.feather_power = 2.0
        self.enable_feathering = True

        # Pixmap rendering cache
        self.pixmap_cache = {}

        # Canvas Hotkey HUD System
        self.hud_key = None
        self.hud_label = ""
        self.hud_timestamp = None
        self.hud_history = []
        self.hud_alpha = 1.0

        self.hud_timer = QTimer(self)
        self.hud_timer.setInterval(25)
        self.hud_timer.timeout.connect(self._update_hud_fade)
        self.hud_timer.start()

        # Alignment Inspector: Opacity of top selected piece
        self.selection_opacity = 1.0
        self.edge_brightness = 0.0
        self.current_lang = 'fa'
        self.is_busy = False
        self.active_workers = set()

        # AI & Backup Engines Settings (موتورهای محاسباتی یدکی و پیشرفته)
        self.align_engine = 'superglue'   # 'superglue' (Default) or 'dense' (Backup / Dense Feature)
        self.warp_engine = 'elastic_mls'  # 'elastic_mls' (Default), 'deformable_dir' (Neural Diffeomorphic), or 'parallax_udis' (Parallax-Tolerant)
        self.use_global_graph = True      # Enable global least-squares pose graph solve for 2D matrix
        self.stitch_mode = 'auto'         # 'auto' (Smart 2D/3D), '2d' (Strict Planar), or '3d' (Cone-Beam Parallax-Aware)

    def set_stitch_mode(self, mode):
        self.stitch_mode = mode
        mode_map = {'auto': 'stitch_mode_auto', '2d': 'stitch_mode_2d', '3d': 'stitch_mode_3d'}
        tr_key = mode_map.get(mode, 'stitch_mode_auto')
        name = self.tr(tr_key)
        msg = f"مد پردازش تصویر: {name}" if self.current_lang == 'fa' else f"Stitch Mode: {name}"
        self.statusChanged.emit(msg)

    def set_align_engine(self, engine):
        self.align_engine = engine
        name = self.tr('engine_' + engine) if engine in ('superglue', 'dense') else engine
        msg = f"موتور تراز هوشمند: {name}" if self.current_lang == 'fa' else f"AI Align Engine: {name}"
        self.statusChanged.emit(msg)

    def set_warp_engine(self, engine):
        self.warp_engine = engine
        engine_map = {'elastic_mls': 'engine_mls', 'deformable_dir': 'engine_deformable', 'parallax_udis': 'engine_parallax'}
        tr_key = engine_map.get(engine, '')
        name = self.tr(tr_key) if tr_key else engine
        msg = f"موتور دفرماسیون هوشمند: {name}" if self.current_lang == 'fa' else f"AI Warp Engine: {name}"
        self.statusChanged.emit(msg)

    def set_use_global_graph(self, enabled):
        self.use_global_graph = bool(enabled)

    def set_blend_mode(self, mode):
        self.blend_mode = mode
        if self.current_state == self.STATE_OVERLAP_ADJUST:
            self._update_preview_from_current_pos()
        self.update()

    def toggle_stretch_mode(self):
        if self.current_state == self.STATE_STRETCH_WARP:
            self._reset_modes()
            self.statusChanged.emit(self.tr('status_ready'))
        else:
            if len(self.selection) != 1:
                if len(self.pieces) > 0 and len(self.selection) == 0:
                    self.selection = [self.pieces[0].group_id]
                    for p in self.pieces:
                        p.is_selected = (p.group_id == self.selection[0])
                    self.selectionChanged.emit(self.selection)
                elif len(self.selection) > 1:
                    self.statusChanged.emit(self.tr('status_select_one_stretch'))
                    return
                elif len(self.pieces) == 0:
                    return

            self._reset_modes()
            self.current_state = self.STATE_STRETCH_WARP
            sel_piece = next((p for p in self.pieces if p.group_id in self.selection), None)
            self.stretch_active_piece = sel_piece
            self.stretch_active_handle = -1
            self.stretch_cur_boundary = None
            self.stretch_orig_boundary = None
            self.stretch_cur_corners = None
            self.stretch_orig_corners = None
            self.stretch_snap_info = None
            self.trigger_hotkey_hud('W', self.tr('hud_stretch_mode'))
            self.statusChanged.emit(self.tr('status_stretch_activated'))
        self.update()

    def reset_piece_stretch(self):
        sel_pieces = [p for p in self.pieces if p.group_id in self.selection]
        if not sel_pieces and self.stretch_active_piece:
            sel_pieces = [self.stretch_active_piece]
        if not sel_pieces:
            return
        self.save_state()
        for p in sel_pieces:
            orig = p.original_image_ref
            p.image = orig.copy()
            p.w = orig.shape[1]
            p.h = orig.shape[0]
            p.sync_corners_to_rect()
            self.pixmap_cache[p.id] = cv_mat_to_qpixmap(p.image)
        self.stretch_cur_boundary = None
        self.stretch_orig_boundary = None
        self.stretch_cur_corners = None
        self.stretch_orig_corners = None
        self.stretch_snap_info = None
        self.trigger_hotkey_hud('W', self.tr('hud_stretch_reset'))
        self.statusChanged.emit(self.tr('status_stretch_reset'))
        self.update()

    def _hit_test_stretch_handles(self, piece, wx, wy, insert_if_seg_hit=False):
        """
        Hit tests boundary control pins and dashed edge segments.
        1. If clicking within tolerance of an existing pin, returns (pin_index, pin_pos).
        2. If hovering or clicking on the dashed boundary line between pins:
           - if insert_if_seg_hit is True: dynamically inserts a new pin at that exact
             (wx, wy) pixel coordinate and returns (new_pin_index, (wx, wy)).
           - if insert_if_seg_hit is False: returns (best_seg, (proj_x, proj_y)) without modifying data.
        Returns: (handle_index, (hx, hy))
        """
        if piece is None:
            return -1, None
        boundary = piece.get_boundary_world()
        N = len(boundary)
        thresh = 11.0 / max(0.08, self.scale)

        # 1. Direct hit on any existing perimeter pin
        best_node = -1
        min_node_dist = thresh
        for i in range(N):
            d = math.hypot(wx - boundary[i, 0], wy - boundary[i, 1])
            if d < min_node_dist:
                min_node_dist = d
                best_node = i
        if best_node >= 0:
            return best_node, (boundary[best_node, 0], boundary[best_node, 1])

        # 2. Hit on the dashed edge segment between pins -> Dynamic Pin Insertion at exact pixel!
        seg_thresh = thresh + 4.0
        best_seg = -1
        best_t = 0.5
        best_proj = None
        min_seg_dist = seg_thresh

        for i in range(N):
            p1 = boundary[i]
            p2 = boundary[(i + 1) % N]
            vx = p2[0] - p1[0]
            vy = p2[1] - p1[1]
            l2 = vx * vx + vy * vy
            if l2 < 1e-4:
                continue
            t = ((wx - p1[0]) * vx + (wy - p1[1]) * vy) / l2
            if 0.02 <= t <= 0.98:
                proj_x = p1[0] + t * vx
                proj_y = p1[1] + t * vy
                d = math.hypot(wx - proj_x, wy - proj_y)
                if d < min_seg_dist:
                    min_seg_dist = d
                    best_seg = i
                    best_t = t
                    best_proj = (proj_x, proj_y)

        if best_seg >= 0:
            if not insert_if_seg_hit:
                # Hover feedback only - do NOT modify data structure
                return best_seg, best_proj

            # Insert a new pin right at this exact coordinate!
            insert_idx = best_seg + 1
            new_world_pt = np.array([wx, wy], dtype=np.float32)

            orig = piece.original_image_ref
            orig_h, orig_w = orig.shape[:2]
            if not hasattr(piece, 'boundary_src') or piece.boundary_src is None or len(piece.boundary_src) != N:
                piece.boundary_src = PuzzlePiece._calc_rect_boundary(0, 0, orig_w, orig_h, num_per_edge=max(1, N // 4))

            s1 = piece.boundary_src[best_seg]
            s2 = piece.boundary_src[(best_seg + 1) % N]
            new_src_pt = (1.0 - best_t) * s1 + best_t * s2

            piece.boundary_world = np.insert(piece.boundary_world, insert_idx, new_world_pt, axis=0)
            piece.boundary_src = np.insert(piece.boundary_src, insert_idx, new_src_pt, axis=0)

            # Update corner indices
            c_indices = getattr(piece, 'corner_indices', [0, 20, 40, 60])
            piece.corner_indices = [c + 1 if c >= insert_idx else c for c in c_indices]

            return insert_idx, (wx, wy)

        return -1, None

    def _find_nearest_pin(self, piece, wx, wy):
        if piece is None:
            return -1
        boundary = piece.get_boundary_world()
        thresh = 12.0 / max(0.08, self.scale)
        best_node = -1
        min_dist = thresh
        for i in range(len(boundary)):
            d = math.hypot(wx - boundary[i, 0], wy - boundary[i, 1])
            if d < min_dist:
                min_dist = d
                best_node = i
        return best_node

    def _apply_magnetic_snap(self, boundary, active_handle, active_piece):
        """
        Magnetically snaps dragged boundary node to adjacent mosaic tiles within tolerance.
        Snaps ONLY the active handle. All other pins remain 100% FIXED and CONSTANT.
        """
        snap_thresh = 16.0 / max(0.08, self.scale)
        other_pieces = [p for p in self.pieces if p.id != active_piece.id]
        if not other_pieces or active_handle < 0:
            return boundary, None

        cx, cy = boundary[active_handle, 0], boundary[active_handle, 1]
        best_dist = snap_thresh
        best_pt = None

        for op in other_pieces:
            op_boundary = op.get_boundary_world()
            for oc in op_boundary:
                d = math.hypot(cx - oc[0], cy - oc[1])
                if d < best_dist:
                    best_dist = d
                    best_pt = (float(oc[0]), float(oc[1]))

        if best_pt is not None:
            # Snap strictly the active handle only - zero neighbor displacement!
            boundary[active_handle, 0] = best_pt[0]
            boundary[active_handle, 1] = best_pt[1]
            return boundary, ('point', best_pt)

        return boundary, None

    def _apply_stretch_warp(self, piece, deformed_boundary_world):
        """
        Applies real-time 2D MLS Affine elastic mesh deformation to the puzzle piece.
        Always samples from original_image_ref to preserve 100% crystal resolution.
        Clips exactly to the deformed boundary polygon with zero exterior bleed.
        """
        min_x = float(np.min(deformed_boundary_world[:, 0]))
        min_y = float(np.min(deformed_boundary_world[:, 1]))
        max_x = float(np.max(deformed_boundary_world[:, 0]))
        max_y = float(np.max(deformed_boundary_world[:, 1]))

        new_w = max(16, int(round(max_x - min_x)))
        new_h = max(16, int(round(max_y - min_y)))

        # Local destination boundary points
        boundary_dst_local = deformed_boundary_world.copy()
        boundary_dst_local[:, 0] -= min_x
        boundary_dst_local[:, 1] -= min_y

        orig = piece.original_image_ref
        orig_h, orig_w = orig.shape[:2]
        N = len(deformed_boundary_world)
        if not hasattr(piece, 'boundary_src') or piece.boundary_src is None or len(piece.boundary_src) != N:
            piece.boundary_src = PuzzlePiece._calc_rect_boundary(0, 0, orig_w, orig_h, num_per_edge=max(1, N // 4))
        boundary_src = piece.boundary_src

        # Corner indices
        corner_indices = getattr(piece, 'corner_indices', [0, 20, 40, 60])
        c_tl = boundary_dst_local[corner_indices[0]]
        c_tr = boundary_dst_local[corner_indices[1]]
        c_br = boundary_dst_local[corner_indices[2]]
        c_bl = boundary_dst_local[corner_indices[3]]

        # نگه‌داشتن لنگرهای داخلی در موقعیت صلب اولیه نسبت به مرکز تایل
        # تا کشیدن لبه‌ها و گوشه‌ها باعث دفرمه شدن قطعات داخل موزاییک نشود
        orig_x_offset = piece.x - min_x
        orig_y_offset = piece.y - min_y
        int_src = []
        int_dst = []
        for u in (0.25, 0.5, 0.75):
            for v in (0.25, 0.5, 0.75):
                int_src.append([u * orig_w, v * orig_h])
                int_dst.append([orig_x_offset + u * piece.w, orig_y_offset + v * piece.h])

        src_pts_all = np.vstack([boundary_src, np.array(int_src, dtype=np.float32)])
        dst_pts_all = np.vstack([boundary_dst_local, np.array(int_dst, dtype=np.float32)])

        try:
            warped = compute_mls_affine_warp(
                src_pts_all,
                dst_pts_all,
                orig,
                (new_h, new_w),
                boundary_dst_local
            )

            self.save_state()
            piece.image = warped
            piece.x = min_x
            piece.y = min_y
            piece.w = new_w
            piece.h = new_h
            piece.boundary_world = deformed_boundary_world.copy()
            piece.corners_world = deformed_boundary_world[corner_indices].copy()
            self.pixmap_cache[piece.id] = cv_mat_to_qpixmap(piece.image)
            self.trigger_hotkey_hud('W', self.tr('hud_stretch_applied'))
            self.statusChanged.emit(self.tr('status_stretch_done'))
        except Exception as e:
            print(f"Stretch Warp elastic mesh error: {e}")


    def run_in_background(self, task_fn, on_success, on_error=None, status_msg=""):
        if self.is_busy:
            return
        self.is_busy = True
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        if status_msg:
            self.statusChanged.emit(status_msg)

        worker = ProcessingWorker(task_fn)
        self.active_workers.add(worker)

        def _on_done(result):
            QApplication.restoreOverrideCursor()
            self.is_busy = False
            on_success(result)
            self.update()

        def _on_err(err_msg):
            QApplication.restoreOverrideCursor()
            self.is_busy = False
            if on_error:
                on_error(err_msg)
            else:
                self.statusChanged.emit(f"Error: {err_msg}")
            self.update()

        def _cleanup():
            self.active_workers.discard(worker)
            worker.deleteLater()

        worker.result_ready.connect(_on_done)
        worker.error_occurred.connect(_on_err)
        worker.finished.connect(_cleanup)
        worker.start()

    def tr(self, key):
        return TRANSLATIONS.get(self.current_lang, TRANSLATIONS['en']).get(key, key)

    def set_language(self, lang):
        self.current_lang = lang
        self.update()

    def set_edge_brightness(self, val):
        self.edge_brightness = float(val)
        if self.current_state == self.STATE_OVERLAP_ADJUST:
            self._update_preview_from_current_pos()
        self.update()

    def set_selection_opacity(self, opacity):
        self.selection_opacity = float(opacity)
        self.overlap_alpha = float(opacity)
        self.opacityChanged.emit(self.selection_opacity)
        self.update()

    def toggle_ghost_view(self):
        if self.selection_opacity < 0.99:
            self.selection_opacity = 1.0
            self.trigger_hotkey_hud('O', self.tr('hud_see_through_off'))
        else:
            self.selection_opacity = 0.55
            self.trigger_hotkey_hud('O', self.tr('hud_see_through_on'))
        self.overlap_alpha = self.selection_opacity
        self.opacityChanged.emit(self.selection_opacity)
        self.update()

    def set_feather_settings(self, power, enabled):
        self.feather_power = power
        self.enable_feathering = enabled
        if self.current_state == self.STATE_OVERLAP_ADJUST:
            self._update_preview_from_current_pos()
        self.update()

    def trigger_hotkey_hud(self, key_str, action_name):
        now = datetime.datetime.now()
        self.hud_key = key_str.upper()
        self.hud_label = action_name
        self.hud_timestamp = now
        self.hud_alpha = 1.0
        self.hud_history.append((self.hud_key, now))
        if len(self.hud_history) > 5:
            self.hud_history.pop(0)
        self.actionTriggered.emit(self.hud_key, self.hud_label)
        self.update()

    def _update_hud_fade(self):
        if not self.hud_timestamp:
            return
        dt = (datetime.datetime.now() - self.hud_timestamp).total_seconds()
        if dt > 1.3:
            rem = 2.4 - dt
            if rem <= 0:
                self.hud_alpha = 0.0
                self.hud_key = None
                self.hud_timestamp = None
            else:
                self.hud_alpha = max(0.0, rem / 1.1)
            self.update()

    def world_to_view(self, wx, wy):
        return (wx - self.view_offset_x) * self.scale, (wy - self.view_offset_y) * self.scale

    def view_to_world(self, vx, vy):
        return (vx / self.scale) + self.view_offset_x, (vy / self.scale) + self.view_offset_y

    def get_piece_pixmap(self, piece):
        if piece.id not in self.pixmap_cache:
            self.pixmap_cache[piece.id] = cv_mat_to_qpixmap(piece.image)
        return self.pixmap_cache[piece.id]

    def save_state(self):
        self.history.append(copy.deepcopy(self.pieces))
        self.redo_history.clear()

    def undo(self):
        self.trigger_hotkey_hud('Z', self.tr('hud_undo'))
        if len(self.history) > 1:
            self.redo_history.append(self.history.pop())
            self.pieces = copy.deepcopy(self.history[-1])
            self.pixmap_cache.clear()
            self._reset_modes()
            self.statusChanged.emit(self.tr('status_undone'))
            self.update()
        else:
            self.statusChanged.emit(self.tr('status_nothing_undo'))

    def redo(self):
        self.trigger_hotkey_hud('Y', self.tr('hud_redo'))
        if self.redo_history:
            state = self.redo_history.pop()
            self.history.append(copy.deepcopy(state))
            self.pieces = copy.deepcopy(state)
            self.pixmap_cache.clear()
            self._reset_modes()
            self.statusChanged.emit(self.tr('status_redone'))
            self.update()
        else:
            self.statusChanged.emit(self.tr('status_nothing_redo'))

    def delete_selected(self):
        self.trigger_hotkey_hud('D', self.tr('hud_delete'))
        if not self.selection:
            self.statusChanged.emit(self.tr('status_no_del_selection'))
            return
        self.save_state()
        self.pieces = [p for p in self.pieces if p.group_id not in self.selection]
        count = len(self.selection)
        self.selection.clear()
        self.pixmap_cache.clear()
        self.statusChanged.emit(self.tr('status_deleted').format(count=count))
        self.selectionChanged.emit(self.selection)
        self.update()

    def select_all(self):
        self.trigger_hotkey_hud('A', self.tr('hud_select_all'))
        all_groups = list({p.group_id for p in self.pieces})
        self.selection = all_groups
        for p in self.pieces:
            p.is_selected = True
        self.statusChanged.emit(self.tr('status_selected_all').format(count=len(all_groups)))
        self.selectionChanged.emit(self.selection)
        self.update()

    def deselect_all(self):
        self.selection.clear()
        for p in self.pieces:
            p.is_selected = False
        self.selectionChanged.emit(self.selection)
        self.update()

    def fit_view(self):
        if not self.pieces:
            return
        min_x = min(p.x for p in self.pieces)
        min_y = min(p.y for p in self.pieces)
        max_x = max(p.x + p.w for p in self.pieces)
        max_y = max(p.y + p.h for p in self.pieces)
        pw = max_x - min_x + 120
        ph = max_y - min_y + 120
        if pw <= 0 or ph <= 0:
            return
        sx = self.width() / pw
        sy = self.height() / ph
        self.scale = max(0.1, min(2.5, min(sx, sy)))
        self.view_offset_x = min_x - 60
        self.view_offset_y = min_y - 60
        self.update()

    def arrange_pieces(self, spacing=40, cols=0, mode='matrix'):
        """
        Arranges pieces in an aligned 2D Mosaic Matrix (or row/column) with exact proportional spacing.
        If pieces are selected, arranges the selected pieces. Otherwise arranges all pieces.
        Columns and rows are aligned with uniform proportional gap in both dimensions.
        """
        if not self.pieces:
            return

        self.save_state()

        gap = float(spacing)

        # 1. Target pieces: selected pieces if any, otherwise all pieces
        if self.selection:
            target_pieces = [p for p in self.pieces if p.group_id in self.selection]
            if not target_pieces:
                target_pieces = self.pieces
        else:
            target_pieces = self.pieces

        n = len(target_pieces)
        if n == 0:
            return

        # 2. Detect 2D grid matrix cells
        num_rows, num_cols, piece_cells = detect_2d_grid_cells(target_pieces, cols_hint=cols, mode=mode)

        # 3. Calculate maximum width per column and maximum height per row
        col_widths = [0.0] * num_cols
        row_heights = [0.0] * num_rows

        for r, c, p in piece_cells:
            if c < num_cols and p.w > col_widths[c]:
                col_widths[c] = float(p.w)
            if r < num_rows and p.h > row_heights[r]:
                row_heights[r] = float(p.h)

        # 4. Compute starting (X, Y) coordinates: keep near current location
        start_x = min(p.x for p in target_pieces)
        start_y = min(p.y for p in target_pieces)
        if start_x < 50.0:
            start_x = 50.0
        if start_y < 50.0:
            start_y = 50.0

        col_x = [0.0] * num_cols
        cur_x = start_x
        for c in range(num_cols):
            col_x[c] = cur_x
            cur_x += col_widths[c] + gap

        row_y = [0.0] * num_rows
        cur_y = start_y
        for r in range(num_rows):
            row_y[r] = cur_y
            cur_y += row_heights[r] + gap

        # 5. Position each piece exactly in its matrix cell (translating its entire group)
        moved_groups = set()
        for r, c, p in piece_cells:
            if p.group_id in moved_groups:
                continue
            target_x = col_x[c]
            target_y = row_y[r]
            dx = target_x - p.x
            dy = target_y - p.y
            for piece in self.pieces:
                if piece.group_id == p.group_id:
                    piece.translate(dx, dy)
            moved_groups.add(p.group_id)

        self.fit_view()
        self.trigger_hotkey_hud('L', self.tr('hud_arrange'))
        self.statusChanged.emit(
            self.tr('status_arranged_matrix').format(
                count=n, rows=num_rows, cols=num_cols, spacing=int(gap)
            )
        )
        self.update()

    def _reset_modes(self):
        if hasattr(self, 'overlap_preview_timer'):
            self.overlap_preview_timer.stop()
        self.is_dragging = False
        self.current_state = self.STATE_NORMAL
        self.hovered_edge_info = None
        self.manual_snap_selection_1 = None
        self.overlap_info = None
        self.temp_preview = None
        self.temp_preview_pixmap = None
        self.stretch_active_piece = None
        self.stretch_active_handle = -1
        self.stretch_drag_start = None
        self.stretch_orig_boundary = None
        self.stretch_cur_boundary = None
        self.stretch_orig_corners = None
        self.stretch_cur_corners = None
        self.stretch_snap_info = None
        self.capture_rect_world = None
        self.capture_active_handle = None
        self.capture_drag_start = None
        self.unsetCursor()
    def start_region_capture(self):
        if not self.pieces:
            return
        self.current_state = self.STATE_CAPTURE_REGION
        self.is_dragging = False
        self.capture_active_handle = None
        self.capture_drag_start = None

        # Pre-set bounding box around all puzzle pieces on canvas
        min_x = min(p.x for p in self.pieces)
        min_y = min(p.y for p in self.pieces)
        max_x = max(p.x + p.w for p in self.pieces)
        max_y = max(p.y + p.h for p in self.pieces)
        self.capture_rect_world = QRectF(min_x, min_y, max(10.0, max_x - min_x), max(10.0, max_y - min_y))

        self.trigger_hotkey_hud('P', self.tr('hud_capture_mode'))
        self.statusChanged.emit(self.tr('status_capture_mode'))
        self.update()

    def _hit_test_capture_handles(self, vx, vy):
        if self.capture_rect_world is None:
            return 'new'

        # Check button clicks on the floating action pill first
        if hasattr(self, '_capture_confirm_rect') and self._capture_confirm_rect and self._capture_confirm_rect.contains(vx, vy):
            return 'btn_confirm'
        if hasattr(self, '_capture_all_rect') and self._capture_all_rect and self._capture_all_rect.contains(vx, vy):
            return 'btn_all'
        if hasattr(self, '_capture_cancel_rect') and self._capture_cancel_rect and self._capture_cancel_rect.contains(vx, vy):
            return 'btn_cancel'

        r = self.capture_rect_world.normalized()
        vx1, vy1 = self.world_to_view(r.x(), r.y())
        vx2, vy2 = self.world_to_view(r.x() + r.width(), r.y() + r.height())
        rx = min(vx1, vx2)
        ry = min(vy1, vy2)
        rw = abs(vx2 - vx1)
        rh = abs(vy2 - vy1)

        tol = 12.0
        # 4 Corners
        if abs(vx - rx) <= tol and abs(vy - ry) <= tol:
            return 'tl'
        if abs(vx - (rx + rw)) <= tol and abs(vy - ry) <= tol:
            return 'tr'
        if abs(vx - rx) <= tol and abs(vy - (ry + rh)) <= tol:
            return 'bl'
        if abs(vx - (rx + rw)) <= tol and abs(vy - (ry + rh)) <= tol:
            return 'br'

        # 4 Edges
        if abs(vy - ry) <= tol and rx - tol <= vx <= rx + rw + tol:
            return 'top'
        if abs(vy - (ry + rh)) <= tol and rx - tol <= vx <= rx + rw + tol:
            return 'bottom'
        if abs(vx - rx) <= tol and ry - tol <= vy <= ry + rh + tol:
            return 'left'
        if abs(vx - (rx + rw)) <= tol and ry - tol <= vy <= ry + rh + tol:
            return 'right'

        # Inside box
        if rx < vx < rx + rw and ry < vy < ry + rh:
            return 'move'

        return 'new'

    def select_all_pieces_capture(self):
        if not self.pieces:
            return
        min_x = min(p.x for p in self.pieces)
        min_y = min(p.y for p in self.pieces)
        max_x = max(p.x + p.w for p in self.pieces)
        max_y = max(p.y + p.h for p in self.pieces)
        self.capture_rect_world = QRectF(min_x, min_y, max(10.0, max_x - min_x), max(10.0, max_y - min_y))
        self.update()

    def confirm_region_capture(self):
        if not self.pieces:
            self.current_state = self.STATE_NORMAL
            self.update()
            return

        if self.capture_rect_world is None or self.capture_rect_world.width() < 4 or self.capture_rect_world.height() < 4:
            min_x = min(p.x for p in self.pieces)
            min_y = min(p.y for p in self.pieces)
            max_x = max(p.x + p.w for p in self.pieces)
            max_y = max(p.y + p.h for p in self.pieces)
            self.capture_rect_world = QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

        r = self.capture_rect_world.normalized()
        rx = int(round(r.x()))
        ry = int(round(r.y()))
        rw = max(16, int(round(r.width())))
        rh = max(16, int(round(r.height())))

        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        # Determine base name from input filenames or current project
        win = self.window()
        base_name = getattr(win, 'current_project_name', None) if win else getattr(self, 'current_project_name', None)
        if not base_name or base_name in ("xray_project", "mosaic_export", "."):
            fns = [os.path.splitext(os.path.basename(p.filename))[0] for p in self.pieces if getattr(p, 'filename', None)]
            if fns:
                common = os.path.commonprefix(fns).rstrip('_- ')
                base_name = common if len(common) >= 3 else fns[0]
        if not base_name:
            base_name = f"snapshot_{ts}"

        default_filename = os.path.join(OUTPUT_SAVE_DIRECTORY, f"{base_name}.png")
        os.makedirs(OUTPUT_SAVE_DIRECTORY, exist_ok=True)

        chosen_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            self.tr('dialog_capture_save'),
            default_filename,
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;TIFF Image (*.tif *.tiff);;Bitmap (*.bmp);;All Files (*)"
        )

        if not chosen_path:
            return

        out_stem = os.path.splitext(os.path.basename(chosen_path))[0]

        # Render high-resolution 1:1 image of the exact canvas area
        target_img = QImage(rw, rh, QImage.Format.Format_ARGB32)
        target_img.fill(QColor(0, 0, 0, 255))  # Professional solid black X-ray background

        painter = QPainter(target_img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        # Draw all pieces located inside or intersecting the target bounding box
        for piece in sorted(self.pieces, key=lambda p: (p.group_id, p.id)):
            pw = piece.w
            ph = piece.h
            if (piece.x + pw < rx or piece.x > rx + rw or
                piece.y + ph < ry or piece.y > ry + rh):
                continue

            pix = self.get_piece_pixmap(piece)
            dest_x = piece.x - rx
            dest_y = piece.y - ry
            dest_rect = QRectF(dest_x, dest_y, pw, ph)

            painter.save()
            if piece.is_selected and self.selection_opacity < 0.99:
                painter.setOpacity(self.selection_opacity)
            painter.drawPixmap(dest_rect.toRect(), pix)
            painter.restore()

        painter.end()

        # Flat canonical export copy
        flat_copy_path = os.path.join(OUTPUT_SAVE_DIRECTORY, f"{out_stem}_HQ.png")

        # Contract export folder with metadata (Section 3 & 8 of Contract)
        run_folder = os.path.join(OUTPUT_SAVE_DIRECTORY, f"{out_stem}_export")
        os.makedirs(run_folder, exist_ok=True)

        # 1. Validity Mask & Source Index Map
        validity_img = QImage(rw, rh, QImage.Format.Format_Grayscale8)
        validity_img.fill(0)
        source_map_img = QImage(rw, rh, QImage.Format.Format_ARGB32)
        source_map_img.fill(QColor(0, 0, 0, 255))

        palette = [
            QColor(59, 130, 246), QColor(16, 185, 129), QColor(245, 158, 11),
            QColor(239, 68, 68), QColor(168, 85, 247), QColor(236, 72, 153),
            QColor(20, 184, 166), QColor(249, 115, 22)
        ]

        piece_transforms = []
        for idx, piece in enumerate(sorted(self.pieces, key=lambda p: (p.group_id, p.id))):
            pw, ph = piece.w, piece.h
            if (piece.x + pw < rx or piece.x > rx + rw or
                piece.y + ph < ry or piece.y > ry + rh):
                continue

            dest_x = piece.x - rx
            dest_y = piece.y - ry
            dest_rect = QRectF(dest_x, dest_y, pw, ph)

            piece_transforms.append({
                "piece_id": int(piece.id),
                "group_id": int(piece.group_id),
                "world_x": float(piece.x),
                "world_y": float(piece.y),
                "crop_x": float(dest_x),
                "crop_y": float(dest_y),
                "width": int(pw),
                "height": int(ph)
            })

            # Use the actual warped alpha footprint, never the bounding rectangle.
            rgba = piece.image
            support = (rgba[:, :, 3] > 0) if rgba.ndim == 3 and rgba.shape[2] == 4 else np.ones(rgba.shape[:2], bool)
            footprint = np.zeros((*support.shape, 4), dtype=np.uint8)
            footprint[support] = [255, 255, 255, 255]
            p_val = QPainter(validity_img)
            p_val.drawPixmap(dest_rect.toRect(), cv_mat_to_qpixmap(footprint))
            p_val.end()

            color = palette[idx % len(palette)]
            footprint[support] = [color.blue(), color.green(), color.red(), 255]
            p_src = QPainter(source_map_img)
            p_src.drawPixmap(dest_rect.toRect(), cv_mat_to_qpixmap(footprint))
            p_src.end()

        # 2. Global Transforms JSON
        transforms_data = {
            "coordinate_system": "top_left_origin_view_pixels",
            "capture_bounding_box": {
                "min_x": rx, "min_y": ry,
                "width": rw, "height": rh
            },
            "saved_image": chosen_path,
            "pieces": piece_transforms
        }

        # 3. QA & Calibration Report JSON (Section 8 Contract Compliant)
        win = self.window()
        is_cal = win.is_calibrated() if hasattr(win, 'is_calibrated') else False
        sdd = win.spin_sdd.value() if hasattr(win, 'spin_sdd') else 0.0
        sod = win.spin_sod.value() if hasattr(win, 'spin_sod') else 0.0
        pitch_um = win.spin_pitch.value() if hasattr(win, 'spin_pitch') else 0.0
        odd = (sdd - sod) if is_cal else None
        m0 = (sdd / sod) if is_cal else None
        pitch_ref_mm = ((pitch_um / 1000.0) / m0) if (is_cal and pitch_um > 0) else None

        report_data = {
            "schema_version": "1.0",
            "project_id": getattr(win, 'current_project_id', f"xray_proj_{ts}"),
            "run_id": f"run_{ts}",
            "stage_name": "mosaic_export",
            "mask_semantics": "visual warped-alpha footprint; not a calibrated metric-validity mask",
            "status": "VISUAL_ONLY",
            "created_at": datetime.datetime.now().isoformat(),
            "calibration": {
                "is_calibrated": False,
                "manual_parameters_present": is_cal,
                "SDD_mm": sdd if is_cal else None,
                "SOD_mm": sod if is_cal else None,
                "ODD_mm": round(odd, 3) if odd is not None else None,
                "reference_magnification_M0": round(m0, 4) if m0 is not None else None,
                "detector_pixel_pitch_um": pitch_um if pitch_um > 0 else None,
                "object_plane_pixel_size_mm": round(pitch_ref_mm, 5) if pitch_ref_mm is not None else None,
                "reference_plane": "detector_parallel_table",
                "height_sign_convention": "positive_towards_detector"
            },
            "physics_model_audit": {
                "cone_beam_divergence_mitigation": "Visual seam selection only; no physical cone-beam correction in legacy canvas",
                "depth_parallax_warning": "Warning: Features at elevated heights (h > 0) exhibit radial disparity proportional to M(h) = SDD / (SOD + h). Without an independent 3D height map (e.g. CT or multi-view stereo), seam selection changes appearance only; out-of-plane parallax cannot be mathematically flattened by 2D homography alone per Contract Sec 2 & 6.",
                "measurement_validity": "VISUAL_ONLY: manual/elastic canvas transforms are not a calibrated metric pipeline. Use reference-plane batch export for physical coordinates."
            },
            "canvas_region": {
                "min_x": rx, "min_y": ry,
                "width_px": rw, "height_px": rh,
                "pieces_included": len(piece_transforms)
            },
            "blending_configuration": {
                "blend_mode": self.blend_mode,
                "feather_enabled": self.enable_feathering,
                "feather_power": self.feather_power,
                "edge_brightness": self.edge_brightness
            },
            "warnings": ["VISUAL_ONLY: canvas image is an appearance export, without verified metric geometry."] + ([] if is_cal else [
                "Uncalibrated project: SDD and SOD were unspecified. Pixel coordinates and visual mosaic were exported, but physical mm dimensions and 3D depth compensation are withheld per Contract Sec 2."
            ])
        }

        self.current_state = self.STATE_NORMAL
        self.capture_rect_world = None
        self.unsetCursor()
        self.trigger_hotkey_hud('P', self.tr('hud_export'))
        self.update()

        def task_export_save():
            # Heavy PNG encoding and disk writes occur purely in background worker thread
            target_img.save(chosen_path)
            if os.path.abspath(chosen_path) != os.path.abspath(flat_copy_path):
                target_img.save(flat_copy_path)
            target_img.save(os.path.join(run_folder, f"{out_stem}.png"))
            target_img.save(os.path.join(run_folder, "mosaic.png"))
            validity_img.save(os.path.join(run_folder, "validity_mask.png"))
            source_map_img.save(os.path.join(run_folder, "source_index_map.png"))
            with open(os.path.join(run_folder, "transforms.json"), 'w', encoding='utf-8') as f:
                json.dump(transforms_data, f, indent=4, ensure_ascii=False)
            with open(os.path.join(run_folder, "report.json"), 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=4, ensure_ascii=False)
            return True

        def on_export_done(res):
            self.statusChanged.emit(self.tr('status_export_saved').format(path=chosen_path))
            dim_info = (f"ابعاد عکس کادر انتخابی: {rw} × {rh} پیکسل" if self.current_lang == 'fa' else f"Captured Image Dimensions: {rw} × {rh} px")
            QMessageBox.information(
                self,
                self.tr('dialog_export_title'),
                f"{self.tr('status_export_saved').format(path=chosen_path)}\n\n{dim_info}"
            )

        self.run_in_background(task_export_save, on_export_done, status_msg=self.tr('status_exporting'))

    def _draw_capture_region_overlay(self, painter):
        if self.current_state != self.STATE_CAPTURE_REGION:
            return

        w = self.width()
        h = self.height()

        if self.capture_rect_world is None:
            banner_rect = QRectF(w / 2.0 - 240, 20, 480, 40)
            painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(banner_rect, 8, 8)
            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            painter.setPen(QColor("#FFFFFF"))
            msg = "📸 کادر مورد نظر را روی بوم بکشید | [Esc] انصراف" if self.current_lang == 'fa' else "📸 Drag on canvas to select capture area | [Esc] Cancel"
            painter.drawText(banner_rect, Qt.AlignmentFlag.AlignCenter, msg)
            return

        r = self.capture_rect_world.normalized()
        vx1, vy1 = self.world_to_view(r.x(), r.y())
        vx2, vy2 = self.world_to_view(r.x() + r.width(), r.y() + r.height())
        rx = min(vx1, vx2)
        ry = min(vy1, vy2)
        rw = abs(vx2 - vx1)
        rh = abs(vy2 - vy1)
        crop_rect_v = QRectF(rx, ry, rw, rh)

        painter.save()

        # 1. Semi-transparent dark mask over everything outside crop_rect_v
        mask_path = QPainterPath()
        mask_path.addRect(0, 0, w, h)
        crop_path = QPainterPath()
        crop_path.addRect(crop_rect_v)
        dim_path = mask_path.subtracted(crop_path)
        painter.fillPath(dim_path, QBrush(QColor(0, 0, 0, 150)))

        # 2. Glowing emerald crop rectangle outline with rule-of-thirds grid
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#10B981"), 2.5))
        painter.drawRect(crop_rect_v)

        if rw > 60 and rh > 60:
            painter.setPen(QPen(QColor(16, 185, 129, 60), 1.0, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(rx + rw / 3.0, ry), QPointF(rx + rw / 3.0, ry + rh))
            painter.drawLine(QPointF(rx + 2.0 * rw / 3.0, ry), QPointF(rx + 2.0 * rw / 3.0, ry + rh))
            painter.drawLine(QPointF(rx, ry + rh / 3.0), QPointF(rx + rw, ry + rh / 3.0))
            painter.drawLine(QPointF(rx, ry + 2.0 * rh / 3.0), QPointF(rx + rw, ry + 2.0 * rh / 3.0))

        # 3. Eight Corner & Edge Handles
        handles = [
            (rx, ry), (rx + rw / 2.0, ry), (rx + rw, ry),
            (rx, ry + rh / 2.0), (rx + rw, ry + rh / 2.0),
            (rx, ry + rh), (rx + rw / 2.0, ry + rh), (rx + rw, ry + rh)
        ]
        painter.setBrush(QBrush(QColor("#FFFFFF")))
        painter.setPen(QPen(QColor("#059669"), 2.0))
        for hx, hy in handles:
            painter.drawRect(QRectF(hx - 5, hy - 5, 10, 10))

        # 4. Top Info Pill: Dimensions in Pixels
        native_w = int(round(r.width()))
        native_h = int(round(r.height()))
        badge_text = f"📐 {native_w} × {native_h} px"
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        badge_w = 170
        badge_rect = QRectF(rx, max(10.0, ry - 30.0), badge_w, 24)
        painter.setBrush(QBrush(QColor("#059669")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(badge_rect, 5, 5)
        painter.setPen(QColor("#FFFFFF"))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_text)

        # 5. Bottom floating action pill
        btn_w = 420 if self.current_lang == 'fa' else 380
        btn_h = 34
        btn_x = min(w - btn_w - 15, max(15.0, rx + rw / 2.0 - btn_w / 2.0))
        btn_y = ry + rh + 10
        if btn_y + btn_h > h - 15:
            btn_y = max(15.0, ry + rh - btn_h - 10)
        action_pill = QRectF(btn_x, btn_y, btn_w, btn_h)

        painter.setBrush(QBrush(QColor(15, 23, 42, 235)))
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawRoundedRect(action_pill, 6, 6)

        c_w = 150
        all_w = 130 if self.current_lang == 'fa' else 110
        esc_w = 110

        self._capture_pill_rect = action_pill
        self._capture_confirm_rect = QRectF(btn_x, btn_y, c_w, btn_h)
        self._capture_all_rect = QRectF(btn_x + c_w, btn_y, all_w, btn_h)
        self._capture_cancel_rect = QRectF(btn_x + btn_w - esc_w, btn_y, esc_w, btn_h)

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor("#34D399"))
        confirm_txt = "💾 ذخیره عکس [Enter]" if self.current_lang == 'fa' else "💾 Save [Enter]"
        painter.drawText(self._capture_confirm_rect, Qt.AlignmentFlag.AlignCenter, confirm_txt)

        painter.setPen(QColor("#38BDF8"))
        all_txt = "⊞ کادر کل قطعات" if self.current_lang == 'fa' else "⊞ Full Composition"
        painter.drawText(self._capture_all_rect, Qt.AlignmentFlag.AlignCenter, all_txt)

        painter.setPen(QColor("#94A3B8"))
        cancel_txt = "✖️ انصراف [Esc]" if self.current_lang == 'fa' else "✖️ Cancel [Esc]"
        painter.drawText(self._capture_cancel_rect, Qt.AlignmentFlag.AlignCenter, cancel_txt)

        painter.restore()

    def _create_image_from_group(self, group_id):
        group_pieces = [p for p in self.pieces if p.group_id == group_id]
        if not group_pieces:
            return None, None
        min_x = min(p.x for p in group_pieces)
        min_y = min(p.y for p in group_pieces)
        # F-06: Use active p.image to preserve stretch/elastic warp edits
        max_x = max(p.x + p.image.shape[1] for p in group_pieces)
        max_y = max(p.y + p.image.shape[0] for p in group_pieces)
        cw, ch = math.ceil(max_x - min_x), math.ceil(max_y - min_y)
        if cw <= 0 or ch <= 0:
            return None, None
        group_canvas = np.zeros((ch, cw, 4), dtype=np.uint8)
        for p in group_pieces:
            img_rgba = p.image
            if img_rgba.ndim == 2:
                img_rgba = cv.cvtColor(img_rgba, cv.COLOR_GRAY2BGRA)
            elif img_rgba.shape[2] == 3:
                img_rgba = cv.cvtColor(img_rgba, cv.COLOR_BGR2BGRA)
            xp, yp = int(p.x - min_x), int(p.y - min_y)
            h, w = img_rgba.shape[:2]
            roi = group_canvas[yp:yp+h, xp:xp+w]
            mask = img_rgba[:, :, 3] > 0
            roi[mask] = img_rgba[mask]
        return group_canvas, (min_x, min_y)

    def initiate_smart_snap(self):
        self.trigger_hotkey_hud('N', self.tr('hud_smart_snap'))
        if len(self.selection) < 2:
            self.statusChanged.emit(self.tr('status_smart_snap_req'))
            return
        if self.is_busy:
            return

        self.save_state()
        if len(self.selection) == 2:
            g1, g2 = self.selection[0], self.selection[1]
            img1, origin1 = self._create_image_from_group(g1)
            img2, origin2 = self._create_image_from_group(g2)
            if img1 is None or img2 is None:
                self.history.pop()
                return

            def task_snap():
                mode = getattr(self, 'stitch_mode', 'auto')
                return calculate_smart_snap_offset(img1, img2, engine=self.align_engine, stitch_mode=mode)

            def on_snap_done(offset):
                if offset is not None:
                    dx, dy = offset
                    target_x = origin1[0] + dx
                    target_y = origin1[1] + dy
                    move_x = target_x - origin2[0]
                    move_y = target_y - origin2[1]
                    for p in self.pieces:
                        if p.group_id == g2:
                            p.translate(move_x, move_y)
                    self.statusChanged.emit(self.tr('status_smart_snap_success'))
                else:
                    self.history.pop()
                    self.statusChanged.emit(self.tr('status_smart_snap_fail'))

            self.run_in_background(task_snap, on_snap_done, status_msg=self.tr('status_aligning'))
        else:
            # 2D Mosaic Matrix Alignment for multiple selected blocks
            selected_pieces = [p for p in self.pieces if p.group_id in self.selection]
            rep_pieces = []
            seen_gids = set()
            for p in selected_pieces:
                if p.group_id not in seen_gids:
                    rep_pieces.append(p)
                    seen_gids.add(p.group_id)

            parent_win = self.window()
            cols_hint = parent_win.spin_cols.value() if hasattr(parent_win, 'spin_cols') else 0
            spacing_hint = float(parent_win.spin_spacing.value() if hasattr(parent_win, 'spin_spacing') else 0)

            num_rows, num_cols, piece_cells = detect_2d_grid_cells(rep_pieces, cols_hint=cols_hint)
            cell_to_gid = {(r, c): p.group_id for r, c, p in piece_cells}

            align_plan = []
            for r in range(num_rows):
                for c in range(1, num_cols):
                    if (r, c) in cell_to_gid and (r, c - 1) in cell_to_gid:
                        align_plan.append((cell_to_gid[(r, c)], cell_to_gid[(r, c - 1)], 'h'))
            for r in range(1, num_rows):
                if (r, 0) in cell_to_gid and (r - 1, 0) in cell_to_gid:
                    align_plan.append((cell_to_gid[(r, 0)], cell_to_gid[(r - 1, 0)], 'v'))

            group_images = {}
            for gid in seen_gids:
                img, orig = self._create_image_from_group(gid)
                group_images[gid] = (img, orig)

            def task_matrix_snap():
                # مد ۴: بهینه‌ساز سراسری گراف در صورت انتخاب
                if getattr(self, 'use_global_graph', True) and len(seen_gids) >= 3:
                    pairwise_offsets = {}
                    for child_gid, parent_gid, direction in align_plan:
                        p_img, p_orig = group_images.get(parent_gid, (None, None))
                        c_img, c_orig = group_images.get(child_gid, (None, None))
                        if p_img is None or c_img is None:
                            continue
                        offset = calculate_smart_snap_offset(p_img, c_img, engine=self.align_engine)
                        if offset is not None:
                            pairwise_offsets[(child_gid, parent_gid)] = (offset[0], offset[1], 1.0)
                    if pairwise_offsets:
                        anchor_gid = rep_pieces[len(rep_pieces)//2].group_id
                        solved = optimize_global_grid_offsets(cell_to_gid, pairwise_offsets, anchor_gid=anchor_gid)
                        return ('global_graph', solved)

                results = []
                for child_gid, parent_gid, direction in align_plan:
                    p_img, p_orig = group_images.get(parent_gid, (None, None))
                    c_img, c_orig = group_images.get(child_gid, (None, None))
                    if p_img is None or c_img is None:
                        results.append((child_gid, parent_gid, direction, None))
                        continue
                    offset = calculate_smart_snap_offset(p_img, c_img, engine=self.align_engine)
                    results.append((child_gid, parent_gid, direction, offset))
                return ('chain', results)

            def on_matrix_snap_done(results_package):
                if isinstance(results_package, tuple) and results_package[0] == 'global_graph':
                    solved = results_package[1]
                    anchor_gid = rep_pieces[len(rep_pieces)//2].group_id
                    anchor_pieces = [p for p in self.pieces if p.group_id == anchor_gid]
                    base_ax = min(p.x for p in anchor_pieces) if anchor_pieces else 0.0
                    base_ay = min(p.y for p in anchor_pieces) if anchor_pieces else 0.0
                    succ = 0
                    for gid, (gx, gy) in solved.items():
                        g_pieces = [p for p in self.pieces if p.group_id == gid]
                        if not g_pieces: continue
                        cur_gx = min(p.x for p in g_pieces)
                        cur_gy = min(p.y for p in g_pieces)
                        target_x = base_ax + gx
                        target_y = base_ay + gy
                        move_x = target_x - cur_gx
                        move_y = target_y - cur_gy
                        for p in g_pieces:
                            p.translate(move_x, move_y)
                        succ += 1
                    self.statusChanged.emit(f"Global Graph Mosaic alignment completed for {succ} tiles (Zero-Drift).")
                    self.fit_view()
                    return

                results = results_package[1] if isinstance(results_package, tuple) else results_package
                succ = 0
                group_bounds = {}
                for gid in seen_gids:
                    g_pieces = [p for p in self.pieces if p.group_id == gid]
                    if g_pieces:
                        gx = min(p.x for p in g_pieces)
                        gy = min(p.y for p in g_pieces)
                        gw = max(p.x + p.w for p in g_pieces) - gx
                        gh = max(p.y + p.h for p in g_pieces) - gy
                        group_bounds[gid] = [gx, gy, gw, gh]

                for child_gid, parent_gid, direction, offset in results:
                    pb = group_bounds.get(parent_gid)
                    cb = group_bounds.get(child_gid)
                    if pb is None or cb is None:
                        continue

                    if offset is not None:
                        dx, dy = offset
                        target_x = pb[0] + dx
                        target_y = pb[1] + dy
                        succ += 1
                    else:
                        if direction == 'h':
                            target_x = pb[0] + pb[2] + spacing_hint
                            target_y = pb[1]
                        else:
                            target_x = pb[0]
                            target_y = pb[1] + pb[3] + spacing_hint

                    move_x = target_x - cb[0]
                    move_y = target_y - cb[1]

                    for p in self.pieces:
                        if p.group_id == child_gid:
                            p.translate(move_x, move_y)

                    cb[0] += move_x
                    cb[1] += move_y

                self.statusChanged.emit(self.tr('status_smart_snap_chain').format(count=succ))
                self.fit_view()

            self.run_in_background(task_matrix_snap, on_matrix_snap_done, status_msg=self.tr('status_aligning'))

    def initiate_manual_registration(self):
        """Inspect two selected rasters using explicit fit and held-out check pairs."""
        if self.is_busy:
            return
        if len(self.selection) != 2:
            self.statusChanged.emit('دقیقاً دو تصویر یا گروه را انتخاب کنید | Select exactly two groups')
            return
        from manual_registration_qt import CorrespondenceDialog
        from PyQt6.QtWidgets import QDialog
        group_ids = sorted(self.selection)
        inputs, labels = [], []
        for gid in group_ids:
            members = [p for p in self.pieces if p.group_id == gid]
            if len(members) == 1:
                # Preserve the active raster dtype, including native uint16.
                inputs.append((members[0].image.copy(), (members[0].x, members[0].y)))
            else:
                inputs.append(self._create_image_from_group(gid))
            labels.append(' + '.join(p.filename or str(p.id) for p in members))
        if any(im is None for im, _ in inputs):
            self.statusChanged.emit('Cannot read selected image groups')
            return
        dialog = CorrespondenceDialog([im for im, _ in inputs], labels,
                    getattr(self, 'visual_mosaic_model', 'projective'), self)
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.result is None:
            return
        result = dialog.result
        self.save_state()
        self.pieces = [p for p in self.pieces if p.group_id not in group_ids]
        image = result['image']
        origin, offset = inputs[0][1], result['origin_anchor_px']
        pid = self.next_piece_id
        self.next_piece_id += 1
        piece = PuzzlePiece(pid, image, origin[0]+offset[0], origin[1]+offset[1],
                            original_ref=image, filename='manual_checked_mosaic.png')
        piece.visual_mosaic_qa = result['qa']
        self.pieces.append(piece)
        self.selection.clear()
        self.pixmap_cache.clear()
        self.save_state()
        self.fit_view()
        check = result['qa']['manual_registration']['check']['max_px']
        self.statusChanged.emit(f'Manual mosaic: independent check max {check:.3f} px; VISUAL_ONLY')

    def initiate_visual_mosaic(self):
        """Full projective batch composition of selected groups, with undo and gap reporting."""
        if self.is_busy:
            return
        if len(self.selection) < 2:
            self.statusChanged.emit(self.tr('status_auto_align_req'))
            return
        from visual_mosaic import build_mosaic, Options
        import threading
        self.window()._visual_cancel = threading.Event()
        cancel = self.window()._visual_cancel
        selected = set(self.selection)
        group_ids = sorted(selected)
        inputs = [self._create_image_from_group(gid) for gid in group_ids]
        if any(image is None for image, _ in inputs):
            self.statusChanged.emit('Cannot read selected image groups')
            return
        self.save_state()
        selected_model = getattr(self, 'visual_mosaic_model', 'projective')
        def task():
            return build_mosaic([image for image, _ in inputs], Options(model=selected_model), cancelled=cancel.is_set)
        def done(result):
            # Replace each verified component by one composed image. Unmatched
            # singleton groups keep their original position and pixels.
            joined = [c for c in result['components'] if len(c['qa']['frames']) > 1]
            if not joined:
                self.history.pop()
                self.statusChanged.emit('No reliable overlap found | هم‌پوشانی قابل اعتماد پیدا نشد')
                return
            for component in joined:
                members = {group_ids[i] for i in component['qa']['frames']}
                anchor = component['qa']['anchor']
                origin = inputs[anchor][1]
                offset = component['origin_anchor_px']
                self.pieces = [p for p in self.pieces if p.group_id not in members]
                pid = self.next_piece_id
                self.next_piece_id += 1
                image = component['image']
                piece = PuzzlePiece(pid, image, origin[0]+offset[0], origin[1]+offset[1],
                                    original_ref=image, filename='perspective_mosaic.png')
                piece.visual_mosaic_qa = component['qa']
                self.pieces.append(piece)
            self.selection.clear()
            self.pixmap_cache.clear()
            self.save_state()
            self.fit_view()
            residuals = [e['median_px'] for c in joined for e in c['qa']['global_match_residuals']]
            value = max(residuals) if residuals else float('nan')
            self.statusChanged.emit(f'Image mosaic ({selected_model}): {len(joined)} joined component(s), '
                                    f'{len(result["components"])} total; max edge median {value:.3f} px; VISUAL_ONLY')
        def failed(error):
            if self.history:
                self.history.pop()
            self.statusChanged.emit('Perspective alignment failed: '+error)
        self.run_in_background(task, done, failed, status_msg='Matching perspective images and optimizing all connections…')

    def initiate_auto_align(self):
        self.trigger_hotkey_hud('G', self.tr('hud_auto_align'))
        if (len(self.selection) > 2 or
                (len(self.selection) >= 2 and getattr(self, 'visual_mosaic_model', 'projective') == 'affine') or
                any(sum(p.group_id == gid for p in self.pieces) > 1 for gid in self.selection)):
            self.initiate_visual_mosaic()
            return
        if len(self.selection) < 2:
            self.statusChanged.emit(self.tr('status_auto_align_req'))
            return
        if self.is_busy:
            return

        self.save_state()
        if len(self.selection) == 2:
            g1, g2 = self.selection[0], self.selection[1]
            img1, origin1 = self._create_image_from_group(g1)
            img2, origin2 = self._create_image_from_group(g2)
            if img1 is None or img2 is None:
                self.history.pop()
                return

            def task_align():
                mode = getattr(self, 'stitch_mode', 'auto')
                return calculate_auto_align_transform(img1, img2, engine=self.align_engine, stitch_mode=mode)

            def on_align_done(H):
                if H is not None:
                    h2, w2 = img2.shape[:2]
                    # ارزیابی وجود تبدیل پرسپکتیو، چرخش یا مقیاس
                    # Judge the discarded warp in pixels over the whole image.
                    # A 0.3% scale change is already several pixels on a large shot.
                    probes = np.array([[[0., 0.], [w2, 0.], [w2, h2], [0., h2], [w2/2., h2/2.]]], np.float32)
                    mapped = cv.perspectiveTransform(probes, H)[0]
                    centre_shift = mapped[4]-probes[0, 4]
                    deviation = np.linalg.norm(mapped[:4]-probes[0, :4]-centre_shift, axis=1)
                    is_affine_or_perspective = bool(np.max(deviation) > 0.05)
                    g2_members = [p for p in self.pieces if p.group_id == g2]

                    if is_affine_or_perspective and len(g2_members) == 1:
                        # تصحیح کامل ذوزنقه‌ای و مقیاس با تبدیل هندسی کامل ماتریس H
                        corners2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32).reshape(-1, 1, 2)
                        warped_corners1 = cv.perspectiveTransform(corners2, H).reshape(-1, 2)

                        x_min = float(np.min(warped_corners1[:, 0]))
                        y_min = float(np.min(warped_corners1[:, 1]))
                        x_max = float(np.max(warped_corners1[:, 0]))
                        y_max = float(np.max(warped_corners1[:, 1]))
                        out_w = max(16, int(round(x_max - x_min)))
                        out_h = max(16, int(round(y_max - y_min)))

                        Ht = np.array([
                            [1.0, 0.0, -x_min],
                            [0.0, 1.0, -y_min],
                            [0.0, 0.0, 1.0]
                        ], dtype=np.float32)

                        H_composed = Ht @ H
                        warped_img2 = cv.warpPerspective(
                            img2, H_composed, (out_w, out_h),
                            flags=cv.INTER_CUBIC,
                            borderMode=cv.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0)
                        )

                        p = g2_members[0]
                        p.image = warped_img2
                        p.original_image_ref = warped_img2
                        p.x = origin1[0] + x_min
                        p.y = origin1[1] + y_min
                        p.w = out_w
                        p.h = out_h
                        p.sync_corners_to_rect()
                        self.pixmap_cache.clear()
                    else:
                        # در صورت تبدیل صلب، جابه‌جایی بر اساس نگاشت مرکز هندسی جهت دقت زیرپیکسل
                        c2 = np.array([[[w2 / 2.0, h2 / 2.0]]], dtype=np.float32)
                        c1 = cv.perspectiveTransform(c2, H).reshape(2)
                        target_center_x = origin1[0] + float(c1[0])
                        target_center_y = origin1[1] + float(c1[1])
                        current_center_x = origin2[0] + w2 / 2.0
                        current_center_y = origin2[1] + h2 / 2.0
                        dx = target_center_x - current_center_x
                        dy = target_center_y - current_center_y
                        for p in self.pieces:
                            if p.group_id == g2:
                                p.translate(dx, dy)

                    self.statusChanged.emit(self.tr('status_auto_align_success'))
                else:
                    self.history.pop()
                    self.statusChanged.emit(self.tr('status_auto_align_fail'))

            self.run_in_background(task_align, on_align_done, status_msg=self.tr('status_aligning'))
        else:
            # 2D Mosaic Matrix Auto-Align for multiple selected blocks
            selected_pieces = [p for p in self.pieces if p.group_id in self.selection]
            rep_pieces = []
            seen_gids = set()
            for p in selected_pieces:
                if p.group_id not in seen_gids:
                    rep_pieces.append(p)
                    seen_gids.add(p.group_id)

            parent_win = self.window()
            cols_hint = parent_win.spin_cols.value() if hasattr(parent_win, 'spin_cols') else 0
            spacing_hint = float(parent_win.spin_spacing.value() if hasattr(parent_win, 'spin_spacing') else 0)

            num_rows, num_cols, piece_cells = detect_2d_grid_cells(rep_pieces, cols_hint=cols_hint)
            cell_to_gid = {(r, c): p.group_id for r, c, p in piece_cells}

            align_plan = []
            for r in range(num_rows):
                for c in range(1, num_cols):
                    if (r, c) in cell_to_gid and (r, c - 1) in cell_to_gid:
                        align_plan.append((cell_to_gid[(r, c)], cell_to_gid[(r, c - 1)], 'h'))
            for r in range(1, num_rows):
                if (r, 0) in cell_to_gid and (r - 1, 0) in cell_to_gid:
                    align_plan.append((cell_to_gid[(r, 0)], cell_to_gid[(r - 1, 0)], 'v'))

            group_images = {}
            for gid in seen_gids:
                img, orig = self._create_image_from_group(gid)
                group_images[gid] = (img, orig)

            def task_matrix_align():
                results = []
                for child_gid, parent_gid, direction in align_plan:
                    p_img, p_orig = group_images.get(parent_gid, (None, None))
                    c_img, c_orig = group_images.get(child_gid, (None, None))
                    if p_img is None or c_img is None:
                        results.append((child_gid, parent_gid, direction, None))
                        continue
                    H = calculate_auto_align_transform(p_img, c_img, engine=self.align_engine)
                    results.append((child_gid, parent_gid, direction, H))
                return results

            def on_matrix_align_done(results):
                succ = 0
                group_bounds = {}
                for gid in seen_gids:
                    g_pieces = [p for p in self.pieces if p.group_id == gid]
                    if g_pieces:
                        gx = min(p.x for p in g_pieces)
                        gy = min(p.y for p in g_pieces)
                        gw = max(p.x + p.w for p in g_pieces) - gx
                        gh = max(p.y + p.h for p in g_pieces) - gy
                        group_bounds[gid] = [gx, gy, gw, gh]

                for child_gid, parent_gid, direction, H in results:
                    pb = group_bounds.get(parent_gid)
                    cb = group_bounds.get(child_gid)
                    if pb is None or cb is None:
                        continue

                    if H is not None:
                        tx, ty = H[0, 2], H[1, 2]
                        target_x = pb[0] + tx
                        target_y = pb[1] + ty
                        succ += 1
                    else:
                        if direction == 'h':
                            target_x = pb[0] + pb[2] + spacing_hint
                            target_y = pb[1]
                        else:
                            target_x = pb[0]
                            target_y = pb[1] + pb[3] + spacing_hint

                    move_x = target_x - cb[0]
                    move_y = target_y - cb[1]

                    for p in self.pieces:
                        if p.group_id == child_gid:
                            p.translate(move_x, move_y)

                    cb[0] += move_x
                    cb[1] += move_y

                self.statusChanged.emit(self.tr('status_auto_align_chain').format(count=succ))
                self.fit_view()

            self.run_in_background(task_matrix_align, on_matrix_align_done, status_msg=self.tr('status_aligning'))

    def initiate_elastic_warp(self):
        self.trigger_hotkey_hud('E', self.tr('hud_elastic'))
        if len(self.selection) != 2:
            self.statusChanged.emit(self.tr('status_elastic_req'))
            return
        if self.is_busy:
            return

        self.save_state()
        g1, g2 = self.selection[0], self.selection[1]
        img1, origin1 = self._create_image_from_group(g1)
        img2, origin2 = self._create_image_from_group(g2)
        if img1 is None or img2 is None:
            self.history.pop()
            return

        def task_elastic():
            return calculate_warp_deformation(img1, img2, engine=self.warp_engine)

        def on_elastic_done(res):
            if res is not None and isinstance(res, (tuple, list)) and len(res) == 2:
                warped_img2, offset = res
            else:
                warped_img2, offset = None, None
            if warped_img2 is not None and offset is not None:
                global_dx, global_dy = offset
                target_x = origin1[0] + global_dx
                target_y = origin1[1] + global_dy
                move_x = target_x - origin2[0]
                move_y = target_y - origin2[1]

                # F-05: Handle single-piece vs multi-piece group cleanly
                g2_members = [p for p in self.pieces if p.group_id == g2]
                if len(g2_members) == 1:
                    p = g2_members[0]
                    p.image = warped_img2
                    p.original_image_ref = warped_img2
                    p.w = warped_img2.shape[1]
                    p.h = warped_img2.shape[0]
                    p.translate(move_x, move_y)
                    p.sync_corners_to_rect()
                else:
                    # Multi-piece group: consolidate into a single unified warped piece
                    self.pieces = [p for p in self.pieces if p.group_id != g2]
                    new_p = PuzzlePiece(
                        self.next_piece_id,
                        warped_img2,
                        origin2[0] + move_x,
                        origin2[1] + move_y,
                        original_ref=warped_img2,
                        filename=f"group_{g2}_elastic.png"
                    )
                    new_p.group_id = g2
                    new_p.is_selected = True
                    self.next_piece_id += 1
                    self.pieces.append(new_p)

                self.pixmap_cache.clear()
                self.statusChanged.emit(self.tr('status_elastic_success'))
            else:
                self.history.pop()
                self.statusChanged.emit(self.tr('status_elastic_fail'))

        self.run_in_background(task_elastic, on_elastic_done, status_msg=self.tr('status_aligning'))

    def initiate_batch_merge(self):
        self.trigger_hotkey_hud('M', self.tr('hud_merge'))
        if not self.selection:
            self.statusChanged.emit(self.tr('status_merge_no_sel'))
            return
        selected = [p for p in self.pieces if p.group_id in self.selection]
        if len(selected) < 2:
            self.statusChanged.emit(self.tr('status_merge_min2'))
            return
        if self.is_busy:
            return

        # F-06: Use active p.image to preserve stretch/elastic warp edits
        pieces_data = [(p.image.copy(), float(p.x), float(p.y), p.group_id) for p in selected]
        feather_power = self.feather_power
        enable_feathering = self.enable_feathering
        blend_mode = getattr(self, 'blend_mode', 'seamline')
        selected_gids = list(self.selection)

        def task_batch():
            if not pieces_data:
                return None, 0, 0

            if blend_mode == 'multiband':
                # ادغام هرمی چندبانده فرکانسی لاپلاسین برای حفظ ۱۰۰٪ تیزی ترک‌ها و خطوط
                from neural_stitcher_engine import merge_images_multiband_pyramid
                base_img, bx, by, _ = pieces_data[0]
                if base_img.shape[2] == 3:
                    base_img = cv.cvtColor(base_img, cv.COLOR_BGR2BGRA)
                curr_canvas = base_img
                curr_x, curr_y = bx, by

                for img, px, py, _ in pieces_data[1:]:
                    if img.shape[2] == 3:
                        img = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
                    rel_pos = (int(round(px - curr_x)), int(round(py - curr_y)))
                    curr_canvas, new_rel_offset = merge_images_multiband_pyramid(curr_canvas, img, rel_pos)
                    curr_x += new_rel_offset[0]
                    curr_y += new_rel_offset[1]
                return curr_canvas, curr_x, curr_y

            min_x = min(item[1] for item in pieces_data)
            min_y = min(item[2] for item in pieces_data)
            max_x = max(item[1] + item[0].shape[1] for item in pieces_data)
            max_y = max(item[2] + item[0].shape[0] for item in pieces_data)
            fw, fh = math.ceil(max_x - min_x), math.ceil(max_y - min_y)
            if fw <= 0 or fh <= 0:
                return None, 0, 0

            final_canvas = np.zeros((fh, fw, 4), dtype=np.uint8)
            from image_merger import find_optimal_seamline_mask

            for img, px, py, gid in pieces_data:
                if img.shape[2] == 3:
                    img = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
                xr, yr = int(px - min_x), int(py - min_y)
                hi, wi = img.shape[:2]
                ye = min(yr + hi, fh)
                xe = min(xr + wi, fw)
                if ye <= yr or xe <= xr:
                    continue

                roi = final_canvas[yr:ye, xr:xe]
                img_part = img[0:(ye - yr), 0:(xe - xr)]

                mask_top = img_part[:, :, 3] > 0
                mask_base = roi[:, :, 3] > 0
                overlap = mask_top & mask_base

                if np.any(overlap) and enable_feathering:
                    if blend_mode == 'seamline':
                        # برش خط درز پویا برای حذف قطعی شبح‌وارگی و دوبل شدن قطعات سه‌بعدی
                        y_idxs, x_idxs = np.where(overlap)
                        ymin, ymax = y_idxs.min(), y_idxs.max()
                        xmin, xmax = x_idxs.min(), x_idxs.max()

                        crop_base = roi[ymin:ymax+1, xmin:xmax+1]
                        crop_top = img_part[ymin:ymax+1, xmin:xmax+1]
                        crop_mask = overlap[ymin:ymax+1, xmin:xmax+1]

                        d_top = cv.distanceTransform(mask_top.astype(np.uint8), cv.DIST_L2, 5)[ymin:ymax+1, xmin:xmax+1]
                        d_base = cv.distanceTransform(mask_base.astype(np.uint8), cv.DIST_L2, 5)[ymin:ymax+1, xmin:xmax+1]

                        weight_top_crop = find_optimal_seamline_mask(crop_base, crop_top, crop_mask, d_base, d_top)
                        seam_choice = (weight_top_crop >= 0.5) & crop_mask
                        crop_base[seam_choice] = crop_top[seam_choice]
                        roi[ymin:ymax+1, xmin:xmax+1] = crop_base
                        roi[overlap, 3] = np.maximum(roi[overlap, 3], img_part[overlap, 3])
                    else:
                        d_top = cv.distanceTransform(mask_top.astype(np.uint8), cv.DIST_L2, 5)
                        d_base = cv.distanceTransform(mask_base.astype(np.uint8), cv.DIST_L2, 5)
                        wt = np.power(d_top[overlap].astype(np.float32), feather_power)
                        wb = np.power(d_base[overlap].astype(np.float32), feather_power)
                        sum_w = wt + wb
                        sum_w[sum_w == 0] = 1.0
                        wt_3d = (wt / sum_w)[:, np.newaxis]
                        wb_3d = (wb / sum_w)[:, np.newaxis]
                        blended = (img_part[overlap, :3].astype(np.float32) * wt_3d +
                                   roi[overlap, :3].astype(np.float32) * wb_3d)
                        roi[overlap, :3] = np.clip(blended, 0, 255).astype(np.uint8)
                        roi[overlap, 3] = np.maximum(roi[overlap, 3], img_part[overlap, 3])
                elif np.any(mask_top):
                    roi[mask_top] = img_part[mask_top]

                only_top = mask_top & (~mask_base)
                if np.any(only_top):
                    roi[only_top] = img_part[only_top]

            return final_canvas, min_x, min_y

        def on_done_batch(res):
            final_canvas, min_x, min_y = res
            if final_canvas is None:
                return
            self.save_state()
            new_id = self.next_piece_id
            self.next_piece_id += 1
            new_piece = PuzzlePiece(new_id, final_canvas, min_x, min_y, original_ref=final_canvas)
            self.pieces = [p for p in self.pieces if p.group_id not in selected_gids]
            self.pieces.append(new_piece)
            self.selection = [new_id]
            new_piece.is_selected = True
            self.pixmap_cache.clear()
            mode_name = "Seamline Cut" if blend_mode == 'seamline' else ("Multi-Band Pyramid" if blend_mode == 'multiband' else "Feathering")
            self.statusChanged.emit(self.tr('status_merge_success').format(mode=mode_name))
            self.selectionChanged.emit(self.selection)

        self.run_in_background(task_batch, on_done_batch, status_msg=self.tr('status_merging'))

    def initiate_overlap_mode(self):
        self.trigger_hotkey_hud('M', self.tr('hud_overlap'))
        if self.is_busy:
            return
        if len(self.selection) != 2:
            self.statusChanged.emit(self.tr('status_overlap_req'))
            return
        g1_id, g2_id = self.selection
        p1 = next((p for p in self.pieces if p.group_id == g1_id), None)
        p2 = next((p for p in self.pieces if p.group_id == g2_id), None)
        if not p1 or not p2:
            return
        top_p, base_p, axis = self._determine_overlap_axis(p1, p2)
        base_img, base_orig = self._create_image_from_group(base_p.group_id)
        top_img, top_orig = self._create_image_from_group(top_p.group_id)
        initial_pos = (int(top_orig[0] - base_orig[0]), int(top_orig[1] - base_orig[1]))

        # Precompute distance maps for rapid real-time seamline preview
        dist_base = compute_center_weighted_mask(base_img[:, :, 3] > 0)
        dist_top = compute_center_weighted_mask(top_img[:, :, 3] > 0)

        self.overlap_info = {
            "base_gid": base_p.group_id,
            "top_gid": top_p.group_id,
            "axis": axis,
            "initial_pos": initial_pos,
            "current_pos": initial_pos,
            "base_origin": base_orig,
            "base_img": base_img,
            "top_img": top_img,
            "dist_base": dist_base,
            "dist_top": dist_top
        }
        self.current_state = self.STATE_OVERLAP_ADJUST
        self.overlap_alpha = self.selection_opacity
        self.temp_preview = None
        self.temp_preview_pixmap = None
        self._update_preview_from_current_pos()
        self.statusChanged.emit(self.tr('status_overlap_enter'))
        self.update()

    def _determine_overlap_axis(self, p1, p2):
        p1_cx, p1_cy = p1.x + p1.w / 2, p1.y + p1.h / 2
        p2_cx, p2_cy = p2.x + p2.w / 2, p2.y + p2.h / 2
        ox = min(p1.x + p1.w, p2.x + p2.w) - max(p1.x, p2.x)
        oy = min(p1.y + p1.h, p2.y + p2.h) - max(p1.y, p2.y)
        if ox > oy:
            return (p2, p1, 'y') if p1_cy < p2_cy else (p1, p2, 'y')
        else:
            return (p2, p1, 'x') if p1_cx < p2_cx else (p1, p2, 'x')

    def _update_preview_from_current_pos(self):
        if not self.overlap_info or self.current_state != self.STATE_OVERLAP_ADJUST:
            return
        if getattr(self, '_is_updating_preview', False):
            self.overlap_preview_timer.start(50)
            return
        self._is_updating_preview = True
        try:
            base_img = self.overlap_info.get("base_img")
            top_img = self.overlap_info.get("top_img")
            if base_img is None or top_img is None:
                base_img, base_orig = self._create_image_from_group(self.overlap_info["base_gid"])
                top_img, _ = self._create_image_from_group(self.overlap_info["top_gid"])
                self.overlap_info["base_img"] = base_img
                self.overlap_info["top_img"] = top_img
            if base_img is None or top_img is None:
                return

            dist_base = self.overlap_info.get("dist_base")
            dist_top = self.overlap_info.get("dist_top")
            if dist_base is None:
                dist_base = compute_center_weighted_mask(base_img[:, :, 3] > 0)
                self.overlap_info["dist_base"] = dist_base
            if dist_top is None:
                dist_top = compute_center_weighted_mask(top_img[:, :, 3] > 0)
                self.overlap_info["dist_top"] = dist_top

            merged, new_offset = merge_images_with_offset(
                base_img, top_img, self.overlap_info["current_pos"],
                feather_power=self.feather_power,
                enable_feathering=self.enable_feathering,
                edge_brightness=self.edge_brightness,
                blend_mode=self.blend_mode,
                precomputed_dist_base=dist_base,
                precomputed_dist_top=dist_top
            )
            self.temp_preview = merged
            self.overlap_info['final_offset'] = new_offset
            self.temp_preview_pixmap = cv_mat_to_qpixmap(merged)
            self.update()
        finally:
            self._is_updating_preview = False

    def finalize_overlap(self):
        self.trigger_hotkey_hud('ENTER', self.tr('hud_finalize'))
        if self.current_state != self.STATE_OVERLAP_ADJUST or not self.overlap_info:
            return
        if self.is_busy:
            return

        base_img = self.overlap_info.get("base_img")
        top_img = self.overlap_info.get("top_img")
        cur_pos = self.overlap_info.get("current_pos")
        base_orig = self.overlap_info.get("base_origin")
        base_gid = self.overlap_info.get("base_gid")
        top_gid = self.overlap_info.get("top_gid")
        feather_power = self.feather_power
        enable_feathering = self.enable_feathering
        edge_brightness = self.edge_brightness
        blend_mode = self.blend_mode

        def task_merge():
            return merge_images_with_offset(
                base_img, top_img, cur_pos,
                feather_power=feather_power,
                enable_feathering=enable_feathering,
                edge_brightness=edge_brightness,
                blend_mode=blend_mode
            )

        def on_done_merge(result):
            merged, final_offset = result
            self.save_state()
            new_x = base_orig[0] + final_offset[0]
            new_y = base_orig[1] + final_offset[1]
            new_id = self.next_piece_id
            self.next_piece_id += 1

            new_piece = PuzzlePiece(new_id, merged, new_x, new_y, original_ref=merged)
            self.pieces = [p for p in self.pieces if p.group_id not in (base_gid, top_gid)]
            self.pieces.append(new_piece)
            self.selection = [new_id]
            new_piece.is_selected = True
            self.pixmap_cache.clear()
            self._reset_modes()
            self.statusChanged.emit(self.tr('status_overlap_finalized'))
            self.selectionChanged.emit(self.selection)

        self.run_in_background(task_merge, on_done_merge, status_msg=self.tr('status_merging'))

    # ================== Mouse Handling ==================
    def mousePressEvent(self, event):
        if self.is_busy:
            return
        pos = event.position()
        vx, vy = pos.x(), pos.y()
        wx, wy = self.view_to_world(vx, vy)

        if event.button() == Qt.MouseButton.RightButton:
            if self.current_state == self.STATE_STRETCH_WARP:
                sel_piece = next((p for p in self.pieces if p.group_id in self.selection), None)
                if sel_piece is None and self.stretch_active_piece:
                    sel_piece = self.stretch_active_piece
                if sel_piece and len(sel_piece.get_boundary_world()) > 20:
                    pin_idx = self._find_nearest_pin(sel_piece, wx, wy)
                    corner_indices = getattr(sel_piece, 'corner_indices', [0, 20, 40, 60])
                    if pin_idx >= 0 and pin_idx not in corner_indices:
                        # Right-click on existing pin: delete it!
                        sel_piece.boundary_world = np.delete(sel_piece.boundary_world, pin_idx, axis=0)
                        if hasattr(sel_piece, 'boundary_src') and sel_piece.boundary_src is not None and len(sel_piece.boundary_src) > pin_idx:
                            sel_piece.boundary_src = np.delete(sel_piece.boundary_src, pin_idx, axis=0)
                        sel_piece.corner_indices = [c - 1 if c > pin_idx else c for c in corner_indices]
                        self._apply_stretch_warp(sel_piece, sel_piece.boundary_world)
                        self.update()
                        return
            self.is_panning = True
            self.drag_start_pos = pos
            self.last_mouse_pos = pos
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self.current_state == self.STATE_CAPTURE_REGION:
                h_type = self._hit_test_capture_handles(vx, vy)
                if h_type == 'btn_confirm':
                    self.confirm_region_capture()
                    return
                elif h_type == 'btn_all':
                    self.select_all_pieces_capture()
                    return
                elif h_type == 'btn_cancel':
                    self.current_state = self.STATE_NORMAL
                    self.capture_rect_world = None
                    self.unsetCursor()
                    self.update()
                    return

                self.is_dragging = True
                self.capture_active_handle = h_type
                self.capture_drag_start = QPointF(wx, wy)
                if h_type == 'new':
                    self.capture_rect_world = QRectF(wx, wy, 0, 0)
                self.update()
                return

            if self.current_state == self.STATE_STRETCH_WARP:
                sel_piece = next((p for p in self.pieces if p.group_id in self.selection), None)
                if sel_piece is None and self.stretch_active_piece:
                    sel_piece = self.stretch_active_piece

                # Check if clicking on an existing pin or dashed edge line
                if sel_piece:
                    h_idx, h_pos = self._hit_test_stretch_handles(sel_piece, wx, wy, insert_if_seg_hit=True)
                    if h_idx >= 0:
                        self.is_dragging = True
                        self.stretch_active_piece = sel_piece
                        self.stretch_active_handle = h_idx
                        self.stretch_drag_start = QPointF(wx, wy)
                        self.stretch_orig_boundary = sel_piece.get_boundary_world().copy()
                        self.stretch_cur_boundary = self.stretch_orig_boundary.copy()
                        self.stretch_orig_corners = sel_piece.get_corners_world().copy()
                        self.stretch_cur_corners = self.stretch_orig_corners.copy()
                        self.stretch_snap_info = None
                        self.update()
                        return

                # If didn't click on handle, check clicking on another piece to switch active stretch target
                clicked_piece = None
                for p in reversed(self.pieces):
                    if p.x <= wx < p.x + p.w and p.y <= wy < p.y + p.h:
                        clicked_piece = p
                        break
                if clicked_piece:
                    self.selection = [clicked_piece.group_id]
                    for p in self.pieces:
                        p.is_selected = (p.group_id == clicked_piece.group_id)
                    self.stretch_active_piece = clicked_piece
                    self.stretch_active_handle = -1
                    self.stretch_cur_boundary = None
                    self.stretch_orig_boundary = None
                    self.stretch_cur_corners = None
                    self.stretch_orig_corners = None
                    self.selectionChanged.emit(self.selection)
                    self.update()
                return

            if self.current_state == self.STATE_OVERLAP_ADJUST:
                self.is_dragging = True
                self.drag_start_pos = QPointF(wx, wy)
                return

            if self.current_state == self.STATE_NORMAL:
                clicked_piece = None
                for p in reversed(self.pieces):
                    if p.x <= wx < p.x + p.w and p.y <= wy < p.y + p.h:
                        clicked_piece = p
                        break

                if clicked_piece:
                    self.is_dragging = True
                    self.drag_start_pos = pos
                    self.last_mouse_pos = pos
                    gid = clicked_piece.group_id

                    if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                        if gid in self.selection:
                            self.selection.remove(gid)
                        else:
                            self.selection.append(gid)
                    else:
                        if gid not in self.selection:
                            self.selection = [gid]

                    for p in self.pieces:
                        p.is_selected = p.group_id in self.selection
                    self.selectionChanged.emit(self.selection)
                else:
                    if not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                        self.deselect_all()
                self.update()

    def mouseMoveEvent(self, event):
        pos = event.position()
        vx, vy = pos.x(), pos.y()
        wx, wy = self.view_to_world(vx, vy)

        if self.is_panning:
            dx = vx - self.last_mouse_pos.x()
            dy = vy - self.last_mouse_pos.y()
            self.view_offset_x -= dx / self.scale
            self.view_offset_y -= dy / self.scale
            self.last_mouse_pos = pos
            self.update()
            return

        if self.is_dragging:
            if self.current_state == self.STATE_CAPTURE_REGION:
                if self.capture_drag_start:
                    dwx = wx - self.capture_drag_start.x()
                    dwy = wy - self.capture_drag_start.y()
                    h = self.capture_active_handle

                    if h == 'new':
                        x1 = min(self.capture_drag_start.x(), wx)
                        y1 = min(self.capture_drag_start.y(), wy)
                        x2 = max(self.capture_drag_start.x(), wx)
                        y2 = max(self.capture_drag_start.y(), wy)
                        self.capture_rect_world = QRectF(x1, y1, x2 - x1, y2 - y1)
                    elif h == 'move' and self.capture_rect_world:
                        self.capture_rect_world.translate(dwx, dwy)
                        self.capture_drag_start = QPointF(wx, wy)
                    elif self.capture_rect_world:
                        r = self.capture_rect_world
                        if h == 'tl':
                            self.capture_rect_world = QRectF(wx, wy, r.right() - wx, r.bottom() - wy).normalized()
                        elif h == 'tr':
                            self.capture_rect_world = QRectF(r.left(), wy, wx - r.left(), r.bottom() - wy).normalized()
                        elif h == 'bl':
                            self.capture_rect_world = QRectF(wx, r.top(), r.right() - wx, wy - r.top()).normalized()
                        elif h == 'br':
                            self.capture_rect_world = QRectF(r.left(), r.top(), wx - r.left(), wy - r.top()).normalized()
                        elif h == 'top':
                            self.capture_rect_world = QRectF(r.left(), wy, r.width(), r.bottom() - wy).normalized()
                        elif h == 'bottom':
                            self.capture_rect_world = QRectF(r.left(), r.top(), r.width(), wy - r.top()).normalized()
                        elif h == 'left':
                            self.capture_rect_world = QRectF(wx, r.top(), r.right() - wx, r.height()).normalized()
                        elif h == 'right':
                            self.capture_rect_world = QRectF(r.left(), r.top(), wx - r.left(), r.height()).normalized()

                    self.update()
                    return

            if self.current_state == self.STATE_STRETCH_WARP and self.stretch_active_piece and self.stretch_active_handle >= 0:
                dx = wx - self.stretch_drag_start.x()
                dy = wy - self.stretch_drag_start.y()
                b = self.stretch_orig_boundary.copy()
                k = self.stretch_active_handle

                # اعمال توزیع نرم گوسی روی پین‌های همسایه برای جلوگیری از شکستگی تیز و موج‌دار شدن لبه
                N_pts = len(b)
                for idx_p in range(N_pts):
                    # فاصله حلقوی روی محیط مرز
                    d_idx = min(abs(idx_p - k), N_pts - abs(idx_p - k))
                    if d_idx <= 4:
                        w_falloff = math.exp(-(d_idx ** 2) / 4.5)
                        b[idx_p, 0] = self.stretch_orig_boundary[idx_p, 0] + dx * w_falloff
                        b[idx_p, 1] = self.stretch_orig_boundary[idx_p, 1] + dy * w_falloff

                b, snap_info = self._apply_magnetic_snap(b, k, self.stretch_active_piece)
                self.stretch_cur_boundary = b
                corners = getattr(self.stretch_active_piece, 'corner_indices', [0, 20, 40, 60])
                self.stretch_cur_corners = b[corners].copy()
                self.stretch_snap_info = snap_info
                self.update()
                return

            if self.current_state == self.STATE_OVERLAP_ADJUST and self.overlap_info:
                dx = wx - self.drag_start_pos.x()
                dy = wy - self.drag_start_pos.y()
                ix, iy = self.overlap_info["initial_pos"]
                if self.overlap_info["axis"] == 'x':
                    self.overlap_info["current_pos"] = (int(ix + dx), iy)
                else:
                    self.overlap_info["current_pos"] = (ix, int(iy + dy))
                self.overlap_preview_timer.start(75)
                self.update()
                return

            if self.current_state == self.STATE_NORMAL:
                dx = (vx - self.last_mouse_pos.x()) / self.scale
                dy = (vy - self.last_mouse_pos.y()) / self.scale
                for p in self.pieces:
                    if p.group_id in self.selection:
                        p.translate(dx, dy)
                self.last_mouse_pos = pos
                self.update()
                return

        # Cursor and handle hover feedback in Capture Region Mode
        if self.current_state == self.STATE_CAPTURE_REGION and not self.is_dragging:
            h = self._hit_test_capture_handles(vx, vy)
            if h in ('btn_confirm', 'btn_all', 'btn_cancel'):
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            elif h in ('tl', 'br'):
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif h in ('tr', 'bl'):
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)
            elif h in ('top', 'bottom'):
                self.setCursor(Qt.CursorShape.SizeVerCursor)
            elif h in ('left', 'right'):
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif h == 'move':
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)
            return

        # Cursor and handle hover feedback in Stretch Mode
        if self.current_state == self.STATE_STRETCH_WARP and not self.is_dragging:
            sel_piece = next((p for p in self.pieces if p.group_id in self.selection), None)
            if sel_piece is None and self.stretch_active_piece:
                sel_piece = self.stretch_active_piece
            if sel_piece:
                h_idx, _ = self._hit_test_stretch_handles(sel_piece, wx, wy, insert_if_seg_hit=False)
                corners = getattr(sel_piece, 'corner_indices', [0, 20, 40, 60])
                if h_idx in corners:
                    self.setCursor(Qt.CursorShape.SizeFDiagCursor)
                elif h_idx >= 0:
                    self.setCursor(Qt.CursorShape.CrossCursor)
                else:
                    self.unsetCursor()
            else:
                self.unsetCursor()

        # Hover tracking in normal mode
        cur_hover = -1
        for p in reversed(self.pieces):
            if p.x <= wx < p.x + p.w and p.y <= wy < p.y + p.h:
                cur_hover = p.id
                break
        if cur_hover != self.hovered_piece_id:
            self.hovered_piece_id = cur_hover
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.is_panning = False
        elif event.button() == Qt.MouseButton.LeftButton:
            if self.current_state == self.STATE_CAPTURE_REGION and self.is_dragging:
                self.is_dragging = False
                self.capture_active_handle = None
                if self.capture_rect_world:
                    self.capture_rect_world = self.capture_rect_world.normalized()
                self.update()
                return
            if self.current_state == self.STATE_STRETCH_WARP and self.is_dragging:
                self.is_dragging = False
                if self.stretch_active_piece and self.stretch_cur_boundary is not None and self.stretch_active_handle >= 0:
                    self._apply_stretch_warp(self.stretch_active_piece, self.stretch_cur_boundary)
                self.stretch_active_handle = -1
                self.stretch_cur_boundary = None
                self.stretch_orig_boundary = None
                self.stretch_cur_corners = None
                self.stretch_orig_corners = None
                self.stretch_snap_info = None
                self.update()
                return
            if self.current_state == self.STATE_OVERLAP_ADJUST and self.is_dragging:
                self.is_dragging = False
                self.overlap_preview_timer.stop()
                self._update_preview_from_current_pos()
                self.update()
                return
            self.is_dragging = False

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        zoom = 1.12 if delta > 0 else 1.0 / 1.12
        pos = event.position()
        wx, wy = self.view_to_world(pos.x(), pos.y())
        self.scale = max(0.05, min(10.0, self.scale * zoom))
        nwx, nwy = self.view_to_world(pos.x(), pos.y())
        self.view_offset_x += (wx - nwx)
        self.view_offset_y += (wy - nwy)
        self.update()

    def keyPressEvent(self, event):
        if self.is_busy:
            return
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.trigger_hotkey_hud('ESC', self.tr('hud_reset'))
            self._reset_modes()
        elif key == Qt.Key.Key_Z:
            self.undo()
        elif key == Qt.Key.Key_Y:
            self.redo()
        elif key == Qt.Key.Key_D:
            self.delete_selected()
        elif key == Qt.Key.Key_A:
            self.select_all()
        elif key == Qt.Key.Key_N:
            self.initiate_smart_snap()
        elif key == Qt.Key.Key_G:
            self.initiate_auto_align()
        elif key == Qt.Key.Key_E:
            self.initiate_elastic_warp()
        elif key == Qt.Key.Key_W:
            if event.modifiers() & Qt.KeyboardModifier.AltModifier:
                self.reset_piece_stretch()
            else:
                self.toggle_stretch_mode()
        elif key == Qt.Key.Key_M:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.initiate_overlap_mode()
            elif self.current_state == self.STATE_OVERLAP_ADJUST:
                self.finalize_overlap()
            elif len(self.selection) >= 2:
                self.initiate_batch_merge()
            elif len(self.selection) == 1:
                self.statusChanged.emit(self.tr('status_merge_min2'))
            else:
                self.statusChanged.emit(self.tr('status_merge_no_sel'))
        elif key == Qt.Key.Key_F:
            self.initiate_batch_merge()
        elif key == Qt.Key.Key_S:
            parent_win = self.window()
            if parent_win and hasattr(parent_win, 'save_project'):
                parent_win.save_project()
            else:
                self.trigger_hotkey_hud('S', self.tr('hud_save'))
        elif key == Qt.Key.Key_P:
            if self.current_state == self.STATE_CAPTURE_REGION:
                self.confirm_region_capture()
            else:
                self.start_region_capture()
        elif key == Qt.Key.Key_L:
            spacing = 40
            cols_val = 0
            mode_str = 'matrix'
            parent_win = self.window()
            if parent_win:
                if hasattr(parent_win, 'spin_spacing'):
                    spacing = parent_win.spin_spacing.value()
                if hasattr(parent_win, 'spin_cols'):
                    cols_val = parent_win.spin_cols.value()
                if hasattr(parent_win, 'combo_arrange_mode'):
                    idx = parent_win.combo_arrange_mode.currentIndex()
                    mode_str = 'matrix' if idx == 0 else ('row' if idx == 1 else 'col')
            self.arrange_pieces(spacing=spacing, cols=cols_val, mode=mode_str)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.current_state == self.STATE_CAPTURE_REGION:
                self.confirm_region_capture()
            elif self.current_state == self.STATE_OVERLAP_ADJUST:
                self.finalize_overlap()
        elif key == Qt.Key.Key_Escape:
            if self.current_state == self.STATE_CAPTURE_REGION:
                self.current_state = self.STATE_NORMAL
                self.capture_rect_world = None
                self.unsetCursor()
                self.update()
            elif self.current_state == self.STATE_STRETCH_WARP:
                self._reset_modes()
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            modifiers = event.modifiers()
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                step = 5
            elif modifiers & Qt.KeyboardModifier.ControlModifier:
                step = 15
            else:
                step = 1
            dx, dy = 0, 0
            key_name = ""
            if key == Qt.Key.Key_Left:
                dx, key_name = -step, "←"
            elif key == Qt.Key.Key_Right:
                dx, key_name = step, "→"
            elif key == Qt.Key.Key_Up:
                dy, key_name = -step, "↑"
            elif key == Qt.Key.Key_Down:
                dy, key_name = step, "↓"

            if self.current_state == self.STATE_OVERLAP_ADJUST and self.overlap_info:
                ix, iy = self.overlap_info["current_pos"]
                self.overlap_info["current_pos"] = (ix + dx, iy + dy)
                self.trigger_hotkey_hud(key_name, self.tr('hud_micro_nudge').format(dx=dx, dy=dy))
                self.overlap_preview_timer.start(75)
                self.update()
            elif self.current_state == self.STATE_NORMAL and self.selection:
                for p in self.pieces:
                    if p.group_id in self.selection:
                        p.translate(dx, dy)
                self.trigger_hotkey_hud(key_name, self.tr('hud_piece_nudge').format(dx=dx, dy=dy))
                self.update()
        else:
            super().keyPressEvent(event)

    # ================== Canvas Rendering ==================
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        # 1. Clean White / Technical Light Grid
        self._draw_light_grid(painter)

        # 2. Draw pieces or Overlap Preview
        if self.current_state == self.STATE_OVERLAP_ADJUST and self.overlap_info:
            base_gid = self.overlap_info["base_gid"]
            top_gid = self.overlap_info["top_gid"]
            base_p = next((p for p in self.pieces if p.group_id == base_gid), None)
            top_p = next((p for p in self.pieces if p.group_id == top_gid), None)

            for p in self.pieces:
                if p.group_id not in (base_gid, top_gid):
                    self._draw_piece(painter, p)

            # Live interactive zero-lag hardware compositing when dragging, or when preview is pending, or in ghost mode
            if (self.is_dragging or self.temp_preview_pixmap is None or self.selection_opacity < 0.99) and base_p and top_p:
                # 1. Draw base piece completely solid
                self._draw_piece(painter, base_p, force_opacity=1.0)

                # 2. Draw top piece with live see-through opacity at dragged position
                base_orig = self.overlap_info["base_origin"]
                cur_pos = self.overlap_info["current_pos"]
                top_world_x = base_orig[0] + cur_pos[0]
                top_world_y = base_orig[1] + cur_pos[1]

                vx, vy = self.world_to_view(top_world_x, top_world_y)
                vw = top_p.w * self.scale
                vh = top_p.h * self.scale
                top_rect = QRectF(vx, vy, vw, vh)

                pix = self.get_piece_pixmap(top_p)
                painter.save()
                active_alpha = self.selection_opacity if self.selection_opacity < 0.99 else 0.72
                painter.setOpacity(active_alpha)
                painter.drawPixmap(top_rect.toRect(), pix)
                painter.restore()

                # Highlight top piece boundary with dashed cyan/blue outline
                painter.setPen(QPen(QColor("#0284C7"), 2.5, Qt.PenStyle.DashLine))
                painter.drawRect(top_rect)

                # Live offset pill indicator
                badge_str = f"Δx: {cur_pos[0]:+d}px  Δy: {cur_pos[1]:+d}px  [Enter: ثبت ادغام]"
                painter.save()
                painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
                fm = painter.fontMetrics()
                bw = fm.horizontalAdvance(badge_str) + 24
                bh = 28
                bx = max(15.0, min(float(self.width() - bw - 15), vx + 10))
                by = max(15.0, min(float(self.height() - bh - 15), vy + 10))
                badge_rect = QRectF(bx, by, bw, bh)
                painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
                painter.setPen(QPen(QColor("#38BDF8"), 1.5))
                painter.drawRoundedRect(badge_rect, 6, 6)
                painter.setPen(QColor("#F8FAFC"))
                painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_str)
                painter.restore()
            elif self.temp_preview_pixmap:
                base_orig = self.overlap_info.get('base_origin', (0, 0))
                offset = self.overlap_info.get('final_offset', (0, 0))
                vx, vy = self.world_to_view(base_orig[0] + offset[0], base_orig[1] + offset[1])
                vw = self.temp_preview_pixmap.width() * self.scale
                vh = self.temp_preview_pixmap.height() * self.scale

                target_rect = QRectF(vx, vy, vw, vh)
                painter.drawPixmap(target_rect.toRect(), self.temp_preview_pixmap)
                painter.setPen(QPen(QColor("#2563EB"), 2.5, Qt.PenStyle.DashLine))
                painter.drawRect(target_rect)
        else:
            for p in sorted(self.pieces, key=lambda x: x.is_selected):
                self._draw_piece(painter, p)

        # 3. Draw Corner & Edge Stretch Handles and Boundary Reticle
        self._draw_stretch_handles(painter)

        # 4. Draw Capture Region Snipping Frame
        self._draw_capture_region_overlay(painter)

        # 5. Floating Hotkey HUD Overlay on Canvas
        self._draw_light_hotkey_hud(painter)

    def _draw_stretch_handles(self, painter):
        if self.current_state != self.STATE_STRETCH_WARP:
            return
        sel_piece = next((p for p in self.pieces if p.group_id in self.selection), None)
        if sel_piece is None and self.stretch_active_piece:
            sel_piece = self.stretch_active_piece
        if sel_piece is None:
            return

        if self.is_dragging and self.stretch_cur_boundary is not None:
            boundary = self.stretch_cur_boundary
        else:
            boundary = sel_piece.get_boundary_world()

        N = len(boundary)
        v_pts = [self.world_to_view(boundary[i, 0], boundary[i, 1]) for i in range(N)]
        poly = QPolygonF([QPointF(x, y) for x, y in v_pts])

        painter.save()
        # 1. Subtle violet deformation polygon fill & dashed border
        painter.setBrush(QBrush(QColor(99, 102, 241, 35)))
        painter.setPen(QPen(QColor("#6366F1"), 2.0, Qt.PenStyle.DashLine))
        painter.drawPolygon(poly)

        # 2. Magnetic Snap Visual Indicator
        if self.stretch_snap_info:
            snap_type, snap_target = self.stretch_snap_info
            if snap_type == 'point':
                sx, sy = self.world_to_view(snap_target[0], snap_target[1])
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor("#10B981"), 3.0))
                painter.drawEllipse(QPointF(sx, sy), 10, 10)
                painter.setPen(QPen(QColor("#34D399"), 1.5, Qt.PenStyle.DotLine))
                painter.drawEllipse(QPointF(sx, sy), 16, 16)

                badge_text = "⚡ فیت لبه" if self.current_lang == 'fa' else "⚡ Edge Fit Snap"
                painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                painter.setBrush(QBrush(QColor("#059669")))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(QRectF(sx + 14, sy - 12, 85, 22), 4, 4)
                painter.setPen(QColor("#FFFFFF"))
                painter.drawText(QRectF(sx + 14, sy - 12, 85, 22), Qt.AlignmentFlag.AlignCenter, badge_text)

        # 3. Perimeter Control Pins
        corner_indices = getattr(sel_piece, 'corner_indices', [0, 20, 40, 60])
        corner_set = set(corner_indices)

        for i in range(N):
            if i in corner_set:
                continue
            cx, cy = v_pts[i]
            is_active = (self.stretch_active_handle == i)
            r = 6.5 if is_active else 3.8

            if is_active:
                painter.setBrush(QBrush(QColor("#06B6D4")))
                painter.setPen(QPen(QColor("#FFFFFF"), 2.0))
            else:
                painter.setBrush(QBrush(QColor("#0284C7")))
                painter.setPen(QPen(QColor("#FFFFFF"), 1.2))
            painter.drawEllipse(QPointF(cx, cy), r, r)

        # 4. Corner Handles (Distinctive anchor rings)
        for idx in corner_indices:
            if idx >= N:
                continue
            cx, cy = v_pts[idx]
            is_active = (self.stretch_active_handle == idx)
            r = 8.5 if is_active else 6.5

            painter.setBrush(QBrush(QColor("#FFFFFF")))
            painter.setPen(QPen(QColor("#4F46E5"), 2.5))
            painter.drawEllipse(QPointF(cx, cy), r, r)

            painter.setBrush(QBrush(QColor("#6366F1")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(cx, cy), r * 0.45, r * 0.45)

        # 5. Top Information Header Badge
        top_idx_1 = corner_indices[0] if len(corner_indices) > 0 else 0
        top_idx_2 = corner_indices[1] if len(corner_indices) > 1 else min(N - 1, 20)
        top_y = min(v_pts[top_idx_1][1], v_pts[top_idx_2][1])
        mid_top_x = (v_pts[top_idx_1][0] + v_pts[top_idx_2][0]) / 2.0
        badge_w = 340 if self.current_lang == 'fa' else 310
        badge_h = 26
        badge_y = max(6.0, top_y - badge_h - 10)
        badge_rect = QRectF(mid_top_x - badge_w / 2.0, badge_y, badge_w, badge_h)

        painter.setBrush(QBrush(QColor(67, 56, 202, 235)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(badge_rect, 6, 6)

        badge_text = "🔲 کشش مستقل نقاط (W) | هر نقطه کاملاً ثابت | کلیک روی خط = پین جدید" if self.current_lang == 'fa' else "🔲 Independent Point Stretch (W) | Points are Fixed | Click Edge = Add Pin"
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor("#FFFFFF"))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_text)

        painter.restore()

    def _draw_light_grid(self, painter):
        w = self.width()
        h = self.height()
        # Clean subtle off-white background
        painter.fillRect(0, 0, w, h, QColor("#F8FAFC"))

        grid_size = int(48 * self.scale)
        if grid_size > 12:
            painter.setPen(QPen(QColor("#E2E8F0"), 1))
            start_x = int((-self.view_offset_x * self.scale) % grid_size)
            start_y = int((-self.view_offset_y * self.scale) % grid_size)
            for x in range(start_x, w, grid_size):
                painter.drawLine(x, 0, x, h)
            for y in range(start_y, h, grid_size):
                painter.drawLine(0, y, w, y)

    def _draw_piece(self, painter, piece, force_opacity=None):
        vx, vy = self.world_to_view(piece.x, piece.y)
        vw = piece.w * self.scale
        vh = piece.h * self.scale

        if vx + vw < 0 or vy + vh < 0 or vx > self.width() or vy > self.height():
            return

        pix = self.get_piece_pixmap(piece)
        target_rect = QRectF(vx, vy, vw, vh)

        painter.save()
        if force_opacity is not None:
            painter.setOpacity(force_opacity)
        elif piece.is_selected and self.selection_opacity < 0.99:
            if len(self.selection) == 1 or (len(self.selection) >= 2 and piece.group_id != self.selection[0]):
                painter.setOpacity(self.selection_opacity)

        painter.drawPixmap(target_rect.toRect(), pix)
        painter.setOpacity(1.0)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        if piece.is_selected:
            # Highlight selected piece with a clean boundary outline (NO solid fill)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#2563EB"), 2.5))
            if hasattr(piece, 'boundary_world') and piece.boundary_world is not None:
                b = piece.boundary_world
                poly = QPolygonF([QPointF(*self.world_to_view(b[i, 0], b[i, 1])) for i in range(len(b))])
                painter.drawPolygon(poly)
            else:
                painter.drawRect(target_rect)

            # Order Badge (1, 2, ...) in top-left corner
            if piece.group_id in self.selection:
                idx = self.selection.index(piece.group_id) + 1
                if hasattr(piece, 'boundary_world') and piece.boundary_world is not None:
                    tl_vx, tl_vy = self.world_to_view(piece.boundary_world[0, 0], piece.boundary_world[0, 1])
                else:
                    tl_vx, tl_vy = vx, vy
                badge_rect = QRectF(tl_vx + 6, tl_vy + 6, 26, 26)
                painter.setBrush(QBrush(QColor("#EA580C")))
                painter.setPen(QPen(QColor("#FFFFFF"), 2.0))
                painter.drawEllipse(badge_rect)
                painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
                painter.setPen(QColor("#FFFFFF"))
                painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, str(idx))
        elif piece.id == self.hovered_piece_id and self.current_state == self.STATE_NORMAL:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#06B6D4"), 2.0))
            if hasattr(piece, 'boundary_world') and piece.boundary_world is not None:
                b = piece.boundary_world
                poly = QPolygonF([QPointF(*self.world_to_view(b[i, 0], b[i, 1])) for i in range(len(b))])
                painter.drawPolygon(poly)
            else:
                painter.drawRect(target_rect)

        # Sequence Number / Name Badge (when not selected)
        if not piece.is_selected:
            label_text = ""
            if hasattr(piece, 'filename') and piece.filename:
                label_text = os.path.splitext(os.path.basename(piece.filename))[0]
            else:
                label_text = f"#{piece.id + 1}"

            if label_text:
                if hasattr(piece, 'boundary_world') and piece.boundary_world is not None:
                    tl_vx, tl_vy = self.world_to_view(piece.boundary_world[0, 0], piece.boundary_world[0, 1])
                else:
                    tl_vx, tl_vy = vx, vy
                tag_w = max(26.0, float(len(label_text) * 7.5 + 12.0))
                tag_h = 20.0
                tag_rect = QRectF(tl_vx + 6, tl_vy + 6, tag_w, tag_h)
                painter.setBrush(QBrush(QColor(15, 23, 42, 175)))
                painter.setPen(QPen(QColor(255, 255, 255, 180), 1.0))
                painter.drawRoundedRect(tag_rect, 5, 5)
                painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                painter.setPen(QColor("#FFFFFF"))
                painter.drawText(tag_rect, Qt.AlignmentFlag.AlignCenter, label_text)

        painter.restore()

    def _draw_light_hotkey_hud(self, painter):
        """Draw modern Light Theme Hotkey HUD badge on Canvas with vibrant accents"""
        if not self.hud_key or self.hud_alpha <= 0.01:
            return

        painter.save()
        hud_w, hud_h = 340, 72
        margin_right = 24
        margin_bottom = 24
        hx = self.width() - hud_w - margin_right
        hy = self.height() - hud_h - margin_bottom

        alpha_int = int(255 * self.hud_alpha)

        # Card Background: Pure White with Soft Colored Border & Shadow
        card_rect = QRectF(hx, hy, hud_w, hud_h)
        painter.setBrush(QBrush(QColor(255, 255, 255, int(245 * self.hud_alpha))))
        painter.setPen(QPen(QColor(37, 99, 235, int(200 * self.hud_alpha)), 2.0))
        painter.drawRoundedRect(card_rect, 14, 14)

        # Keycap Badge (Gradient Blue / Violet)
        cap_size = 48
        cx = hx + 14
        cy = hy + 12
        cap_rect = QRectF(cx, cy, cap_size, cap_size)

        cap_grad = QLinearGradient(cx, cy, cx + cap_size, cy + cap_size)
        cap_grad.setColorAt(0.0, QColor(37, 99, 235, alpha_int))
        cap_grad.setColorAt(1.0, QColor(79, 70, 229, alpha_int))
        painter.setBrush(QBrush(cap_grad))
        painter.setPen(QPen(QColor(255, 255, 255, alpha_int), 1.5))
        painter.drawRoundedRect(cap_rect, 10, 10)

        # Key Text
        painter.setPen(QColor(255, 255, 255, alpha_int))
        painter.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        painter.drawText(cap_rect, Qt.AlignmentFlag.AlignCenter, self.hud_key)

        # Action Details
        painter.setPen(QColor(100, 116, 139, alpha_int))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.drawText(int(hx + 74), int(hy + 28), self.tr('hud_header'))

        painter.setPen(QColor(15, 23, 42, alpha_int))
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.drawText(int(hx + 74), int(hy + 52), self.hud_label)

        # Recent Ticker Trail
        valid_hist = [k for k, t in self.hud_history if (datetime.datetime.now() - t).total_seconds() < 7.0]
        if len(valid_hist) > 1:
            ticker_text = self.tr('hud_recent') + " > ".join([f"[{k}]" for k in valid_hist[-4:]])
            painter.setPen(QColor(79, 70, 229, int(220 * self.hud_alpha)))
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.drawText(int(hx), int(hy - 8), ticker_text)

        painter.restore()


# ================== Language Switcher Toolbar Widget ==================
class LanguageSwitcherWidget(QWidget):
    languageChanged = pyqtSignal(str)

    def __init__(self, current_lang='fa', parent=None):
        super().__init__(parent)
        self.current_lang = current_lang
        self.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.setFixedHeight(32)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        # Globe Toggle Button
        self.btn_globe = QToolButton(self)
        self.btn_globe.setIcon(create_globe_language_icon(22))
        self.btn_globe.setIconSize(QSize(18, 18))
        self.btn_globe.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_globe.setToolTip("Switch Language" if self.current_lang == 'en' else "تغییر زبان")
        self.btn_globe.setFixedSize(26, 26)
        self.btn_globe.setStyleSheet("""
            QToolButton {
                background: transparent;
                border: none;
                padding: 2px;
                border-radius: 6px;
            }
            QToolButton:hover {
                background-color: #EFF6FF;
            }
        """)
        self.btn_globe.clicked.connect(self.toggle_language)
        layout.addWidget(self.btn_globe)

        self.lbl_title = QLabel("Language:" if self.current_lang == 'en' else "زبان:")
        self.lbl_title.setStyleSheet("font-weight: 700; color: #475569; font-size: 11px; font-family: 'Segoe UI', Tahoma, sans-serif;")
        layout.addWidget(self.lbl_title)

        # Segmented Pill Switcher
        self.pill_container = QFrame()
        self.pill_container.setFixedHeight(26)
        self.pill_container.setStyleSheet("""
            QFrame {
                background-color: #F1F5F9;
                border: 1px solid #CBD5E1;
                border-radius: 13px;
            }
        """)
        pill_layout = QHBoxLayout(self.pill_container)
        pill_layout.setContentsMargins(2, 1, 2, 1)
        pill_layout.setSpacing(2)
        pill_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self.btn_en = QPushButton("English")
        self.btn_fa = QPushButton("فارسی")

        for btn in (self.btn_en, self.btn_fa):
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedHeight(22)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.btn_en.clicked.connect(lambda: self.set_language('en'))
        self.btn_fa.clicked.connect(lambda: self.set_language('fa'))

        pill_layout.addWidget(self.btn_en)
        pill_layout.addWidget(self.btn_fa)
        layout.addWidget(self.pill_container)

        self._update_styles()

    def toggle_language(self):
        new_lang = 'fa' if self.current_lang == 'en' else 'en'
        self.set_language(new_lang)

    def set_language(self, lang):
        if lang != self.current_lang:
            self.current_lang = lang
            self._update_styles()
            self.languageChanged.emit(self.current_lang)

    def _update_styles(self):
        active_style = """
            QPushButton {
                background-color: #2563EB;
                color: #FFFFFF;
                border: none;
                border-radius: 11px;
                font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
                font-weight: 700;
                font-size: 11px;
                padding: 2px 10px;
                text-align: center;
            }
        """
        inactive_style = """
            QPushButton {
                background-color: transparent;
                color: #64748B;
                border: none;
                border-radius: 11px;
                font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
                font-weight: 600;
                font-size: 11px;
                padding: 2px 8px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #E2E8F0;
                color: #0F172A;
            }
        """
        if self.current_lang == 'en':
            self.btn_en.setChecked(True)
            self.btn_fa.setChecked(False)
            self.btn_en.setStyleSheet(active_style)
            self.btn_fa.setStyleSheet(inactive_style)
            self.btn_globe.setToolTip("Switch Language")
            self.lbl_title.setText("Language:")
        else:
            self.btn_en.setChecked(False)
            self.btn_fa.setChecked(True)
            self.btn_en.setStyleSheet(inactive_style)
            self.btn_fa.setStyleSheet(active_style)
            self.btn_globe.setToolTip("تغییر زبان")
            self.lbl_title.setText("زبان:")


PERSIAN_TO_LATIN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٫،", "0123456789..")


# ================== Sidebar Mouse-Only & Focus Isolation Filter ==================
class SidebarMouseOnlyFilter(QObject):
    """
    Event filter for sidebar/dock controls:
    1. Blocks wheel events on spinboxes, comboboxes, and sliders from modifying values,
       redirecting wheel scroll directly to the dock QScrollArea.
    2. Allows direct keyboard entry into editable numeric line-edits (spinboxes) when clicked,
       supporting both English and Persian digits smoothly.
    3. For non-editable widgets (buttons, sliders, combos, etc.), prevents stealing keyboard focus
       and redirects key events to the canvas so hotkeys always operate on the workspace.
    """
    def __init__(self, scroll_area, canvas, parent=None):
        super().__init__(parent)
        self.scroll_area = scroll_area
        self.canvas = canvas

    def eventFilter(self, obj, event):
        t = event.type()
        if t == QEvent.Type.Wheel:
            # Forward wheel scroll directly to dock scroll_area so the sidebar page scrolls,
            # leaving the widget value completely untouched.
            if self.scroll_area:
                self.scroll_area.wheelEvent(event)
            return True

        # Check if target is an editable input (QLineEdit, CalibrationField, QSpinBox, QDoubleSpinBox)
        is_editable_input = (
            (isinstance(obj, QLineEdit) and not obj.isReadOnly()) or
            isinstance(obj, (QSpinBox, QDoubleSpinBox)) or
            (hasattr(obj, 'lineEdit') and obj.lineEdit() is not None)
        )

        if is_editable_input:
            if t == QEvent.Type.KeyPress:
                # Enter or Return commits value, clears focus, and restores focus to canvas
                if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                    if hasattr(obj, 'interpretText'):
                        obj.interpretText()
                    elif hasattr(obj, 'parent') and isinstance(obj.parent(), (QSpinBox, QDoubleSpinBox)):
                        obj.parent().interpretText()
                    obj.clearFocus()
                    if self.canvas:
                        self.canvas.setFocus()
                    return True
                elif event.key() == Qt.Key.Key_Escape:
                    obj.clearFocus()
                    if self.canvas:
                        self.canvas.setFocus()
                    return True

                # Transparently translate Persian keyboard numerals to standard ASCII digits
                txt = event.text()
                if txt and any(ch in "۰۱۲۳۴۵۶۷۸۹٫،" for ch in txt):
                    converted = txt.translate(PERSIAN_TO_LATIN_DIGITS)
                    if isinstance(obj, QLineEdit):
                        obj.insert(converted)
                    elif hasattr(obj, 'lineEdit') and obj.lineEdit():
                        obj.lineEdit().insert(converted)
                    return True

                # Let normal text editing keys pass through natively
                return False

            elif t == QEvent.Type.KeyRelease:
                return False

            elif t in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease):
                return False

        if t in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            # Non-input controls (buttons, sliders, combos): forward key events to canvas
            if self.canvas:
                if t == QEvent.Type.KeyPress:
                    self.canvas.keyPressEvent(event)
                else:
                    self.canvas.keyReleaseEvent(event)
            return True

        elif t == QEvent.Type.MouseButtonRelease:
            # After clicking buttons or dragging sliders, return focus to canvas
            if self.canvas:
                QTimer.singleShot(10, self.canvas.setFocus)

        return super().eventFilter(obj, event)


# ================== Direct-Keyboard Calibration Field ==================
class CalibrationField(QLineEdit):
    """
    Dedicated numeric input field for physical X-ray calibration (SDD, SOD, Pitch).
    Allows entering ANY arbitrary number directly from the keyboard without up/down stepper arrows.
    Supports both English (0-9) and Persian (۰-۹) digits and decimal separators cleanly.
    """
    valueChanged = pyqtSignal(float)

    def __init__(self, parent=None, placeholder=""):
        super().__init__(parent)
        self._value = 0.0
        self.setPlaceholderText(placeholder)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLineEdit {
                padding: 6px 10px;
                border: 1.5px solid #CBD5E1;
                border-radius: 6px;
                background-color: #FFFFFF;
                font-weight: 700;
                color: #0F172A;
                font-size: 11px;
                selection-background-color: #2563EB;
                selection-color: #FFFFFF;
            }
            QLineEdit:focus {
                border: 2px solid #2563EB;
                background-color: #F8FAFC;
            }
        """)
        self.textChanged.connect(self._handle_text_changed)
        self.editingFinished.connect(self._handle_editing_finished)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        event.accept()
        QTimer.singleShot(0, self.selectAll)

    def keyPressEvent(self, event):
        txt = event.text()
        if txt and any(ch in "۰۱۲۳۴۵۶۷۸۹٫،" for ch in txt):
            converted = txt.translate(PERSIAN_TO_LATIN_DIGITS)
            self.insert(converted)
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._handle_editing_finished()
            self.clearFocus()
            win = self.window()
            if hasattr(win, 'canvas') and win.canvas:
                win.canvas.setFocus()
            return
        elif event.key() == Qt.Key.Key_Escape:
            self.clearFocus()
            win = self.window()
            if hasattr(win, 'canvas') and win.canvas:
                win.canvas.setFocus()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event):
        # Ignore mouse wheel so the sidebar scrolls smoothly without changing calibration values
        event.ignore()

    def setSpecialValueText(self, text):
        self.setPlaceholderText(text)

    def value(self):
        return self._value

    def setValue(self, val):
        try:
            f = float(val) if val is not None else 0.0
        except (ValueError, TypeError):
            f = 0.0
        self._value = f
        self.blockSignals(True)
        if f <= 0.0:
            self.setText("")
        else:
            self.setText(f"{f:g}")
        self.blockSignals(False)

    def _parse_text(self, txt):
        if not txt:
            return 0.0
        clean = txt.translate(PERSIAN_TO_LATIN_DIGITS).strip()
        clean = "".join(ch for ch in clean if ch in "0123456789.")
        try:
            return float(clean) if clean else 0.0
        except ValueError:
            return 0.0

    def _handle_text_changed(self, txt):
        val = self._parse_text(txt)
        if val != self._value:
            self._value = val
            self.valueChanged.emit(val)

    def _handle_editing_finished(self):
        val = self._parse_text(self.text())
        self._value = val
        self.blockSignals(True)
        if val <= 0.0:
            self.setText("")
        else:
            self.setText(f"{val:g}")
        self.blockSignals(False)
        self.valueChanged.emit(val)

    # Spinbox compatibility stubs
    def setRange(self, min_val, max_val): pass
    def setDecimals(self, dec): pass
    def setSingleStep(self, step): pass
    def setKeyboardTracking(self, tracking): pass


# ================== Main Window (White Theme Graphic UI) ==================
class XRayPuzzleStudioWindow(QMainWindow):
    def __init__(self, initial_directory=INPUT_DIRECTORY):
        super().__init__()
        self.current_lang = 'fa'
        self.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.setWindowTitle(self.tr('window_title'))
        self.resize(1380, 860)
        self.setMinimumWidth(1150)
        self.directory = initial_directory
        self.current_project_path = None
        self.current_project_name = os.path.basename(os.path.normpath(initial_directory)) if initial_directory else "xray_project"

        # Crisp White Modern GUI Theme Stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F8FAFC;
            }
            QToolBar {
                background: #FFFFFF;
                border-bottom: 1px solid #E2E8F0;
                spacing: 5px;
                padding: 0px 8px;
                min-height: 48px;
                max-height: 48px;
            }
            QToolButton {
                background-color: #FFFFFF;
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                padding: 2px 7px;
                font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
                font-size: 11px;
                font-weight: 600;
                color: #1E293B;
                min-height: 32px;
                max-height: 32px;
            }
            QToolButton:hover {
                background-color: #F1F5F9;
                border-color: #2563EB;
                color: #2563EB;
            }
            QToolButton:pressed {
                background-color: #E2E8F0;
            }
            QToolBar::separator {
                width: 1px;
                background-color: #CBD5E1;
                margin: 6px 4px;
            }
            QDockWidget {
                color: #0F172A;
                font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
                font-weight: bold;
                font-size: 13px;
            }
            QDockWidget::title {
                background: #FFFFFF;
                padding: 10px;
                border-bottom: 1px solid #E2E8F0;
                color: #1E293B;
            }
            QScrollArea {
                background: #FFFFFF;
                border: none;
            }
            QGroupBox {
                border: 1px solid #E2E8F0;
                border-radius: 10px;
                margin-top: 14px;
                padding: 14px 10px 10px 10px;
                background-color: #FFFFFF;
                color: #1E293B;
                font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
                font-weight: 700;
                font-size: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
                background: #FFFFFF;
            }
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #CBD5E1;
                color: #1E293B;
                border-radius: 8px;
                padding: 9px 14px;
                font-size: 12px;
                font-weight: 600;
                font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #F1F5F9;
                border-color: #2563EB;
                color: #2563EB;
            }
            QPushButton:pressed {
                background-color: #E2E8F0;
            }
            /* Colorful Accent Buttons */
            QPushButton#btnSmartSnap {
                background-color: #EFF6FF;
                border: 1.5px solid #3B82F6;
                color: #1D4ED8;
                font-weight: 700;
            }
            QPushButton#btnSmartSnap:hover {
                background-color: #DBEAFE;
                border-color: #1D4ED8;
            }
            QPushButton#btnAutoAlign {
                background-color: #F0FDF4;
                border: 1.5px solid #10B981;
                color: #047857;
                font-weight: 700;
            }
            QPushButton#btnAutoAlign:hover {
                background-color: #DCFCE7;
                border-color: #047857;
            }
            QPushButton#btnElastic {
                background-color: #FAF5FF;
                border: 1.5px solid #8B5CF6;
                color: #6D28D9;
                font-weight: 700;
            }
            QPushButton#btnElastic:hover {
                background-color: #F3E8FF;
                border-color: #6D28D9;
            }
            QPushButton#btnStretch {
                background-color: #EEF2FF;
                border: 1.5px solid #6366F1;
                color: #4338CA;
                font-weight: 700;
            }
            QPushButton#btnStretch:hover {
                background-color: #E0E7FF;
                border-color: #4338CA;
            }
            QPushButton#btnResetStretch {
                background-color: #F8FAFC;
                border: 1px solid #CBD5E1;
                color: #64748B;
                font-weight: 600;
                text-align: center;
            }
            QPushButton#btnResetStretch:hover {
                background-color: #FEF3C7;
                border-color: #D97706;
                color: #B45309;
            }
            QPushButton#btnOverlap {
                background-color: #FFF7ED;
                border: 1.5px solid #F97316;
                color: #C2410C;
                font-weight: 700;
            }
            QPushButton#btnOverlap:hover {
                background-color: #FFEDD5;
                border-color: #C2410C;
            }
            QPushButton#btnLayerOpacity {
                background-color: #F0F9FF;
                border: 1.5px solid #0EA5E9;
                color: #0369A1;
                font-weight: 700;
            }
            QPushButton#btnLayerOpacity:hover {
                background-color: #E0F2FE;
                border-color: #0284C7;
            }
            QPushButton#btnExport {
                background-color: #FAF5FF;
                border: 1.5px solid #A855F7;
                color: #7E22CE;
                font-weight: 700;
            }
            QPushButton#btnExport:hover {
                background-color: #F3E8FF;
                border-color: #7E22CE;
            }
            QPushButton#btnArrange {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F0FDF4, stop:1 #DCFCE7);
                border: 1.5px solid #16A34A;
                color: #15803D;
                font-weight: 700;
            }
            QPushButton#btnArrange:hover {
                background-color: #BBF7D0;
                border-color: #15803D;
            }
            QPushButton#btnDelete {
                background-color: #FEF2F2;
                border: 1px solid #FCA5A5;
                color: #DC2626;
                font-weight: 600;
            }
            QPushButton#btnDelete:hover {
                background-color: #FEE2E2;
                border-color: #DC2626;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #E2E8F0;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #2563EB;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #FFFFFF;
                border: 2.5px solid #2563EB;
                width: 16px;
                margin-top: -6px;
                margin-bottom: -6px;
                border-radius: 8px;
            }
            QLabel {
                color: #334155;
                font-size: 11px;
                font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
            }
            QCheckBox {
                color: #1E293B;
                font-size: 12px;
                font-weight: 600;
                font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 5px;
                border: 1.5px solid #CBD5E1;
                background: #FFFFFF;
            }
            QCheckBox::indicator:checked {
                background: #2563EB;
                border-color: #2563EB;
            }
            QStatusBar {
                background-color: #FFFFFF;
                border-top: 1px solid #E2E8F0;
                color: #2563EB;
                font-weight: 600;
                padding: 5px;
            }
        """)

        # Canvas Widget
        self.canvas = InteractivePuzzleCanvas(self)
        self.canvas.set_language(self.current_lang)
        self.setCentralWidget(self.canvas)

        # Status Bar
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.canvas.statusChanged.connect(self.status_bar.showMessage)

        # Create Top Toolbar with Language Switcher
        self._create_top_toolbar()

        # Create GUI Control Dock
        self._create_control_dock()
        self.canvas.opacityChanged.connect(self._sync_opacity_ui)

        # Global Project Shortcuts
        self.shortcut_save = QShortcut(QKeySequence("Ctrl+S"), self)
        self.shortcut_save.activated.connect(self.save_project)
        self.shortcut_open = QShortcut(QKeySequence("Ctrl+O"), self)
        self.shortcut_open.activated.connect(self.load_project)

        # Synchronize UI texts in selected language
        self.update_ui_language()

        # Load initial images
        self.load_images_from_directory(self.directory)

    def tr(self, key):
        return TRANSLATIONS.get(self.current_lang, TRANSLATIONS['en']).get(key, key)

    def set_language(self, lang):
        if lang not in TRANSLATIONS:
            return
        self.current_lang = lang
        self.canvas.set_language(lang)
        self.lang_switcher.set_language(lang)
        self.update_ui_language()
        self.status_bar.showMessage(self.tr('status_lang_changed'))

    def update_ui_language(self):
        # 1. Window & Main Containers (Keep Left-to-Right layout direction intact!)
        if getattr(self, 'current_project_name', None):
            self.setWindowTitle(f"{self.tr('window_title')} - [{self.current_project_name}]")
        else:
            self.setWindowTitle(self.tr('window_title'))
        self.dock.setWindowTitle(self.tr('dock_title'))
        self.toolbar.setWindowTitle(self.tr('toolbar_title'))

        # 2. Language Switcher Label
        self.lang_switcher.lbl_title.setText(self.tr('lang_label'))

        # 3. Toolbar Action Buttons
        self.tb_open.setText(self.tr('btn_open_tb'))
        self.tb_open.setToolTip(self.tr('btn_open'))
        if hasattr(self, 'tb_load'):
            self.tb_load.setText(self.tr('btn_load_tb'))
            self.tb_load.setToolTip(self.tr('btn_load_project'))
        self.tb_save.setText(self.tr('btn_save_tb'))
        self.tb_save.setToolTip(self.tr('btn_save'))
        self.tb_export.setText(self.tr('btn_export_tb'))
        self.tb_export.setToolTip(self.tr('btn_export'))

        self.tb_fit.setText(self.tr('btn_fit_tb'))
        self.tb_fit.setToolTip(self.tr('btn_fit'))
        self.tb_arrange.setText(self.tr('btn_arrange_tb'))
        self.tb_arrange.setToolTip(self.tr('tip_arrange'))

        self.tb_smart_snap.setText(self.tr('btn_smart_snap_tb'))
        self.tb_smart_snap.setToolTip(self.tr('tip_smart_snap'))
        self.tb_auto_align.setText(self.tr('btn_auto_align_tb'))
        self.tb_auto_align.setToolTip(self.tr('tip_auto_align'))
        self.tb_elastic.setText(self.tr('btn_elastic_tb'))
        self.tb_elastic.setToolTip(self.tr('tip_elastic'))
        self.tb_stretch.setText(self.tr('btn_stretch_tb'))
        self.tb_stretch.setToolTip(self.tr('tip_stretch'))
        if hasattr(self, 'tb_merge'):
            self.tb_merge.setText(self.tr('btn_merge_tb'))
            self.tb_merge.setToolTip(self.tr('tip_merge'))
        self.tb_overlap.setText(self.tr('btn_overlap_tb'))
        self.tb_overlap.setToolTip(self.tr('tip_overlap'))

        self.tb_undo.setText(self.tr('btn_undo_tb'))
        self.tb_undo.setToolTip(self.tr('btn_undo'))
        self.tb_redo.setText(self.tr('btn_redo_tb'))
        self.tb_redo.setToolTip(self.tr('btn_redo'))
        self.tb_ghost.setText(self.tr('btn_ghost_tb'))
        self.tb_ghost.setToolTip(self.tr('tip_toggle_opacity'))

        # 4. Dock Group 1: Project & File Operations
        self.file_group.setTitle(self.tr('group_file'))
        self.btn_open.setText(self.tr('btn_open'))
        if hasattr(self, 'btn_load_project'):
            self.btn_load_project.setText(self.tr('btn_load_project'))
        self.btn_fit.setText(self.tr('btn_fit'))
        self.btn_arrange.setText(self.tr('btn_arrange'))
        self.btn_arrange.setToolTip(self.tr('tip_arrange'))
        self.lbl_cols.setText(self.tr('lbl_cols'))
        self.spin_cols.setSpecialValueText(self.tr('cols_auto'))
        self.lbl_spacing.setText(self.tr('lbl_spacing'))
        if hasattr(self, 'lbl_arrange_mode'):
            self.lbl_arrange_mode.setText(self.tr('lbl_arrange_mode'))
        self.combo_arrange_mode.setItemText(0, self.tr('mode_matrix'))
        self.combo_arrange_mode.setItemText(1, self.tr('mode_row'))
        self.combo_arrange_mode.setItemText(2, self.tr('mode_col'))
        self.btn_save.setText(self.tr('btn_save'))
        self.btn_export.setText(self.tr('btn_export'))

        # 5. Dock Group 2: Smart Stitching Tools
        self.tools_group.setTitle(self.tr('group_tools'))
        if hasattr(self, 'lbl_ai_engines'):
            self.lbl_ai_engines.setText(self.tr('lbl_ai_engines'))
        if hasattr(self, 'lbl_stitch_mode'):
            self.lbl_stitch_mode.setText(self.tr('lbl_stitch_mode'))
        if hasattr(self, 'cmb_stitch_mode'):
            self.cmb_stitch_mode.setItemText(0, self.tr('stitch_mode_auto'))
            self.cmb_stitch_mode.setItemText(1, self.tr('stitch_mode_2d'))
            self.cmb_stitch_mode.setItemText(2, self.tr('stitch_mode_3d'))
        if hasattr(self, 'lbl_align_engine'):
            self.lbl_align_engine.setText(self.tr('lbl_align_engine'))
        if hasattr(self, 'cmb_align_engine'):
            self.cmb_align_engine.setItemText(0, self.tr('engine_superglue'))
            self.cmb_align_engine.setItemText(1, self.tr('engine_dense'))
        if hasattr(self, 'lbl_warp_engine'):
            self.lbl_warp_engine.setText(self.tr('lbl_warp_engine'))
        if hasattr(self, 'cmb_warp_engine'):
            self.cmb_warp_engine.setItemText(0, self.tr('engine_mls'))
            self.cmb_warp_engine.setItemText(1, self.tr('engine_deformable'))
            self.cmb_warp_engine.setItemText(2, self.tr('engine_parallax'))
        if hasattr(self, 'chk_global_graph'):
            self.chk_global_graph.setText(self.tr('chk_global_graph'))

        self.btn_smart_snap.setText(self.tr('btn_smart_snap'))
        self.btn_smart_snap.setToolTip(self.tr('tip_smart_snap'))
        self.btn_auto_align.setText(self.tr('btn_auto_align'))
        self.btn_auto_align.setToolTip(self.tr('tip_auto_align'))
        self.btn_elastic.setText(self.tr('btn_elastic'))
        self.btn_elastic.setToolTip(self.tr('tip_elastic'))
        self.btn_stretch.setText(self.tr('btn_stretch'))
        self.btn_stretch.setToolTip(self.tr('tip_stretch'))
        self.btn_reset_stretch.setText(self.tr('btn_reset_stretch'))
        self.btn_reset_stretch.setToolTip(self.tr('tip_reset_stretch'))
        if hasattr(self, 'btn_merge'):
            self.btn_merge.setText(self.tr('btn_merge'))
            self.btn_merge.setToolTip(self.tr('tip_merge'))
        self.btn_overlap.setText(self.tr('btn_overlap'))
        self.btn_overlap.setToolTip(self.tr('tip_overlap'))

        # 6. Dock Group 3: Opacity & Inspection
        self.inspect_group.setTitle(self.tr('group_inspect'))
        self.btn_toggle_opacity.setToolTip(self.tr('tip_toggle_opacity'))
        self._update_opacity_display(self.slider_opacity.value())

        # 7. Dock Group 4: X-Ray Correction
        self.xray_group.setTitle(self.tr('group_xray'))
        self.lbl_blend_mode.setText(self.tr('lbl_blend_mode'))
        self.btn_mode_seamline.setText(self.tr('btn_mode_seamline'))
        self.btn_mode_seamline.setToolTip(self.tr('tip_blend_mode'))
        self.btn_mode_feather.setText(self.tr('btn_mode_feather'))
        self.btn_mode_feather.setToolTip(self.tr('tip_blend_mode'))
        if hasattr(self, 'btn_mode_multiband'):
            self.btn_mode_multiband.setText(self.tr('btn_mode_multiband'))
            self.btn_mode_multiband.setToolTip(self.tr('tip_blend_mode'))
        self.chk_feather.setText(self.tr('chk_feather'))
        self.chk_feather.setToolTip(self.tr('tip_feather'))
        self._update_power_display()
        self._update_edge_bright_display()
        self.slider_edge_bright.setToolTip(self.tr('tip_edge_bright'))

        # 8. Dock Group 5: X-Ray Physical Calibration (Contract Sec 2 & 6)
        if hasattr(self, 'calib_group'):
            self.calib_group.setTitle(self.tr('group_calib'))
            self.lbl_sdd.setText(self.tr('lbl_sdd'))
            self.lbl_sdd.setToolTip(self.tr('tip_sdd'))
            self.spin_sdd.setSpecialValueText(self.tr('val_unspecified'))
            self.lbl_sod.setText(self.tr('lbl_sod'))
            self.lbl_sod.setToolTip(self.tr('tip_sod'))
            self.spin_sod.setSpecialValueText(self.tr('val_unspecified'))
            self.lbl_pitch.setText(self.tr('lbl_pitch'))
            self.lbl_pitch.setToolTip(self.tr('tip_pitch'))
            self.spin_pitch.setSpecialValueText(self.tr('val_unspecified'))
            self.btn_clear_calib.setText(self.tr('btn_clear_calib'))
            if hasattr(self, 'btn_cone_rectify'):
                self.btn_cone_rectify.setText(self.tr('btn_cone_rectify'))
                self.btn_cone_rectify.setToolTip(self.tr('tip_cone_rectify'))
            self._update_calibration_derived()

        # 9. Dock Group 6: Edit & Canvas Selection
        self.edit_group.setTitle(self.tr('group_edit'))
        self.btn_undo.setText(self.tr('btn_undo'))
        self.btn_redo.setText(self.tr('btn_redo'))
        self.btn_select_all.setText(self.tr('btn_select_all'))
        self.btn_del.setText(self.tr('btn_del'))

        # 9. Dock Group 6: Keyboard Shortcuts Card
        self.info_group.setTitle(self.tr('group_shortcuts'))
        for lbl_widget, tr_key in self.shortcut_labels:
            lbl_widget.setText(self.tr(tr_key))

    def _create_top_toolbar(self):
        self.toolbar = QToolBar("Main Toolbar", self)
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        self.toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self.toolbar.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.toolbar.setIconSize(QSize(20, 20))
        self.toolbar.setFixedHeight(48)
        self.toolbar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)

        # 1. Specialized Language Switcher Widget (32px fixed height)
        self.lang_switcher = LanguageSwitcherWidget(current_lang=self.current_lang, parent=self)
        self.lang_switcher.languageChanged.connect(self.set_language)
        self.toolbar.addWidget(self.lang_switcher)

        self.toolbar.addSeparator()

        # 2. File & Project Operations Group
        self.tb_open = QToolButton(self)
        self.tb_open.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_DialogOpenButton))
        self.tb_open.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.tb_open.clicked.connect(self._choose_directory)

        self.tb_load = QToolButton(self)
        self.tb_load.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_FileDialogDetailedView))
        self.tb_load.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.tb_load.clicked.connect(lambda: self.load_project())

        self.tb_save = QToolButton(self)
        self.tb_save.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_DialogSaveButton))
        self.tb_save.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.tb_save.clicked.connect(lambda: self.save_project())

        self.tb_export = QToolButton(self)
        self.tb_export.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_DialogOkButton))
        self.tb_export.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.tb_export.clicked.connect(self._export_layout)

        for btn in (self.tb_open, self.tb_load, self.tb_save, self.tb_export):
            btn.setFixedHeight(32)
            btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self.toolbar.addWidget(btn)

        self.toolbar.addSeparator()

        # 3. View & Layout Arrangement Group
        self.tb_fit = QToolButton(self)
        self.tb_fit.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_BrowserReload))
        self.tb_fit.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.tb_fit.clicked.connect(self.canvas.fit_view)

        self.tb_arrange = QToolButton(self)
        self.tb_arrange.setIcon(create_arrange_spacing_icon(24))
        self.tb_arrange.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.tb_arrange.clicked.connect(self._on_arrange_clicked)

        for btn in (self.tb_fit, self.tb_arrange):
            btn.setFixedHeight(32)
            btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self.toolbar.addWidget(btn)

        self.toolbar.addSeparator()

        # 4. Smart Alignment & Stitching Group
        self.tb_smart_snap = QToolButton(self)
        self.tb_smart_snap.setIcon(create_smart_snap_icon(24))
        self.tb_smart_snap.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.tb_smart_snap.clicked.connect(self.canvas.initiate_smart_snap)

        self.tb_auto_align = QToolButton(self)
        self.tb_auto_align.setIcon(create_homography_icon(24))
        self.tb_auto_align.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.tb_auto_align.clicked.connect(self.canvas.initiate_auto_align)

        self.tb_elastic = QToolButton(self)
        self.tb_elastic.setIcon(create_elastic_warp_icon(24))
        self.tb_elastic.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.tb_elastic.clicked.connect(self.canvas.initiate_elastic_warp)

        self.tb_stretch = QToolButton(self)
        self.tb_stretch.setIcon(create_stretch_warp_icon(24))
        self.tb_stretch.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.tb_stretch.clicked.connect(self.canvas.toggle_stretch_mode)

        self.tb_merge = QToolButton(self)
        self.tb_merge.setIcon(create_merge_icon(24))
        self.tb_merge.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.tb_merge.clicked.connect(self.canvas.initiate_batch_merge)

        self.tb_overlap = QToolButton(self)
        self.tb_overlap.setIcon(create_overlap_adjust_icon(24))
        self.tb_overlap.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.tb_overlap.clicked.connect(self.canvas.initiate_overlap_mode)

        for btn in (self.tb_smart_snap, self.tb_auto_align, self.tb_elastic, self.tb_stretch, self.tb_merge, self.tb_overlap):
            btn.setFixedHeight(32)
            btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self.toolbar.addWidget(btn)

        self.toolbar.addSeparator()

        # 5. History & Overlay Mode Group
        self.tb_undo = QToolButton(self)
        self.tb_undo.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_ArrowBack))
        self.tb_undo.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.tb_undo.clicked.connect(self.canvas.undo)

        self.tb_redo = QToolButton(self)
        self.tb_redo.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_ArrowForward))
        self.tb_redo.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.tb_redo.clicked.connect(self.canvas.redo)

        self.tb_ghost = QToolButton(self)
        self.tb_ghost.setIcon(create_dual_blocks_icon(translucent=True, opacity=0.55, size=24))
        self.tb_ghost.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.tb_ghost.clicked.connect(self.canvas.toggle_ghost_view)

        for btn in (self.tb_undo, self.tb_redo, self.tb_ghost):
            btn.setFixedHeight(32)
            btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self.toolbar.addWidget(btn)

    def _create_control_dock(self):
        self.dock = QDockWidget(self.tr('dock_title'), self)
        self.dock.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setLayoutDirection(Qt.LayoutDirection.LeftToRight)

        container = QWidget()
        container.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        container.setStyleSheet("background-color: #FFFFFF;")
        layout = QVBoxLayout(container)
        layout.setSpacing(14)
        layout.setContentsMargins(12, 12, 12, 12)

        # 1. Project & File Management
        self.file_group = QGroupBox(self.tr('group_file'))
        self.file_group.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        f_layout = QGridLayout(self.file_group)
        f_layout.setSpacing(8)

        self.btn_open = QPushButton(self.tr('btn_open'))
        self.btn_open.clicked.connect(self._choose_directory)
        self.btn_load_project = QPushButton(self.tr('btn_load_project'))
        self.btn_load_project.clicked.connect(lambda: self.load_project())
        self.btn_fit = QPushButton(self.tr('btn_fit'))
        self.btn_fit.clicked.connect(self.canvas.fit_view)
        self.btn_save = QPushButton(self.tr('btn_save'))
        self.btn_save.clicked.connect(lambda: self.save_project())
        self.btn_export = QPushButton(self.tr('btn_export'))
        self.btn_export.setObjectName("btnExport")
        self.btn_export.clicked.connect(self._export_layout)

        f_layout.addWidget(self.btn_open, 0, 0)
        f_layout.addWidget(self.btn_load_project, 0, 1)
        f_layout.addWidget(self.btn_fit, 1, 0)
        f_layout.addWidget(self.btn_save, 1, 1)
        f_layout.addWidget(self.btn_export, 2, 0, 1, 2)

        # Divider Line between Project actions and Mosaic Matrix
        div_matrix = QFrame()
        div_matrix.setFrameShape(QFrame.Shape.HLine)
        div_matrix.setStyleSheet("background-color: #E2E8F0; max-height: 1px; margin: 4px 0px;")
        f_layout.addWidget(div_matrix, 3, 0, 1, 2)

        # Arrange Button (Row 4, Spans 2 columns)
        self.btn_arrange = QPushButton(self.tr('btn_arrange'))
        self.btn_arrange.setObjectName("btnArrange")
        self.btn_arrange.setIcon(create_arrange_spacing_icon(48))
        self.btn_arrange.setIconSize(QSize(22, 22))
        self.btn_arrange.setToolTip(self.tr('tip_arrange'))
        self.btn_arrange.clicked.connect(self._on_arrange_clicked)
        f_layout.addWidget(self.btn_arrange, 4, 0, 1, 2)

        # Mode Selector (Row 5)
        self.lbl_arrange_mode = QLabel(self.tr('lbl_arrange_mode'))
        self.lbl_arrange_mode.setStyleSheet("font-weight: 600; color: #1E293B;")
        self.combo_arrange_mode = QComboBox()
        self.combo_arrange_mode.addItem(self.tr('mode_matrix'))
        self.combo_arrange_mode.addItem(self.tr('mode_row'))
        self.combo_arrange_mode.addItem(self.tr('mode_col'))
        self.combo_arrange_mode.setStyleSheet("""
            QComboBox {
                padding: 5px 8px;
                border: 1.5px solid #CBD5E1;
                border-radius: 6px;
                background-color: #F8FAFC;
                font-weight: 600;
                color: #1E293B;
            }
            QComboBox:focus {
                border-color: #2563EB;
            }
        """)
        f_layout.addWidget(self.lbl_arrange_mode, 5, 0)
        f_layout.addWidget(self.combo_arrange_mode, 5, 1)

        # Matrix Columns Setting (Row 6)
        self.lbl_cols = QLabel(self.tr('lbl_cols'))
        self.lbl_cols.setStyleSheet("font-weight: 600; color: #1E293B;")
        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(0, 50)
        self.spin_cols.setValue(0)
        self.spin_cols.setSpecialValueText(self.tr('cols_auto'))
        self.spin_cols.setStyleSheet("""
            QSpinBox {
                padding: 4px 8px;
                border: 1.5px solid #CBD5E1;
                border-radius: 6px;
                background-color: #F8FAFC;
                font-weight: 600;
                color: #1E293B;
            }
            QSpinBox:focus {
                border-color: #2563EB;
                background-color: #FFFFFF;
            }
        """)
        f_layout.addWidget(self.lbl_cols, 6, 0)
        f_layout.addWidget(self.spin_cols, 6, 1)

        # Spacing Controls (Row 7)
        self.lbl_spacing = QLabel(self.tr('lbl_spacing'))
        self.lbl_spacing.setStyleSheet("font-weight: 600; color: #1E293B;")
        self.spin_spacing = QSpinBox()
        self.spin_spacing.setRange(-500, 2000)
        self.spin_spacing.setValue(40)
        self.spin_spacing.setSuffix(" px")
        self.spin_spacing.setSingleStep(10)
        self.spin_spacing.setStyleSheet("""
            QSpinBox {
                padding: 4px 8px;
                border: 1.5px solid #CBD5E1;
                border-radius: 6px;
                background-color: #F8FAFC;
                font-weight: 600;
                color: #1E293B;
            }
            QSpinBox:focus {
                border-color: #2563EB;
                background-color: #FFFFFF;
            }
        """)
        f_layout.addWidget(self.lbl_spacing, 7, 0)
        f_layout.addWidget(self.spin_spacing, 7, 1)
        layout.addWidget(self.file_group)

        # 2. Smart AI & Stitching Tools
        self.tools_group = QGroupBox(self.tr('group_tools'))
        self.tools_group.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        t_layout = QVBoxLayout(self.tools_group)
        t_layout.setSpacing(8)

        # AI & Backup Engines Selector (کنترل موتورهای هوش مصنوعی و مدل‌های یدکی)
        self.ai_engine_frame = QFrame()
        self.ai_engine_frame.setStyleSheet("""
            QFrame {
                background-color: #F8FAFC;
                border: 1px solid #CBD5E1;
                border-radius: 8px;
            }
            QComboBox {
                padding: 3px 6px;
                border: 1px solid #94A3B8;
                border-radius: 4px;
                background-color: #FFFFFF;
                font-size: 10px;
                color: #0F172A;
            }
        """)
        ai_layout = QVBoxLayout(self.ai_engine_frame)
        ai_layout.setContentsMargins(8, 8, 8, 8)
        ai_layout.setSpacing(6)

        self.lbl_ai_engines = QLabel(self.tr('lbl_ai_engines'))
        self.lbl_ai_engines.setStyleSheet("font-weight: 700; color: #1E293B; font-size: 11px;")
        ai_layout.addWidget(self.lbl_ai_engines)

        stitch_mode_row = QHBoxLayout()
        self.lbl_stitch_mode = QLabel(self.tr('lbl_stitch_mode'))
        self.lbl_stitch_mode.setStyleSheet("font-weight: 600; color: #475569; font-size: 10px;")
        self.cmb_stitch_mode = QComboBox()
        self.cmb_stitch_mode.addItem(self.tr('stitch_mode_auto'), 'auto')
        self.cmb_stitch_mode.addItem(self.tr('stitch_mode_2d'), '2d')
        self.cmb_stitch_mode.addItem(self.tr('stitch_mode_3d'), '3d')
        self.cmb_stitch_mode.currentIndexChanged.connect(
            lambda idx: self.canvas.set_stitch_mode(self.cmb_stitch_mode.currentData())
        )
        stitch_mode_row.addWidget(self.lbl_stitch_mode)
        stitch_mode_row.addWidget(self.cmb_stitch_mode, 1)
        ai_layout.addLayout(stitch_mode_row)

        align_row = QHBoxLayout()
        self.lbl_align_engine = QLabel(self.tr('lbl_align_engine'))
        self.lbl_align_engine.setStyleSheet("font-weight: 600; color: #475569; font-size: 10px;")
        self.cmb_align_engine = QComboBox()
        self.cmb_align_engine.addItem(self.tr('engine_superglue'), 'superglue')
        self.cmb_align_engine.addItem(self.tr('engine_dense'), 'dense')
        self.cmb_align_engine.currentIndexChanged.connect(
            lambda idx: self.canvas.set_align_engine(self.cmb_align_engine.currentData())
        )
        align_row.addWidget(self.lbl_align_engine)
        align_row.addWidget(self.cmb_align_engine, 1)
        ai_layout.addLayout(align_row)

        warp_row = QHBoxLayout()
        self.lbl_warp_engine = QLabel(self.tr('lbl_warp_engine'))
        self.lbl_warp_engine.setStyleSheet("font-weight: 600; color: #475569; font-size: 10px;")
        self.cmb_warp_engine = QComboBox()
        self.cmb_warp_engine.addItem(self.tr('engine_mls'), 'elastic_mls')
        self.cmb_warp_engine.addItem(self.tr('engine_deformable'), 'deformable_dir')
        self.cmb_warp_engine.addItem(self.tr('engine_parallax'), 'parallax_udis')
        self.cmb_warp_engine.currentIndexChanged.connect(
            lambda idx: self.canvas.set_warp_engine(self.cmb_warp_engine.currentData())
        )
        warp_row.addWidget(self.lbl_warp_engine)
        warp_row.addWidget(self.cmb_warp_engine, 1)
        ai_layout.addLayout(warp_row)

        self.chk_global_graph = QCheckBox(self.tr('chk_global_graph'))
        self.chk_global_graph.setChecked(True)
        self.chk_global_graph.setStyleSheet("font-weight: 600; color: #1E293B; font-size: 10px;")
        self.chk_global_graph.stateChanged.connect(
            lambda state: self.canvas.set_use_global_graph(state == 2 or state == Qt.CheckState.Checked)
        )
        ai_layout.addWidget(self.chk_global_graph)
        t_layout.addWidget(self.ai_engine_frame)

        self.btn_smart_snap = QPushButton(self.tr('btn_smart_snap'))
        self.btn_smart_snap.setObjectName("btnSmartSnap")
        self.btn_smart_snap.setIcon(create_smart_snap_icon(48))
        self.btn_smart_snap.setIconSize(QSize(20, 20))
        self.btn_smart_snap.setToolTip(self.tr('tip_smart_snap'))
        self.btn_smart_snap.clicked.connect(self.canvas.initiate_smart_snap)

        self.btn_auto_align = QPushButton(self.tr('btn_auto_align'))
        self.btn_auto_align.setObjectName("btnAutoAlign")
        self.btn_auto_align.setIcon(create_homography_icon(48))
        self.btn_auto_align.setIconSize(QSize(20, 20))
        self.btn_auto_align.setToolTip(self.tr('tip_auto_align'))
        self.btn_auto_align.clicked.connect(self.canvas.initiate_auto_align)

        self.btn_elastic = QPushButton(self.tr('btn_elastic'))
        self.btn_elastic.setObjectName("btnElastic")
        self.btn_elastic.setIcon(create_elastic_warp_icon(48))
        self.btn_elastic.setIconSize(QSize(20, 20))
        self.btn_elastic.setToolTip(self.tr('tip_elastic'))
        self.btn_elastic.clicked.connect(self.canvas.initiate_elastic_warp)

        stretch_layout = QHBoxLayout()
        stretch_layout.setSpacing(6)
        self.btn_stretch = QPushButton(self.tr('btn_stretch'))
        self.btn_stretch.setObjectName("btnStretch")
        self.btn_stretch.setIcon(create_stretch_warp_icon(48))
        self.btn_stretch.setIconSize(QSize(20, 20))
        self.btn_stretch.setToolTip(self.tr('tip_stretch'))
        self.btn_stretch.clicked.connect(self.canvas.toggle_stretch_mode)
        stretch_layout.addWidget(self.btn_stretch, 3)

        self.btn_reset_stretch = QPushButton(self.tr('btn_reset_stretch'))
        self.btn_reset_stretch.setObjectName("btnResetStretch")
        self.btn_reset_stretch.setToolTip(self.tr('tip_reset_stretch'))
        self.btn_reset_stretch.clicked.connect(self.canvas.reset_piece_stretch)
        stretch_layout.addWidget(self.btn_reset_stretch, 1)

        self.btn_merge = QPushButton(self.tr('btn_merge'))
        self.btn_merge.setObjectName("btnMerge")
        self.btn_merge.setIcon(create_merge_icon(48))
        self.btn_merge.setIconSize(QSize(20, 20))
        self.btn_merge.setToolTip(self.tr('tip_merge'))
        self.btn_merge.clicked.connect(self.canvas.initiate_batch_merge)

        self.btn_overlap = QPushButton(self.tr('btn_overlap'))
        self.btn_overlap.setObjectName("btnOverlap")
        self.btn_overlap.setIcon(create_overlap_adjust_icon(48))
        self.btn_overlap.setIconSize(QSize(20, 20))
        self.btn_overlap.setToolTip(self.tr('tip_overlap'))
        self.btn_overlap.clicked.connect(self.canvas.initiate_overlap_mode)

        t_layout.addWidget(self.btn_smart_snap)
        t_layout.addWidget(self.btn_auto_align)
        self.btn_visual_mosaic = QPushButton('دوخت و تراز تصاویر | Mosaic alignment')
        self.btn_visual_mosaic.setToolTip('Align selected colour/X-ray frames using full homographies and a shared graph; unmatched images remain separate.')
        self.btn_visual_mosaic.clicked.connect(self.canvas.initiate_visual_mosaic)
        t_layout.addWidget(self.btn_visual_mosaic)
        self.btn_manual_registration = QPushButton('نقاط متناظر و کنترل | Manual alignment')
        self.btn_manual_registration.setToolTip('Select two images, choose same-surface fit and independent check points; inspect checkerboard and difference before applying.')
        self.btn_manual_registration.clicked.connect(self.canvas.initiate_manual_registration)
        t_layout.addWidget(self.btn_manual_registration)
        self.combo_visual_model = QComboBox()
        self.combo_visual_model.addItem('پرسپکتیو کامل | Projective', 'projective')
        self.combo_visual_model.addItem('اسکن تخت | Affine scan', 'affine')
        self.combo_visual_model.setToolTip('Projective supports perspective cameras; affine constrains extrapolation for planar scans. Both are visual and do not recover unknown depth.')
        self.combo_visual_model.currentIndexChanged.connect(lambda: setattr(self.canvas, 'visual_mosaic_model', self.combo_visual_model.currentData()))
        t_layout.addWidget(self.combo_visual_model)
        t_layout.addWidget(self.btn_elastic)
        t_layout.addLayout(stretch_layout)
        t_layout.addWidget(self.btn_merge)
        t_layout.addWidget(self.btn_overlap)
        layout.addWidget(self.tools_group)

        # 3. Layer Transparency & Inspection (See-Through / Ghost View)
        self.inspect_group = QGroupBox(self.tr('group_inspect'))
        self.inspect_group.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        ins_layout = QVBoxLayout(self.inspect_group)
        ins_layout.setSpacing(10)

        opacity_hdr = QHBoxLayout()
        self.lbl_opacity_icon = QLabel()
        self.lbl_opacity_icon.setPixmap(create_opacity_level_icon(100, 24).pixmap(20, 20))
        self.lbl_opacity = QLabel(self.tr('lbl_opacity').format(val=100))
        self.lbl_opacity.setStyleSheet("font-weight: 700; color: #1E293B; font-size: 12px;")
        opacity_hdr.addWidget(self.lbl_opacity_icon)
        opacity_hdr.addWidget(self.lbl_opacity)
        opacity_hdr.addStretch()
        ins_layout.addLayout(opacity_hdr)

        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(20, 100)
        self.slider_opacity.setValue(100)
        self.slider_opacity.valueChanged.connect(self._on_opacity_slider_changed)

        self.btn_toggle_opacity = QPushButton(self.tr('btn_toggle_opacity_off'))
        self.btn_toggle_opacity.setObjectName("btnLayerOpacity")
        self.btn_toggle_opacity.setIcon(create_dual_blocks_icon(translucent=False, opacity=1.0, size=48))
        self.btn_toggle_opacity.setIconSize(QSize(22, 22))
        self.btn_toggle_opacity.setToolTip(self.tr('tip_toggle_opacity'))
        self.btn_toggle_opacity.clicked.connect(self.canvas.toggle_ghost_view)

        ins_layout.addWidget(self.slider_opacity)
        ins_layout.addWidget(self.btn_toggle_opacity)
        layout.addWidget(self.inspect_group)

        # 4. X-Ray Oblique Perspective Correction (Distance Feathering & Seamline)
        self.xray_group = QGroupBox(self.tr('group_xray'))
        self.xray_group.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        x_layout = QVBoxLayout(self.xray_group)
        x_layout.setSpacing(10)

        self.lbl_blend_mode = QLabel(self.tr('lbl_blend_mode'))
        self.lbl_blend_mode.setStyleSheet("font-weight: 700; color: #1E293B; font-size: 11px;")
        x_layout.addWidget(self.lbl_blend_mode)

        blend_pill = QFrame()
        blend_pill.setStyleSheet("""
            QFrame {
                background-color: #F1F5F9;
                border: 1px solid #CBD5E1;
                border-radius: 8px;
            }
        """)
        blend_layout = QHBoxLayout(blend_pill)
        blend_layout.setContentsMargins(2, 2, 2, 2)
        blend_layout.setSpacing(2)

        self.btn_mode_seamline = QPushButton(self.tr('btn_mode_seamline'))
        self.btn_mode_feather = QPushButton(self.tr('btn_mode_feather'))
        self.btn_mode_multiband = QPushButton(self.tr('btn_mode_multiband'))
        for b in (self.btn_mode_seamline, self.btn_mode_feather, self.btn_mode_multiband):
            b.setCheckable(True)
            b.setFixedHeight(28)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.btn_mode_seamline.clicked.connect(lambda: self._set_blend_mode('seamline'))
        self.btn_mode_feather.clicked.connect(lambda: self._set_blend_mode('feather'))
        self.btn_mode_multiband.clicked.connect(lambda: self._set_blend_mode('multiband'))

        blend_layout.addWidget(self.btn_mode_seamline)
        blend_layout.addWidget(self.btn_mode_feather)
        blend_layout.addWidget(self.btn_mode_multiband)
        x_layout.addWidget(blend_pill)
        self._set_blend_mode('seamline')

        self.chk_feather = QCheckBox(self.tr('chk_feather'))
        self.chk_feather.setChecked(True)
        self.chk_feather.setToolTip(self.tr('tip_feather'))
        self.chk_feather.stateChanged.connect(self._on_feather_settings_changed)
        x_layout.addWidget(self.chk_feather)

        self.lbl_power = QLabel(self.tr('lbl_power').format(val=2.0))
        self.lbl_power.setStyleSheet("font-weight: 600; color: #1E293B;")
        self.slider_power = QSlider(Qt.Orientation.Horizontal)
        self.slider_power.setRange(10, 40)
        self.slider_power.setValue(20)
        self.slider_power.valueChanged.connect(self._on_feather_settings_changed)

        x_layout.addWidget(self.lbl_power)
        x_layout.addWidget(self.slider_power)

        self.lbl_edge_bright = QLabel(self.tr('lbl_edge_bright').format(sign="", val=0))
        self.lbl_edge_bright.setStyleSheet("font-weight: 600; color: #1E293B;")
        self.slider_edge_bright = QSlider(Qt.Orientation.Horizontal)
        self.slider_edge_bright.setRange(-50, 50)
        self.slider_edge_bright.setValue(0)
        self.slider_edge_bright.setToolTip(self.tr('tip_edge_bright'))
        self.slider_edge_bright.valueChanged.connect(self._on_edge_brightness_changed)

        x_layout.addWidget(self.lbl_edge_bright)
        x_layout.addWidget(self.slider_edge_bright)
        layout.addWidget(self.xray_group)

        # 5. X-Ray Geometric & Physical Calibration Group (Contract Sec 2 & 6)
        self.calib_group = QGroupBox(self.tr('group_calib'))
        self.calib_group.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        c_layout = QVBoxLayout(self.calib_group)
        c_layout.setSpacing(8)

        # Status Badge (Uncalibrated vs Calibrated)
        self.lbl_calib_status = QLabel(self.tr('lbl_calib_status_uncal'))
        self.lbl_calib_status.setStyleSheet("""
            QLabel {
                background-color: #F1F5F9;
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                padding: 5px 8px;
                color: #475569;
                font-size: 11px;
                font-weight: 700;
            }
        """)
        c_layout.addWidget(self.lbl_calib_status)

        calib_grid = QGridLayout()
        calib_grid.setSpacing(6)

        # SDD (Source-to-Detector Distance)
        self.lbl_sdd = QLabel(self.tr('lbl_sdd'))
        self.lbl_sdd.setStyleSheet("font-weight: 600; color: #1E293B; font-size: 11px;")
        self.spin_sdd = CalibrationField(self, placeholder=self.tr('val_unspecified'))
        self.spin_sdd.setToolTip(self.tr('tip_sdd'))
        self.spin_sdd.valueChanged.connect(self._on_calibration_changed)

        # SOD (Source-to-Object Distance)
        self.lbl_sod = QLabel(self.tr('lbl_sod'))
        self.lbl_sod.setStyleSheet("font-weight: 600; color: #1E293B; font-size: 11px;")
        self.spin_sod = CalibrationField(self, placeholder=self.tr('val_unspecified'))
        self.spin_sod.setToolTip(self.tr('tip_sod'))
        self.spin_sod.valueChanged.connect(self._on_calibration_changed)

        # Pixel Pitch (um)
        self.lbl_pitch = QLabel(self.tr('lbl_pitch'))
        self.lbl_pitch.setStyleSheet("font-weight: 600; color: #1E293B; font-size: 11px;")
        self.spin_pitch = CalibrationField(self, placeholder=self.tr('val_unspecified'))
        self.spin_pitch.setToolTip(self.tr('tip_pitch'))
        self.spin_pitch.valueChanged.connect(self._on_calibration_changed)

        calib_grid.addWidget(self.lbl_sdd, 0, 0)
        calib_grid.addWidget(self.spin_sdd, 0, 1)
        calib_grid.addWidget(self.lbl_sod, 1, 0)
        calib_grid.addWidget(self.spin_sod, 1, 1)
        calib_grid.addWidget(self.lbl_pitch, 2, 0)
        calib_grid.addWidget(self.spin_pitch, 2, 1)
        c_layout.addLayout(calib_grid)

        # Derived Values Card (ODD and M0)
        derived_card = QFrame()
        derived_card.setStyleSheet("""
            QFrame {
                background-color: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                padding: 6px;
            }
        """)
        d_layout = QVBoxLayout(derived_card)
        d_layout.setContentsMargins(6, 6, 6, 6)
        d_layout.setSpacing(4)

        self.lbl_odd_display = QLabel(self.tr('lbl_odd_display').format(val=self.tr('val_unspecified')))
        self.lbl_odd_display.setStyleSheet("color: #334155; font-size: 11px; font-weight: 600;")
        self.lbl_mag_display = QLabel(self.tr('lbl_mag_display').format(val=self.tr('val_unspecified')))
        self.lbl_mag_display.setStyleSheet("color: #334155; font-size: 11px; font-weight: 600;")

        d_layout.addWidget(self.lbl_odd_display)
        d_layout.addWidget(self.lbl_mag_display)
        c_layout.addWidget(derived_card)

        # Reset button
        self.btn_clear_calib = QPushButton(self.tr('btn_clear_calib'))
        self.btn_clear_calib.setStyleSheet("""
            QPushButton {
                background-color: #F8FAFC;
                border: 1px solid #CBD5E1;
                color: #64748B;
                font-size: 11px;
                font-weight: 600;
                padding: 5px;
                border-radius: 6px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #F1F5F9;
                color: #DC2626;
                border-color: #EF4444;
            }
        """)
        self.btn_clear_calib.clicked.connect(self._reset_calibration)
        c_layout.addWidget(self.btn_clear_calib)

        # دکمه تصحیح اعوجاج شعاعی پرتو مخروطی
        self.btn_cone_rectify = QPushButton(self.tr('btn_cone_rectify'))
        self.btn_cone_rectify.setStyleSheet("""
            QPushButton {
                background-color: #EEF2FF;
                border: 1px solid #6366F1;
                color: #4338CA;
                font-size: 11px;
                font-weight: 700;
                padding: 6px;
                border-radius: 6px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #4F46E5;
                color: #FFFFFF;
                border-color: #4338CA;
            }
        """)
        self.btn_cone_rectify.setToolTip(self.tr('tip_cone_rectify'))
        self.btn_cone_rectify.clicked.connect(self.apply_cone_beam_orthorectification)
        c_layout.addWidget(self.btn_cone_rectify)
        self.btn_metric_editor = QPushButton("Edit calibration / stage JSON | ویرایش هندسه")
        def open_metric_editor():
            from xray_mosaic.qt_view import edit_manifest
            edit_manifest(self)
        self.btn_metric_editor.clicked.connect(open_metric_editor)
        c_layout.addWidget(self.btn_metric_editor)

        layout.addWidget(self.calib_group)

        # 6. Edit & Selection Controls
        self.edit_group = QGroupBox(self.tr('group_edit'))
        self.edit_group.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        e_layout = QGridLayout(self.edit_group)
        e_layout.setSpacing(8)

        self.btn_undo = QPushButton(self.tr('btn_undo'))
        self.btn_undo.clicked.connect(self.canvas.undo)
        self.btn_redo = QPushButton(self.tr('btn_redo'))
        self.btn_redo.clicked.connect(self.canvas.redo)
        self.btn_select_all = QPushButton(self.tr('btn_select_all'))
        self.btn_select_all.clicked.connect(self.canvas.select_all)
        self.btn_del = QPushButton(self.tr('btn_del'))
        self.btn_del.setObjectName("btnDelete")
        self.btn_del.clicked.connect(self.canvas.delete_selected)

        e_layout.addWidget(self.btn_undo, 0, 0)
        e_layout.addWidget(self.btn_redo, 0, 1)
        e_layout.addWidget(self.btn_select_all, 1, 0)
        e_layout.addWidget(self.btn_del, 1, 1)
        layout.addWidget(self.edit_group)

        # 6. Hotkey & Gesture Reference Card
        self.info_group = QGroupBox(self.tr('group_shortcuts'))
        self.info_group.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        i_layout = QVBoxLayout(self.info_group)
        i_layout.setSpacing(6)

        shortcuts = [
            ("N", 'sc_smart_snap'),
            ("G", 'sc_auto_align'),
            ("E", 'sc_elastic'),
            ("W", 'sc_stretch'),
            ("O", 'sc_transparency'),
            ("M", 'sc_overlap'),
            ("L", 'sc_arrange'),
            ("Arrows", 'sc_nudge'),
            ("Shift+Arrows", 'sc_fast_nudge'),
            ("Enter", 'sc_finalize'),
            ("Z / Y", 'sc_undo_redo'),
            ("Scroll", 'sc_zoom'),
            ("Right Drag", 'sc_pan'),
        ]
        self.shortcut_labels = []
        for k, tr_key in shortcuts:
            row = QHBoxLayout()
            lbl_k = QLabel(f"[{k}]")
            lbl_k.setStyleSheet("color: #2563EB; font-weight: 800; font-family: monospace; font-size: 12px;")
            lbl_d = QLabel(self.tr(tr_key))
            lbl_d.setStyleSheet("color: #475569; font-size: 11px;")
            row.addWidget(lbl_k)
            row.addWidget(lbl_d)
            row.addStretch()
            i_layout.addLayout(row)
            self.shortcut_labels.append((lbl_d, tr_key))

        layout.addWidget(self.info_group)
        layout.addStretch()

        # Mouse-Only & Focus Protection Filter for the entire Sidebar / Dock
        self.sidebar_filter = SidebarMouseOnlyFilter(scroll, self.canvas, self)

        # 1. Non-text widgets (buttons, combos, sliders) - NoFocus & redirect keys to canvas
        dock_buttons_and_sliders = [
            self.btn_open, self.btn_load_project, self.btn_fit, self.btn_save, self.btn_export,
            self.btn_arrange, self.combo_arrange_mode,
            self.cmb_stitch_mode, self.cmb_align_engine, self.cmb_warp_engine, self.chk_global_graph,
            self.btn_smart_snap, self.btn_auto_align, self.btn_elastic, self.btn_stretch,
            self.btn_reset_stretch, self.btn_merge, self.btn_overlap, self.slider_opacity, self.btn_toggle_opacity,
            self.btn_mode_seamline, self.btn_mode_feather, self.btn_mode_multiband, self.chk_feather, self.slider_power,
            self.slider_edge_bright, self.btn_clear_calib, self.btn_cone_rectify,
            self.btn_undo, self.btn_redo, self.btn_select_all, self.btn_del
        ]
        for ctrl in dock_buttons_and_sliders:
            ctrl.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            ctrl.installEventFilter(self.sidebar_filter)

        # 2. Editable numeric input spinboxes - Allow direct typing & select-all on click
        def _setup_numeric_spinbox(sp):
            sp.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            le = sp.lineEdit()
            le.setReadOnly(False)
            le.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            orig_mp = le.mousePressEvent
            def on_mp(e):
                orig_mp(e)
                QTimer.singleShot(0, le.selectAll)
            le.mousePressEvent = on_mp

        for sp in (self.spin_cols, self.spin_spacing):
            _setup_numeric_spinbox(sp)
            sp.installEventFilter(self.sidebar_filter)
            sp.lineEdit().installEventFilter(self.sidebar_filter)

        container.mousePressEvent = lambda e: self.canvas.setFocus()
        scroll.mousePressEvent = lambda e: self.canvas.setFocus()

        scroll.setWidget(container)
        self.dock.setWidget(scroll)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock)

    def is_calibrated(self):
        if not hasattr(self, 'spin_sdd') or not hasattr(self, 'spin_sod'):
            return False
        sdd = self.spin_sdd.value()
        sod = self.spin_sod.value()
        return (sdd > 0 and sod > 0 and sdd > sod)

    def apply_cone_beam_orthorectification(self):
        """Run the shared calibrated offline pipeline from original files + batch metadata."""
        if self.canvas.is_busy:
            return
        from xray_mosaic.pipeline import Batch
        from xray_mosaic.qt_view import ResultDialog
        import uuid
        path, _ = QFileDialog.getOpenFileName(
            self, "Calibration + per-shot stage metadata | هندسه و موقعیت شات‌ها", "", "JSON (*.json)")
        if not path:
            return
        try:
            batch = Batch(path)
        except (ValueError, OSError, TypeError) as error:
            QMessageBox.warning(self, "X-ray geometry", str(error))
            return
        parent = QFileDialog.getExistingDirectory(self, "Choose parent folder for a new immutable export")
        if not parent:
            return
        destination = os.path.join(parent, "xray_mosaic_" + uuid.uuid4().hex[:12])
        def done(report):
            self.status_bar.showMessage(report['status'] + " | " + destination)
            viewer = ResultDialog(self, destination, report)
            viewer.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            viewer.show()
        import threading
        self._metric_cancel = threading.Event()
        def failed(error):
            if not self._metric_cancel.is_set():
                QMessageBox.warning(self, "X-ray geometry", error)
        self.canvas.run_in_background(
            lambda: batch.run(destination, cancelled=self._metric_cancel.is_set), done,
            failed,
            status_msg="Computing calibrated reference-plane mosaic from originals…")

    def _reset_calibration(self):
        if hasattr(self, 'spin_sdd'):
            self.spin_sdd.setValue(0.0)
            self.spin_sod.setValue(0.0)
            self.spin_pitch.setValue(0.0)
            self._update_calibration_derived()

        # بازگرداندن تصاویر به فرم خام اولیه قبل از تصحیح پرتو مخروطی
        for p in self.canvas.pieces:
            if hasattr(p, 'raw_unrectified_ref') and p.raw_unrectified_ref is not None:
                p.image = p.raw_unrectified_ref.copy()
                p.original_image_ref = p.raw_unrectified_ref.copy()
                p.sync_corners_to_rect()
        self.canvas.pixmap_cache.clear()
        self.canvas.update()

    def _on_calibration_changed(self):
        self._update_calibration_derived()

    def _update_calibration_derived(self):
        if not hasattr(self, 'spin_sdd') or not hasattr(self, 'spin_sod'):
            return
        sdd = self.spin_sdd.value()
        sod = self.spin_sod.value()
        pitch_um = self.spin_pitch.value() if hasattr(self, 'spin_pitch') else 0.0

        if sdd > 0 and sod > 0 and sdd > sod:
            odd = sdd - sod
            m0 = sdd / sod
            self.lbl_odd_display.setText(self.tr('lbl_odd_display').format(val=f"{odd:.1f} mm"))
            self.lbl_mag_display.setText(self.tr('lbl_mag_display').format(val=f"{m0:.3f}x"))

            if pitch_um > 0:
                pitch_mm = (pitch_um / 1000.0) / m0
                self.lbl_calib_status.setText(self.tr('lbl_calib_status_cal').format(mag=m0, pitch_mm=pitch_mm))
                self.lbl_calib_status.setStyleSheet("""
                    QLabel {
                        background-color: #ECFDF5;
                        border: 1.5px solid #10B981;
                        border-radius: 6px;
                        padding: 5px 8px;
                        color: #065F46;
                        font-size: 11px;
                        font-weight: 700;
                    }
                """)
            else:
                self.lbl_calib_status.setText(f"M₀ = {m0:.3f}x (Pitch: {self.tr('val_unspecified')})")
                self.lbl_calib_status.setStyleSheet("""
                    QLabel {
                        background-color: #EFF6FF;
                        border: 1.5px solid #3B82F6;
                        border-radius: 6px;
                        padding: 5px 8px;
                        color: #1E40AF;
                        font-size: 11px;
                        font-weight: 700;
                    }
                """)
        else:
            self.lbl_odd_display.setText(self.tr('lbl_odd_display').format(val=self.tr('val_unspecified')))
            self.lbl_mag_display.setText(self.tr('lbl_mag_display').format(val=self.tr('val_unspecified')))
            self.lbl_calib_status.setText(self.tr('lbl_calib_status_uncal'))
            self.lbl_calib_status.setStyleSheet("""
                QLabel {
                    background-color: #F1F5F9;
                    border: 1px solid #CBD5E1;
                    border-radius: 6px;
                    padding: 5px 8px;
                    color: #475569;
                    font-size: 11px;
                    font-weight: 700;
                }
            """)

    def _set_blend_mode(self, mode):
        self.canvas.set_blend_mode(mode)
        active_style = """
            QPushButton {
                background-color: #2563EB;
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                font-family: 'Segoe UI', Tahoma, sans-serif;
                font-weight: 700;
                font-size: 11px;
                padding: 4px 8px;
                text-align: center;
            }
        """
        inactive_style = """
            QPushButton {
                background-color: transparent;
                color: #64748B;
                border: none;
                border-radius: 6px;
                font-family: 'Segoe UI', Tahoma, sans-serif;
                font-weight: 600;
                font-size: 11px;
                padding: 4px 8px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #E2E8F0;
                color: #0F172A;
            }
        """
        if mode == 'seamline':
            self.btn_mode_seamline.setChecked(True)
            self.btn_mode_feather.setChecked(False)
            if hasattr(self, 'btn_mode_multiband'):
                self.btn_mode_multiband.setChecked(False)
                self.btn_mode_multiband.setStyleSheet(inactive_style)
            self.btn_mode_seamline.setStyleSheet(active_style)
            self.btn_mode_feather.setStyleSheet(inactive_style)
        elif mode == 'feather':
            self.btn_mode_seamline.setChecked(False)
            self.btn_mode_feather.setChecked(True)
            if hasattr(self, 'btn_mode_multiband'):
                self.btn_mode_multiband.setChecked(False)
                self.btn_mode_multiband.setStyleSheet(inactive_style)
            self.btn_mode_seamline.setStyleSheet(inactive_style)
            self.btn_mode_feather.setStyleSheet(active_style)
        else: # multiband
            self.btn_mode_seamline.setChecked(False)
            self.btn_mode_feather.setChecked(False)
            if hasattr(self, 'btn_mode_multiband'):
                self.btn_mode_multiband.setChecked(True)
                self.btn_mode_multiband.setStyleSheet(active_style)
            self.btn_mode_seamline.setStyleSheet(inactive_style)
            self.btn_mode_feather.setStyleSheet(inactive_style)

    def _sync_opacity_ui(self, opacity):
        val = int(round(opacity * 100))
        self.slider_opacity.blockSignals(True)
        self.slider_opacity.setValue(val)
        self.slider_opacity.blockSignals(False)
        self._update_opacity_display(val)

    def _update_opacity_display(self, val):
        self.lbl_opacity.setText(self.tr('lbl_opacity').format(val=val))
        self.lbl_opacity_icon.setPixmap(create_opacity_level_icon(val, 24).pixmap(20, 20))
        is_ghost = (val < 99)
        self.btn_toggle_opacity.setIcon(create_dual_blocks_icon(translucent=is_ghost, opacity=val / 100.0, size=48))
        if is_ghost:
            self.btn_toggle_opacity.setText(self.tr('btn_toggle_opacity_on').format(val=val))
            self.btn_toggle_opacity.setStyleSheet("""
                QPushButton#btnLayerOpacity {
                    background-color: #EFF6FF;
                    color: #1D4ED8;
                    border: 1.5px solid #2563EB;
                    font-weight: 700;
                    text-align: left;
                    padding-left: 12px;
                }
                QPushButton#btnLayerOpacity:hover {
                    background-color: #DBEAFE;
                }
            """)
        else:
            self.btn_toggle_opacity.setText(self.tr('btn_toggle_opacity_off'))
            self.btn_toggle_opacity.setStyleSheet("""
                QPushButton#btnLayerOpacity {
                    background-color: #F8FAFC;
                    color: #334155;
                    border: 1.5px solid #CBD5E1;
                    font-weight: 600;
                    text-align: left;
                    padding-left: 12px;
                }
                QPushButton#btnLayerOpacity:hover {
                    background-color: #F1F5F9;
                    border-color: #2563EB;
                    color: #2563EB;
                }
            """)

    def _on_opacity_slider_changed(self):
        val = self.slider_opacity.value()
        self._update_opacity_display(val)
        self.canvas.set_selection_opacity(val / 100.0)

    def _update_edge_bright_display(self):
        val = self.slider_edge_bright.value()
        sign = "+" if val > 0 else ""
        self.lbl_edge_bright.setText(self.tr('lbl_edge_bright').format(sign=sign, val=val))

    def _on_edge_brightness_changed(self):
        self._update_edge_bright_display()
        self.canvas.set_edge_brightness(self.slider_edge_bright.value())

    def _update_power_display(self):
        power = self.slider_power.value() / 10.0
        self.lbl_power.setText(self.tr('lbl_power').format(val=power))

    def _on_feather_settings_changed(self):
        self._update_power_display()
        power = self.slider_power.value() / 10.0
        self.canvas.set_feather_settings(power, self.chk_feather.isChecked())

    def _on_arrange_clicked(self):
        spacing = self.spin_spacing.value() if hasattr(self, 'spin_spacing') else 40
        cols_val = self.spin_cols.value() if hasattr(self, 'spin_cols') else 0
        mode_idx = self.combo_arrange_mode.currentIndex() if hasattr(self, 'combo_arrange_mode') else 0
        mode_str = 'matrix' if mode_idx == 0 else ('row' if mode_idx == 1 else 'col')
        self.canvas.arrange_pieces(spacing=spacing, cols=cols_val, mode=mode_str)

    def _choose_directory(self):
        d = QFileDialog.getExistingDirectory(self, self.tr('dialog_select_dir'), self.directory)
        if d:
            self.directory = d
            self.load_images_from_directory(d)

    def load_images_from_directory(self, folder):
        if not os.path.exists(folder):
            self.status_bar.showMessage(self.tr('status_dir_not_found').format(folder=folder))
            return
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'))]
        if not files:
            self.status_bar.showMessage(self.tr('status_no_images').format(folder=folder))
            return
        files.sort(key=natural_sort_key)

        folder_base = os.path.basename(os.path.normpath(folder))
        if not folder_base or folder_base in ('.', ''):
            folder_base = "xray_project"
        self.current_project_name = folder_base
        self.current_project_path = None
        self.setWindowTitle(f"{self.tr('window_title')} - [{folder_base}]")

        self.canvas.pieces.clear()
        self.canvas.next_piece_id = 0
        raw_items = []
        for f in files:
            p = os.path.join(folder, f)
            img = cv.imread(p, cv.IMREAD_UNCHANGED)
            if img is not None:
                # Display-only conversion. Metric batch reloads originals at native depth.
                if img.dtype == np.uint16:
                    img_8u = (img / 256.0).astype(np.uint8)
                elif img.dtype != np.uint8:
                    img_8u = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
                else:
                    img_8u = img

                if img_8u.ndim == 2:
                    img_bgra = cv.cvtColor(img_8u, cv.COLOR_GRAY2BGRA)
                elif img_8u.shape[2] == 3:
                    img_bgra = cv.cvtColor(img_8u, cv.COLOR_BGR2BGRA)
                elif img_8u.shape[2] == 4:
                    img_bgra = img_8u
                else:
                    continue
                raw_items.append((f, img_bgra))

        if not raw_items:
            self.status_bar.showMessage(self.tr('status_no_images').format(folder=folder))
            return

        spacing = float(self.spin_spacing.value() if hasattr(self, 'spin_spacing') else 40)
        cols_val = self.spin_cols.value() if hasattr(self, 'spin_cols') else 0
        mode_idx = self.combo_arrange_mode.currentIndex() if hasattr(self, 'combo_arrange_mode') else 0
        mode_str = 'matrix' if mode_idx == 0 else ('row' if mode_idx == 1 else 'col')

        for f, img in raw_items:
            pid = self.canvas.next_piece_id
            self.canvas.next_piece_id += 1
            piece = PuzzlePiece(pid, img, 0, 0, original_ref=img, filename=f)
            piece.source_file_path = os.path.abspath(os.path.join(folder, f))
            self.canvas.pieces.append(piece)

        self.canvas.arrange_pieces(spacing=spacing, cols=cols_val, mode=mode_str)
        self.canvas.pixmap_cache.clear()
        self.canvas.history = [copy.deepcopy(self.canvas.pieces)]
        self.canvas.redo_history.clear()
        self.canvas.fit_view()
        self.status_bar.showMessage(self.tr('status_loaded').format(count=len(self.canvas.pieces)))

    def _export_layout(self):
        if not self.canvas.pieces or self.canvas.is_busy:
            return
        self.canvas.start_region_capture()


    def save_project(self, project_path=None, force_ask=False, *args):
        """
        F-08: ذخیره کامل پروژه پازل در قالب فایل متاداده JSON به همراه فایل‌های تصویری و موقعیت دقیق قطعات.
        اگر پروژه قبلاً ذخیره یا باز شده باشد، در همان فایل ذخیره می‌شود و هر بار نام جدید تولید نمی‌کند.
        """
        if not self.canvas.pieces:
            self.status_bar.showMessage(self.tr('status_no_images').format(folder='Canvas'))
            return

        if self.canvas.is_busy:
            busy_txt = "سیستم در حال انجام عملیات دیگری است، لطفاً کمی صبر کنید..." if self.current_lang == 'fa' else "System is busy, please wait..."
            self.status_bar.showMessage(busy_txt)
            return

        self.canvas.trigger_hotkey_hud('S', self.canvas.tr('hud_save'))

        # If project_path not given: check if we already have an established project path
        if not isinstance(project_path, str) or not project_path.strip():
            if self.current_project_path and not force_ask:
                # Re-save to the established project path without prompting or creating a new timestamp folder
                project_path = self.current_project_path
            else:
                os.makedirs(PROJECT_SAVE_DIRECTORY, exist_ok=True)
                # Derive default project name from input directory or piece filenames
                base_name = getattr(self, 'current_project_name', None)
                if not base_name or base_name in ("xray_project", "."):
                    fns = [os.path.splitext(os.path.basename(p.filename))[0] for p in self.canvas.pieces if getattr(p, 'filename', None)]
                    if fns:
                        common = os.path.commonprefix(fns).rstrip('_- ')
                        base_name = common if len(common) >= 3 else fns[0]
                if not base_name:
                    base_name = "xray_project"

                default_json = os.path.join(PROJECT_SAVE_DIRECTORY, base_name, f"{base_name}.json")
                os.makedirs(os.path.dirname(default_json), exist_ok=True)
                chosen_file, _ = QFileDialog.getSaveFileName(
                    self,
                    self.tr('dialog_save_project'),
                    default_json,
                    "Project Manifest (*.json);;All Files (*)"
                )
                if not chosen_file:
                    return
                project_path = chosen_file

        self.current_project_path = project_path
        file_stem = os.path.splitext(os.path.basename(project_path))[0]
        self.current_project_name = file_stem if file_stem not in ("project_data", "manifest") else os.path.basename(os.path.dirname(project_path))

        proj_dir = os.path.dirname(project_path)
        images_dir = os.path.join(proj_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        spacing_val = float(self.spin_spacing.value() if hasattr(self, 'spin_spacing') else 40)
        cols_val = int(self.spin_cols.value() if hasattr(self, 'spin_cols') else 0)
        mode_val = int(self.combo_arrange_mode.currentIndex() if hasattr(self, 'combo_arrange_mode') else 0)

        # Physical calibration persistence per contract
        calib_dict = {
            "SDD_mm": float(self.spin_sdd.value()) if (hasattr(self, 'spin_sdd') and self.spin_sdd.value() > 0) else None,
            "SOD_mm": float(self.spin_sod.value()) if (hasattr(self, 'spin_sod') and self.spin_sod.value() > 0) else None,
            "pixel_pitch_um": float(self.spin_pitch.value()) if (hasattr(self, 'spin_pitch') and self.spin_pitch.value() > 0) else None,
            "is_calibrated": bool(self.is_calibrated()) if hasattr(self, 'is_calibrated') else False
        }

        manifest = {
            "version": "2.0",
            "saved_at": datetime.datetime.now().isoformat(),
            "calibration": calib_dict,
            "view_state": {
                "scale": float(self.canvas.scale),
                "view_offset_x": float(self.canvas.view_offset_x),
                "view_offset_y": float(self.canvas.view_offset_y),
                "enable_feathering": bool(self.canvas.enable_feathering),
                "feather_power": float(self.canvas.feather_power)
            },
            "arrange_settings": {
                "spacing": spacing_val,
                "cols": cols_val,
                "mode": mode_val
            },
            "pieces": []
        }

        used_filenames = set()
        save_items = []
        for piece in self.canvas.pieces:
            base_fn = os.path.basename(piece.filename) if getattr(piece, 'filename', None) else ""
            if not base_fn:
                base_fn = f"piece_{piece.id}.png"
            root, ext = os.path.splitext(base_fn)
            if not ext:
                ext = ".png"
            img_filename = f"{root}{ext}"
            counter = 1
            while img_filename in used_filenames:
                img_filename = f"{root}_{counter}{ext}"
                counter += 1
            used_filenames.add(img_filename)

            img_path = os.path.join(images_dir, img_filename)
            save_items.append((img_path, piece.image.copy()))

            p_entry = {
                "id": int(piece.id),
                "group_id": int(piece.group_id),
                "x": float(piece.x),
                "y": float(piece.y),
                "w": int(piece.w),
                "h": int(piece.h),
                "filename": str(piece.filename),
                "image_filename": img_filename,
                "corners_cur": piece.corners_cur.tolist() if getattr(piece, 'corners_cur', None) is not None else None,
                "corners_orig": piece.corners_orig.tolist() if getattr(piece, 'corners_orig', None) is not None else None,
                "boundary_world": piece.boundary_world.tolist() if getattr(piece, 'boundary_world', None) is not None else None,
                "corners_world": piece.corners_world.tolist() if getattr(piece, 'corners_world', None) is not None else None,
                "boundary_src": piece.boundary_src.tolist() if getattr(piece, 'boundary_src', None) is not None else None,
                "corner_indices": list(piece.corner_indices) if getattr(piece, 'corner_indices', None) is not None else None
            }
            manifest["pieces"].append(p_entry)

        def task_save_project():
            # Heavy cv.imwrite disk writes executed asynchronously in background
            for img_p, img_mat in save_items:
                cv.imwrite(img_p, img_mat)
            with open(project_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=4, ensure_ascii=False)
            return True

        def on_save_project_done(res):
            self.current_project_path = project_path
            self.current_project_name = os.path.basename(os.path.dirname(project_path))
            self.setWindowTitle(f"{self.tr('window_title')} - [{self.current_project_name}]")
            self.status_bar.showMessage(self.tr('status_project_saved').format(path=project_path))
            QMessageBox.information(self, self.tr('dialog_project_saved_title'), self.tr('dialog_project_saved_msg').format(path=project_path))

        status_msg = self.tr('status_project_saving')
        self.canvas.run_in_background(task_save_project, on_save_project_done, status_msg=status_msg)

    def load_project(self, project_path=None, *args):
        """
        F-08: بارگذاری کامل پروژه از فایل JSON و بازیابی دقیق موقعیت، چیدمان و شکل قطعات
        """
        if not isinstance(project_path, str) or not project_path.strip():
            chosen_file, _ = QFileDialog.getOpenFileName(
                self,
                self.tr('dialog_load_project'),
                PROJECT_SAVE_DIRECTORY if os.path.exists(PROJECT_SAVE_DIRECTORY) else ".",
                "Project Manifest (*.json);;All Files (*)"
            )
            if not chosen_file:
                return
            project_path = chosen_file

        self.current_project_path = project_path
        file_stem = os.path.splitext(os.path.basename(project_path))[0]
        self.current_project_name = file_stem if file_stem not in ("project_data", "manifest") else os.path.basename(os.path.dirname(project_path))
        self.setWindowTitle(f"{self.tr('window_title')} - [{self.current_project_name}]")

        if not os.path.exists(project_path):
            self.status_bar.showMessage(self.tr('status_project_not_found').format(path=project_path))
            return

        try:
            with open(project_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        except Exception as e:
            is_fa = getattr(self, 'current_lang', 'en') == 'fa'
            err_msg = f"خطا در خواندن فایل JSON:\n{str(e)}" if is_fa else f"Error reading JSON file:\n{str(e)}"
            QMessageBox.critical(self, self.tr('dialog_load_project'), err_msg)
            return

        if not isinstance(manifest, dict):
            is_fa = getattr(self, 'current_lang', 'en') == 'fa'
            err_msg = "ساختار فایل مانیفست نامعتبر است." if is_fa else "Invalid project manifest structure."
            QMessageBox.warning(self, self.tr('dialog_load_project'), err_msg)
            return

        # Check if the user selected an export artifact (transforms.json or report.json)
        base_name_lower = os.path.basename(project_path).lower()
        if (
            base_name_lower in ("transforms.json", "report.json")
            or ("coordinate_system" in manifest and "capture_bounding_box" in manifest)
            or ("schema_version" in manifest and "physics_model_audit" in manifest)
        ):
            is_fa = getattr(self, 'current_lang', 'en') == 'fa'
            msg = (
                "فایل انتخاب‌شده یک مانیفست خروجی موزاییک (transforms.json / report.json) است.\n\n"
                "بسته‌های خروجی شامل گزارش اعتبارسنجی و تصویر نهایی هستند، نه قطعات اولیه پروژه.\n"
                "برای باز کردن پروژه با قطعات قابل ویرایش، لطفاً فایل مانیفست پروژه (مانند project_data.json) را از پوشه 'puzzle_saves' انتخاب نمایید."
                if is_fa else
                "The selected file is an export artifact (transforms.json / report.json).\n\n"
                "Export packages contain audit reports and stitched mosaics, not individual editable project tiles.\n"
                "To open a saved project, please select a project file (such as project_data.json) from the 'puzzle_saves' folder."
            )
            QMessageBox.information(self, self.tr('dialog_load_project'), msg)
            return

        proj_dir = os.path.dirname(project_path)
        images_dir = os.path.join(proj_dir, "images")

        loaded_pieces = []
        max_id = 0

        for idx, p_info in enumerate(manifest.get("pieces", [])):
            if not isinstance(p_info, dict):
                continue

            # Safely resolve piece ID
            pid_val = p_info.get("id")
            if pid_val is None:
                pid_val = p_info.get("piece_id", idx)
            try:
                pid = int(pid_val)
            except (ValueError, TypeError):
                pid = idx

            # Safely resolve image filename
            img_fn = p_info.get("image_filename") or p_info.get("filename") or f"piece_{pid}.png"

            # Check candidate paths for the piece image
            candidates = [
                os.path.join(images_dir, img_fn),
                os.path.join(images_dir, f"piece_{pid}.png"),
                os.path.join(proj_dir, img_fn),
                os.path.join(proj_dir, f"piece_{pid}.png"),
            ]
            if os.path.isabs(img_fn):
                candidates.insert(0, img_fn)
            orig_p = p_info.get("original_path")
            if orig_p and os.path.exists(orig_p):
                candidates.append(orig_p)

            img_path = None
            for c in candidates:
                if c and os.path.exists(c):
                    img_path = c
                    break

            if not img_path:
                continue

            img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 2:
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                img = cv.cvtColor(img, cv.COLOR_BGR2BGRA)

            gid = int(p_info.get("group_id", pid))
            px = float(p_info.get("x", p_info.get("world_x", 0.0)))
            py = float(p_info.get("y", p_info.get("world_y", 0.0)))
            fn = p_info.get("filename", os.path.basename(img_path))

            piece = PuzzlePiece(pid, img, px, py, original_ref=img, filename=fn, group_id=gid)
            if "corners_cur" in p_info and p_info["corners_cur"] is not None:
                piece.corners_cur = np.array(p_info["corners_cur"], dtype=np.float32)
            if "corners_orig" in p_info and p_info["corners_orig"] is not None:
                piece.corners_orig = np.array(p_info["corners_orig"], dtype=np.float32)
            if "boundary_world" in p_info and p_info["boundary_world"] is not None:
                piece.boundary_world = np.array(p_info["boundary_world"], dtype=np.float32)
            if "corners_world" in p_info and p_info["corners_world"] is not None:
                piece.corners_world = np.array(p_info["corners_world"], dtype=np.float32)
            if "boundary_src" in p_info and p_info["boundary_src"] is not None:
                piece.boundary_src = np.array(p_info["boundary_src"], dtype=np.float32)
            if "corner_indices" in p_info and p_info["corner_indices"] is not None:
                piece.corner_indices = list(p_info["corner_indices"])

            loaded_pieces.append(piece)
            if pid > max_id:
                max_id = pid

        if not loaded_pieces:
            is_fa = getattr(self, 'current_lang', 'en') == 'fa'
            msg = (
                f"هیچ تصویر یا قطعه معتبری از مانیفست زیر بارگذاری نشد:\n{project_path}\n\n"
                f"لطفاً مطمئن شوید که تصاویر در پوشه images کنار فایل مانیفست قرار دارند."
                if is_fa else
                f"No valid puzzle piece images could be found or loaded from:\n{project_path}\n\n"
                f"Please ensure the 'images' folder exists alongside the manifest."
            )
            QMessageBox.warning(self, self.tr('dialog_load_project'), msg)
            return

        # Commit loaded state to canvas
        self.canvas.pieces = loaded_pieces
        self.canvas.selection.clear()
        self.canvas.next_piece_id = max_id + 1

        self.current_project_path = project_path
        file_stem = os.path.splitext(os.path.basename(project_path))[0]
        self.current_project_name = file_stem if file_stem not in ("project_data", "manifest") else os.path.basename(os.path.dirname(project_path))
        self.setWindowTitle(f"{self.tr('window_title')} - [{self.current_project_name}]")

        view_state = manifest.get("view_state", {})
        self.canvas.scale = float(view_state.get("scale", 1.0))
        self.canvas.view_offset_x = float(view_state.get("view_offset_x", 0.0))
        self.canvas.view_offset_y = float(view_state.get("view_offset_y", 0.0))
        if "enable_feathering" in view_state:
            self.canvas.enable_feathering = bool(view_state["enable_feathering"])
            if hasattr(self, 'chk_feather'):
                self.chk_feather.setChecked(self.canvas.enable_feathering)
        if "feather_power" in view_state:
            self.canvas.feather_power = float(view_state["feather_power"])
            if hasattr(self, 'slider_power'):
                self.slider_power.setValue(int(self.canvas.feather_power * 10))

        arrange_settings = manifest.get("arrange_settings", {})
        if "spacing" in arrange_settings and hasattr(self, 'spin_spacing'):
            self.spin_spacing.setValue(int(arrange_settings["spacing"]))
        if "cols" in arrange_settings and hasattr(self, 'spin_cols'):
            self.spin_cols.setValue(int(arrange_settings["cols"]))
        if "mode" in arrange_settings and hasattr(self, 'combo_arrange_mode'):
            self.combo_arrange_mode.setCurrentIndex(int(arrange_settings["mode"]))

        # Restore physical calibration per contract (keep null if uncalibrated)
        calib_data = manifest.get("calibration", {})
        if hasattr(self, 'spin_sdd'):
            self.spin_sdd.setValue(float(calib_data.get("SDD_mm") or 0.0))
        if hasattr(self, 'spin_sod'):
            self.spin_sod.setValue(float(calib_data.get("SOD_mm") or 0.0))
        if hasattr(self, 'spin_pitch'):
            self.spin_pitch.setValue(float(calib_data.get("pixel_pitch_um") or 0.0))
        if hasattr(self, '_update_calibration_derived'):
            self._update_calibration_derived()

        self.canvas.pixmap_cache.clear()
        self.canvas.history = [copy.deepcopy(self.canvas.pieces)]
        self.canvas.redo_history.clear()
        self.canvas.update()
        self.status_bar.showMessage(self.tr('status_project_loaded').format(count=len(self.canvas.pieces), path=project_path))

    def closeEvent(self, event):
        # Cancel offline metric work and keep the window alive until its worker releases files.
        if (hasattr(self, '_metric_cancel') or hasattr(self, '_visual_cancel')) and self.canvas.active_workers:
            if hasattr(self, '_metric_cancel'):
                self._metric_cancel.set()
            if hasattr(self, '_visual_cancel'):
                self._visual_cancel.set()
            event.ignore()
            QTimer.singleShot(100, self.close)
            return
        if hasattr(self, 'canvas') and hasattr(self.canvas, 'active_workers'):
            for w in list(self.canvas.active_workers):
                if w.isRunning():
                    w.wait(2500)
        super().closeEvent(event)


# ================== Application Entry Point ==================
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Picture Puzzle")
    app.setStyle("Fusion")
    # Preserve LeftToRight layout direction across whole app as specified
    app.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
    window = XRayPuzzleStudioWindow()
    window.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()

