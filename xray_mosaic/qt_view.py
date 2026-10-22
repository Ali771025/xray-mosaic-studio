"""Read-only result viewer attached to the existing desktop application."""
from pathlib import Path
import json
import math
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QComboBox, QGraphicsView,
                             QGraphicsScene, QPlainTextEdit, QPushButton, QFileDialog, QMessageBox)


class MetricView(QGraphicsView):
    def __init__(self, scene, label, info, step):
        super().__init__(scene)
        self.label, self.info, self.step = label, info, step
        self.start_point = None
        self.metric_map = None
        self.sigma_map = None
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            pos = self.mapToScene(event.pos())
            xy = np.array([pos.x(), pos.y()])*self.step*self.info['spacing_mm'] + self.info['origin_plane_mm']
            ix, iy = int(round(pos.x()*self.step)), int(round(pos.y()*self.step))
            if (self.metric_map is None or not (0 <= iy < self.metric_map.shape[0] and 0 <= ix < self.metric_map.shape[1])
                    or not self.metric_map[iy, ix]):
                self.start_point = None
                self.label.setText('UNDERCONSTRAINED: measurement requires metric-valid covered pixels on the known plane.')
                return
            sigma = float(self.sigma_map[iy, ix])
            if self.start_point is None:
                self.start_point = (xy, sigma)
            else:
                distance = np.linalg.norm(xy-self.start_point[0])
                bound = self.start_point[1]+sigma
                self.label.setText(f'{self.info["status"]} | plane {self.info["id"]} | '
                                   f'distance {distance:.5g} mm | endpoint 1-sigma bounds sum {bound:.4g} mm | known plane only')
                self.start_point = None
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = self.mapToScene(event.pos())
        xy = np.array([pos.x(), pos.y()])*self.step*self.info['spacing_mm'] + self.info['origin_plane_mm']
        self.setToolTip(f'Plane {self.info["id"]}: ({xy[0]:.4f}, {xy[1]:.4f}) mm; '
                        'coordinates apply only to this plane; Shift-click two points for distance')
        super().mouseMoveEvent(event)


class ResultDialog(QDialog):
    def __init__(self, parent, destination, report):
        super().__init__(parent)
        self.setWindowTitle('X-ray reference-plane mosaic | موزاییک صفحه مرجع')
        self.resize(1100, 800)
        layout = QVBoxLayout(self)
        warning = QLabel(report['status']+'\n'+report['limitation']+'\n'+'; '.join(report['issues']))
        warning.setWordWrap(True)
        layout.addWidget(warning)
        self.selector = QComboBox()
        for plane in report['planes']:
            for name in ('intensity', 'coverage', 'source', 'metric_valid', 'sigma_mm'):
                self.selector.addItem(f'{plane["id"]}: {name}', (plane, name))
        layout.addWidget(self.selector)
        self.label = QLabel('Wheel: zoom; drag: pan; coordinates and uncertainty are plane-specific. Export: '+str(destination))
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        self.scene = QGraphicsScene(self)
        self.view = MetricView(self.scene, self.label, report['planes'][0], 1)
        layout.addWidget(self.view)
        self.destination = Path(destination)
        self.selector.currentIndexChanged.connect(self.show_map)
        self.show_map()

    def show_map(self):
        info, name = self.selector.currentData()
        raw = np.load(self.destination/info['directory']/(name+'.npy'), mmap_mode='r')
        step = max(1, math.ceil(max(raw.shape)/1600))
        a = np.array(raw[::step, ::step], dtype=float)
        valid = np.isfinite(a)
        if name == 'source':
            valid &= a >= 0
        low, high = (np.min(a[valid]), np.max(a[valid])) if valid.any() else (0, 1)
        gray = np.zeros(a.shape, np.uint8)
        gray[valid] = np.clip((a[valid]-low)/max(high-low, 1e-12)*255, 0, 255).astype(np.uint8)
        if name in ('coverage', 'metric_valid'):
            gray[(a > 0) & valid] = 180
            gray[a > 1] = 255
        rgb = np.repeat(gray[..., None], 3, axis=2)
        coverage = np.load(self.destination/info['directory']/'coverage.npy', mmap_mode='r')[::step, ::step]
        rgb[coverage == 0] = [255, 0, 180]
        rgb = np.ascontiguousarray(rgb)
        h, w = gray.shape
        qimg = QImage(rgb.data, w, h, 3*w, QImage.Format.Format_RGB888).copy()
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(qimg))
        self.view.info, self.view.step = info, step
        self.view.metric_map = np.load(self.destination/info['directory']/'metric_valid.npy', mmap_mode='r')
        self.view.sigma_map = np.load(self.destination/info['directory']/'sigma_mm.npy', mmap_mode='r')
        self.view.start_point = None
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.label.setText(f'Plane {info["id"]} | gap {info["gap_fraction"]:.2%} | '
                           f'overlap {info["overlap_fraction"]:.2%} | {info["status"]} | export: {self.destination}')


def edit_manifest(parent):
    path, _ = QFileDialog.getOpenFileName(parent, 'Open measured calibration + frame JSON', '', 'JSON (*.json)')
    if not path:
        return
    dialog = QDialog(parent)
    dialog.setWindowTitle('Calibration / stage metadata editor — save a new copy')
    dialog.resize(900, 700)
    layout = QVBoxLayout(dialog)
    label = QLabel('Enter verified geometry in mm, provenance, validation and absolute poses. '
                   'Unknown geometry must stay unknown. Relative image paths resolve beside the JSON file.')
    label.setWordWrap(True)
    layout.addWidget(label)
    editor = QPlainTextEdit(Path(path).read_text(encoding='utf-8'))
    layout.addWidget(editor)
    button = QPushButton('Validate schema and save new JSON')
    layout.addWidget(button)
    def save():
        from .pipeline import Batch
        import tempfile
        text = editor.toPlainText()
        try:
            json.loads(text)
            # Schema validation has no image loading or hardware effects.
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', encoding='utf-8') as f:
                f.write(text)
                f.flush()
                Batch(f.name)
            target, _ = QFileDialog.getSaveFileName(dialog, 'Save beside input images or use absolute paths', '', 'JSON (*.json)')
            if target:
                with Path(target).open('x', encoding='utf-8') as f:
                    f.write(text)
                dialog.accept()
        except (ValueError, OSError, TypeError) as e:
            QMessageBox.warning(dialog, 'Calibration metadata', str(e))
    button.clicked.connect(save)
    dialog.exec()
