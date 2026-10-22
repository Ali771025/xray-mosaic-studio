"""Two-raster correspondence editor used by the existing desktop canvas."""
import json
from pathlib import Path
import cv2 as cv
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPen, QColor
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QGraphicsView, QGraphicsScene, QTableWidget,
    QTableWidgetItem, QFileDialog, QMessageBox, QSplitter, QPlainTextEdit)
from manual_registration import (session_document, load_session, fit_correspondences,
                                  render_pair, export_result)


def pixmap(image):
    a = image
    if a.dtype == np.uint16:
        a = (a/257).astype(np.uint8)
    if a.ndim == 2:
        a = cv.cvtColor(a, cv.COLOR_GRAY2RGBA)
    elif a.shape[2] == 3:
        a = cv.cvtColor(a, cv.COLOR_BGR2RGBA)
    else:
        a = cv.cvtColor(a, cv.COLOR_BGRA2RGBA)
    a = np.ascontiguousarray(a)
    return QPixmap.fromImage(QImage(a.data, a.shape[1], a.shape[0], a.strides[0],
                                   QImage.Format.Format_RGBA8888).copy())


class PointView(QGraphicsView):
    picked = pyqtSignal(float, float)

    def __init__(self, image):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.scene().addPixmap(pixmap(image))
        self.setSceneRect(0, 0, image.shape[1], image.shape[0])
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.markers = []
        self.press = None

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else .8
        if .03 < self.transform().m11()*factor < 32:
            self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event):
        self.press = event.position() if event.button() == Qt.MouseButton.LeftButton else None
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.press is not None and (event.position()-self.press).manhattanLength() < 4:
            p = self.mapToScene(event.position().toPoint())
            if self.sceneRect().contains(p):
                self.picked.emit(p.x(), p.y())
        self.press = None
        super().mouseReleaseEvent(event)

    def mark(self, rows, key, pending=None):
        for item in self.markers:
            self.scene().removeItem(item)
        self.markers = []
        for i, row in enumerate(rows):
            x, y = row[key]
            pen = QPen(QColor('#28dd88' if row['role'] == 'check' else '#ffae35'))
            pen.setCosmetic(True)
            circle = self.scene().addEllipse(x-4, y-4, 8, 8, pen)
            text = self.scene().addText(str(i+1))
            text.setDefaultTextColor(pen.color())
            text.setPos(x+4, y+4)
            self.markers.extend((circle, text))
        if pending is not None:
            x, y = pending
            self.markers.append(self.scene().addEllipse(x-5, y-5, 10, 10, QPen(QColor('cyan'))))

    def fit_image(self):
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class CorrespondenceDialog(QDialog):
    def __init__(self, images, labels, model='projective', parent=None):
        super().__init__(parent)
        self.setWindowTitle('نقاط متناظر و کنترل مستقل | Manual alignment')
        self.resize(1250, 900)
        self.images, self.labels = images, labels
        self.points, self.pending = [], None
        self.report, self.result = None, None
        root = QVBoxLayout(self)
        root.addWidget(QLabel('ابتدا نقطهٔ تصویر چپ، سپس همان نقطه در راست. نقاط را روی یک سطح مشترک و در سراسر هم‌پوشانی انتخاب کنید.\n'
                              'Fit: برازش؛ Check: کنترل مستقل (حداقل ۳ نقطه). چرخ ماوس: بزرگ‌نمایی؛ کشیدن: جابه‌جایی.'))
        bar = QHBoxLayout()
        self.model = QComboBox()
        self.model.addItem('پرسپکتیو | Projective', 'projective')
        self.model.addItem('اسکن تخت | Affine', 'affine')
        self.model.setCurrentIndex(1 if model == 'affine' else 0)
        self.role = QComboBox()
        self.role.addItem('نقطهٔ برازش | Fit', 'fit')
        self.role.addItem('نقطهٔ کنترل | Check', 'check')
        self.tolerance = QDoubleSpinBox()
        self.tolerance.setRange(.05, 20.)
        self.tolerance.setDecimals(2)
        self.tolerance.setValue(1.)
        self.tolerance.setSuffix(' px')
        for widget in (self.model, self.role, QLabel('حد پذیرش نقاط:'), self.tolerance):
            bar.addWidget(widget)
        root.addLayout(bar)
        split = QSplitter()
        self.views = [PointView(im) for im in images]
        for i, view in enumerate(self.views):
            panel = QVBoxLayout()
            from PyQt6.QtWidgets import QWidget
            box = QWidget()
            box.setLayout(panel)
            panel.addWidget(QLabel(('مرجع | Reference: ' if i == 0 else 'متحرک | Moving: ')+str(labels[i])))
            panel.addWidget(view)
            split.addWidget(box)
            view.picked.connect(lambda x, y, side=i: self.pick(side, x, y))
        root.addWidget(split, 3)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['Role', 'Reference x,y', 'Moving x,y', 'Forward px', 'Reverse px'])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(170)
        root.addWidget(self.table)
        actions = QHBoxLayout()
        for text, slot in [('حذف انتخاب | Delete', self.delete_point), ('لغو نقطه | Cancel point', self.cancel_point),
                           ('نمای کامل | Fit view', self.fit_views), ('ذخیره نقاط | Save', self.save),
                           ('بارگذاری نقاط | Load', self.load), ('محاسبه و کنترل | Evaluate', self.evaluate)]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            actions.addWidget(b)
        root.addLayout(actions)
        self.message = QPlainTextEdit()
        self.message.setReadOnly(True)
        self.message.setMaximumHeight(100)
        root.addWidget(self.message)
        self.preview = PointView(np.zeros((1, 1, 3), np.uint8))
        self.preview.setMinimumHeight(150)
        root.addWidget(self.preview, 1)
        controls = QHBoxLayout()
        self.preview_mode = QComboBox()
        self.preview_mode.addItems(['دوخت | Seam', 'شطرنجی | Checkerboard', 'اختلاف | Difference'])
        self.preview_mode.currentIndexChanged.connect(self.update_preview)
        controls.addWidget(self.preview_mode)
        self.export_button = QPushButton('خروجی و گزارش | Export')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export)
        controls.addWidget(self.export_button)
        self.apply_button = QPushButton('اعمال دوخت | Apply')
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.accept)
        controls.addWidget(self.apply_button)
        close = QPushButton('بستن | Close')
        close.clicked.connect(self.reject)
        controls.addWidget(close)
        root.addLayout(controls)
        self.model.currentIndexChanged.connect(self.invalidate)
        self.tolerance.valueChanged.connect(self.invalidate)
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, self.fit_views)

    def fit_views(self):
        for view in self.views:
            view.fit_image()

    def invalidate(self, *_):
        self.report, self.result = None, None
        self.apply_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.preview.scene().clear()
        self.preview.markers = []
        self.message.setPlainText('برای ارزیابی، حداقل ۴ جفت Fit برای پرسپکتیو (۳ برای Affine) و ۳ جفت Check انتخاب کنید.\n'
                                  'پذیرش نقاط کنترل، تضمین انطباق کل تصویر یا اجسام چندعمقی نیست.')
        self.refresh()

    def pick(self, side, x, y):
        h, w = self.images[side].shape[:2]
        if not (0 <= x <= w-1 and 0 <= y <= h-1):
            return
        if side == 0:
            self.pending = ([x, y], self.role.currentData())
        elif self.pending:
            self.points.append({'reference': self.pending[0], 'moving': [x, y], 'role': self.pending[1]})
            self.pending = None
        self.invalidate()

    def refresh(self):
        self.table.setRowCount(len(self.points))
        for i, p in enumerate(self.points):
            residual = self.report['points'][i] if self.report else {}
            values = [p['role'], ', '.join(f'{v:.2f}' for v in p['reference']),
                      ', '.join(f'{v:.2f}' for v in p['moving']),
                      str(round(residual['forward_px'], 3)) if residual else '',
                      str(round(residual['reverse_px'], 3)) if residual else '']
            for j, value in enumerate(values):
                self.table.setItem(i, j, QTableWidgetItem(value))
        for i, view in enumerate(self.views):
            view.mark(self.points, ('reference', 'moving')[i], self.pending[0] if i == 0 and self.pending else None)

    def delete_point(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.points):
            del self.points[row]
            self.invalidate()

    def cancel_point(self):
        self.pending = None
        self.refresh()

    def save(self):
        name, _ = QFileDialog.getSaveFileName(self, 'Save correspondence session', '', 'JSON (*.json)')
        if name:
            doc = session_document(self.images, self.labels, self.points, self.model.currentData(), self.tolerance.value())
            doc['evaluation'] = self.report
            try:
                Path(name).write_text(json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            except Exception as exc:
                QMessageBox.warning(self, 'Save failed', str(exc))

    def load(self):
        name, _ = QFileDialog.getOpenFileName(self, 'Load correspondence session', '', 'JSON (*.json)')
        if name:
            try:
                doc = load_session(name, self.images)
                self.points, self.pending = doc['points'], None
                self.model.setCurrentIndex(1 if doc['model'] == 'affine' else 0)
                self.tolerance.setValue(doc['tolerance_px'])
                self.invalidate()  # always recompute; never trust stored acceptance
            except Exception as exc:
                QMessageBox.warning(self, 'Load failed', str(exc))

    def export(self):
        if self.result is None:
            return
        name, _ = QFileDialog.getSaveFileName(self, 'New output directory name (must not exist)', '', 'All files (*)')
        if name:
            try:
                export_result(name, self.images, self.labels, self.points, self.report, self.result)
                self.message.appendPlainText('Saved: '+name)
            except Exception as exc:
                QMessageBox.warning(self, 'Export failed', str(exc))

    def evaluate(self):
        self.invalidate()
        if self.pending:
            self.message.setPlainText('نقطهٔ نیمه‌کاره را کامل یا لغو کنید | Complete or cancel the pending point')
            return
        try:
            report = fit_correspondences(self.images, self.points, self.model.currentData(), self.tolerance.value())
            self.report = report
            self.refresh()
            self.message.setPlainText(f"{report['status']} | Fit max: {report['fit']['max_px']:.3f} px; "
                                      f"Check max: {report['check']['max_px']:.3f} px\n"
                                      'خطا در هر دو جهت سنجیده شده است؛ اعتبار فقط برای نقاط انتخاب‌شده است. VISUAL_ONLY')
            if report['accepted']:
                self.result = render_pair(self.images, report)
                self.result['qa']['manual_session'] = session_document(self.images, self.labels, self.points, report['model'], report['tolerance_px'])
                self.apply_button.setEnabled(True)
                self.export_button.setEnabled(True)
            self.update_preview()
        except Exception as exc:
            self.message.setPlainText(str(exc))

    def update_preview(self):
        if not self.report:
            return
        # Downsample only the diagnostic display; fitting and output remain native.
        from visual_mosaic import corners, transform
        H = np.array(self.report['H_moving_to_reference'])
        pts = np.concatenate((corners(self.images[0].shape), transform(corners(self.images[1].shape), H)))
        lo, hi = np.floor(pts.min(axis=0)), np.ceil(pts.max(axis=0))+1
        size = hi-lo
        scale = min(1., 1200/max(size))
        S = np.array([[scale, 0, -lo[0]*scale], [0, scale, -lo[1]*scale], [0, 0, 1.]])
        wh = tuple(np.maximum(1, np.ceil(size*scale).astype(int)))
        views, masks = [], []
        for im, matrix in zip(self.images, (S, S@H)):
            a = im if im.dtype == np.uint8 else (im/257).astype(np.uint8)
            if a.ndim == 2:
                a = cv.cvtColor(a, cv.COLOR_GRAY2BGR)
            alpha = a[..., 3] if a.shape[2] == 4 else np.full(a.shape[:2], 255, np.uint8)
            views.append(cv.warpPerspective(a[..., :3], matrix, wh))
            masks.append(cv.warpPerspective(alpha, matrix, wh, flags=cv.INTER_NEAREST) > 0)
        mode = self.preview_mode.currentIndex()
        if mode == 0 and self.result:
            out = self.result['image']
        else:
            a, b = views
            both = masks[0] & masks[1]
            out = a.copy()
            out[masks[1] & ~masks[0]] = b[masks[1] & ~masks[0]]
            if mode == 2:
                out[both] = cv.absdiff(a, b)[both]
            else:
                yy, xx = np.indices(both.shape)
                choose = both & ((xx//32+yy//32) % 2 == 1)
                out[choose] = b[choose]
        self.preview.scene().clear()
        self.preview.scene().addPixmap(pixmap(out))
        self.preview.setSceneRect(0, 0, out.shape[1], out.shape[0])
        self.preview.fit_image()


if __name__ == '__main__':
    import argparse
    import sys
    from PyQt6.QtWidgets import QApplication
    parser = argparse.ArgumentParser(description='Open the same manual editor directly on two original images')
    parser.add_argument('reference', type=Path)
    parser.add_argument('moving', type=Path)
    parser.add_argument('--model', choices=('projective', 'affine'), default='projective')
    args = parser.parse_args()
    paths = [args.reference, args.moving]
    images = [cv.imread(str(p), cv.IMREAD_UNCHANGED) for p in paths]
    if any(im is None for im in images):
        parser.error('Cannot decode input image')
    app = QApplication(sys.argv)
    dialog = CorrespondenceDialog(images, [str(p) for p in paths], args.model)
    dialog.exec()
