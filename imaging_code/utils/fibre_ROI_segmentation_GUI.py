# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:15:45 2025

@author: LuoD
"""


import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
    QFileDialog, QLabel, QLineEdit, QGridLayout, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor
from skimage.measure import regionprops
from scipy.ndimage import median_filter
import cv2


class ZoomableCanvas(FigureCanvas):
    def __init__(self, figure):
        super().__init__(figure)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setFocus()
        self._zoom = 1.0
        self._base_size = 512
        self.setFixedSize(self._base_size, self._base_size)

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120  # 1 per wheel step
        factor = 1.1 if delta > 0 else 0.9
        self._zoom *= factor
        size = int(self._base_size * self._zoom)
        size = max(256, min(2048, size))
        self.setFixedSize(size, size)
        self.draw()


class ROIEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('fibre-segger v1')
        self.resize(1200, 700)
        self.ref_image = None
        self.roi_dict = {}
        self.selected = set()
        self.labelled = None
        self.initUI()

    def initUI(self):
        # background palette
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(220, 220, 220))
        palette.setColor(QPalette.Base, QColor(240, 240, 240))
        palette.setColor(QPalette.Button, QColor(255, 255, 255))
        palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
        palette.setColor(QPalette.Text, QColor(0, 0, 0))
        palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
        self.setPalette(palette)

        self.fig = Figure(dpi=100, facecolor='white')
        self.canvas = ZoomableCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')

        # left column
        btn_load = QPushButton('load image')
        btn_segment = QPushButton('run segmentation')
        self.param_inputs = {}
        grid = QGridLayout()
        param_names = ['area', 'aspect_ratio', 'solidity', 'eccentricity', 'thinness']
        defaults = [10, 2, 0.7, 0.85, 0.5]
        for i, (name, default) in enumerate(zip(param_names, defaults)):
            label = QLabel(name)
            inp = QLineEdit(str(default))
            self.param_inputs[name] = inp
            grid.addWidget(label, i, 0)
            grid.addWidget(inp, i, 1)

        left_layout = QVBoxLayout()
        left_layout.setAlignment(Qt.AlignTop)
        left_layout.addWidget(btn_load)
        left_layout.addLayout(grid)
        left_layout.addWidget(btn_segment)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setFixedWidth(200)

        canvas_layout = QVBoxLayout()
        canvas_layout.setAlignment(Qt.AlignCenter)
        canvas_layout.addWidget(self.canvas)
        canvas_widget = QWidget()
        canvas_widget.setLayout(canvas_layout)
        canvas_widget.setFixedWidth(600)

        btn_delete = QPushButton('delete selected')
        btn_merge = QPushButton('merge selected')
        btn_save = QPushButton('save ROI dict')
        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignTop)
        right_layout.addWidget(btn_delete)
        right_layout.addWidget(btn_merge)
        right_layout.addWidget(btn_save)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        right_widget.setFixedWidth(200)

        full_layout = QHBoxLayout()

        left_col = QVBoxLayout()
        left_col.addStretch()
        left_col.addWidget(left_widget, alignment=Qt.AlignTop)
        left_col.addStretch()

        center_col = QVBoxLayout()
        center_col.addStretch()
        center_col.addWidget(canvas_widget, alignment=Qt.AlignCenter)
        center_col.addStretch()

        right_col = QVBoxLayout()
        right_col.addStretch()
        right_col.addWidget(right_widget, alignment=Qt.AlignTop)
        right_col.addStretch()

        full_layout.addLayout(left_col)
        full_layout.addLayout(center_col)
        full_layout.addLayout(right_col)

        container = QWidget()
        container.setLayout(full_layout)
        self.setCentralWidget(container)

        btn_load.clicked.connect(self.load_image)
        btn_segment.clicked.connect(self.run_segmentation)
        btn_delete.clicked.connect(self.delete_selected)
        btn_merge.clicked.connect(self.merge_selected)
        btn_save.clicked.connect(self.save_roi_dict)

        self.canvas.mpl_connect('button_press_event', self.on_click)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open NPY Image', filter='*.npy')
        if path:
            self.ref_image = np.load(path)
            self.plot_image()

    def run_segmentation(self):
        if self.ref_image is None:
            return
        try:
            params = {k: float(v.text()) for k, v in self.param_inputs.items()}
        except ValueError:
            print('Invalid segmentation parameters')
            return

        ref_filtered = median_filter(self.ref_image, size=(3, 3))
        thresh_val = np.percentile(ref_filtered, 40)
        binary_mask = ref_filtered > thresh_val
        ref_masked = ref_filtered.copy()
        ref_masked[~binary_mask] = 0
        ref_masked = np.clip(ref_masked, np.percentile(ref_masked, 0), np.percentile(ref_masked, 99))

        img_u8 = (ref_masked / ref_masked.max() * 255).astype(np.uint8)
        mser = cv2.MSER_create(5, 30, 500)
        mser.setMaxVariation(1.0)
        regions, _ = mser.detectRegions(img_u8)

        mask = np.zeros_like(img_u8, dtype=np.int32)
        for i, region in enumerate(regions):
            mask[region[:, 1], region[:, 0]] = i + 1

        props = regionprops(mask)
        roi_dict = {}
        roi_id = 1
        for region in props:
            area = region.area
            ecc = region.eccentricity
            sol = region.solidity
            ar = region.major_axis_length / (region.minor_axis_length + 1e-6)
            perim = region.perimeter
            thin = 4 * np.pi * area / (perim ** 2)
            if (area > params['area'] and sol < params['solidity'] and
                ecc > params['eccentricity'] and ar > params['aspect_ratio'] and thin < params['thinness']):
                ypix, xpix = region.coords[:, 0], region.coords[:, 1]
                roi_dict[roi_id] = {'xpix': xpix, 'ypix': ypix}
                roi_id += 1

        self.roi_dict = roi_dict
        self.labelled = np.zeros_like(self.ref_image, dtype=np.int32)
        for i, roi in self.roi_dict.items():
            self.labelled[roi['ypix'], roi['xpix']] = i
        self.selected.clear()
        self.plot_image()

    def plot_image(self):
        self.ax.clear()
        self.ax.axis('off')
        if self.ref_image is None:
            self.canvas.draw()
            return
        self.ax.imshow(self.ref_image, cmap='gray',
                       vmin=np.percentile(self.ref_image, 0.5),
                       vmax=np.percentile(self.ref_image, 99.9))
        if self.labelled is not None:
            overlay = np.zeros((*self.labelled.shape, 4))
            ids = np.unique(self.labelled)
            ids = ids[ids > 0]
            rng = np.random.default_rng(0)
            colours = rng.uniform(0.2, 1.0, size=(len(ids), 3))
            for idx, roi_id in enumerate(ids):
                mask = self.labelled == roi_id
                overlay[mask, :3] = colours[idx]
                overlay[mask, 3] = 0.5
            self.ax.imshow(overlay)
            for roi_id in self.selected:
                coords = np.column_stack(np.where(self.labelled == roi_id))
                self.ax.plot(coords[:, 1], coords[:, 0], 'c.', markersize=1)
        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax or self.labelled is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        roi_id = self.labelled[y, x]
        if roi_id > 0:
            if roi_id in self.selected:
                self.selected.remove(roi_id)
            else:
                self.selected.add(roi_id)
            self.plot_image()

    def delete_selected(self):
        if self.labelled is None:
            return
        for roi_id in self.selected:
            self.labelled[self.labelled == roi_id] = 0
        self.selected.clear()
        self.update_roi_dict()
        self.plot_image()

    def merge_selected(self):
        if self.labelled is None or len(self.selected) < 2:
            return
        target_id = min(self.selected)
        for roi_id in self.selected:
            if roi_id != target_id:
                self.labelled[self.labelled == roi_id] = target_id
        self.selected = {target_id}
        self.update_roi_dict()
        self.plot_image()

    def update_roi_dict(self):
        props = regionprops(self.labelled)
        self.roi_dict = {}
        for region in props:
            ypix, xpix = region.coords[:, 0], region.coords[:, 1]
            self.roi_dict[region.label] = {'xpix': xpix, 'ypix': ypix}

    def save_roi_dict(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save ROI Dict', filter='*.npz')
        if path:
            np.savez(path, **self.roi_dict)
            print(f'Saved to {path}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    editor = ROIEditor()
    editor.show()
    sys.exit(app.exec_())
