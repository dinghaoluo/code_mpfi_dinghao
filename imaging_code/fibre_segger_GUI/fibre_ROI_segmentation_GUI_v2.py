# -*- coding: utf-8 -*-
"""
Created on Mon 14 Apr 14:15:45 2025
Updated on Fri 18 Apr 14:50:12 2025
    patch note:
        - improved handling of ref image rendering 
        - now supports switching ON and OFF the ROI overlay 
        - now includes a few more parameters for customised segmentation 
        - now supports loading of saved ROI dict (needs to be *.npy)
        - fixed colour map issues so that colours are more salient

GUI for sorting fibres detected with MSER pipeline

@author: Dinghao Luo
"""

#%% imports 
import sys
import os 
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
    QFileDialog, QLabel, QLineEdit, QGridLayout, QTextEdit, QCheckBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor, QTextCursor, QIcon
from skimage.measure import regionprops
from scipy.ndimage import median_filter
import cv2
import colorsys


#%% main 
def generate_distinct_colours(n):
    """
    generate n visually distinct colours with high saturation and mid brightness.

    returns:
    - list of (r, g, b) tuples
    """
    colours = []
    for i in range(n):
        hue = i / n  # evenly spaced hues
        saturation = 0.9
        value = 0.75  # medium brightness
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colours.append(rgb)
    return colours
    

class OutputStream:
    def __init__(self, write_func):
        self.write_func = write_func

    def write(self, text):
        self.write_func(text)

    def flush(self):  # needed for compatibility
        pass


class ZoomableCanvas(FigureCanvas):
    def __init__(self, figure, ax):
        super().__init__(figure)
        self.ax = ax
        self.zoom = 1.0
        self._base_xlim = self.ax.get_xlim()
        self._base_ylim = self.ax.get_ylim()
        self._drag_start_pos = None
        self._drag_start_xlim = None
        self._drag_start_ylim = None

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._drag_start_pos = event.pos()
            self._drag_start_xlim = self.ax.get_xlim()
            self._drag_start_ylim = self.ax.get_ylim()
        else:
            super().mousePressEvent(event)  # allow mpl_connect to catch left click

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.RightButton and self._drag_start_pos is not None:
            dx = event.pos().x() - self._drag_start_pos.x()
            dy = event.pos().y() - self._drag_start_pos.y()
    
            trans = self.ax.transData.inverted()
            dx_data = trans.transform((0, 0))[0] - trans.transform((dx, 0))[0]
            dy_data = trans.transform((0, dy))[1] - trans.transform((0, 0))[1]  # flipped y
    
            new_xlim = (self._drag_start_xlim[0] + dx_data, self._drag_start_xlim[1] + dx_data)
            new_ylim = (self._drag_start_ylim[0] + dy_data, self._drag_start_ylim[1] + dy_data)
    
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self.draw()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self._drag_start_pos = None
        else:
            super().mouseReleaseEvent(event)  # let Matplotlib handle left click

    def reset_view(self):
        if self.ax.images:
            img = self.ax.images[0]
            self.ax.set_xlim(0, img.get_array().shape[1])
            self.ax.set_ylim(img.get_array().shape[0], 0)
            self.draw()
    
    def wheelEvent(self, event):
        if self.ax.images:
            xmouse = event.position().x()
            ymouse = event.position().y()
            xdata, ydata = self.ax.transData.inverted().transform((xmouse, ymouse))

            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            zoom_factor = 1 / 1.2 if event.angleDelta().y() > 0 else 1.2

            xleft = xdata - (xdata - cur_xlim[0]) * zoom_factor
            xright = xdata + (cur_xlim[1] - xdata) * zoom_factor
            ytop = ydata - (ydata - cur_ylim[0]) * zoom_factor
            ybottom = ydata + (cur_ylim[1] - ydata) * zoom_factor

            self.ax.set_xlim([xleft, xright])
            self.ax.set_ylim([ytop, ybottom])
            self.draw()


class ROIEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('fibre-segger v2.0')
        self.showMaximized()
        self.ref_image = None
        self.roi_dict = {}
        self.selected = set()
        self.labelled = None
        self.setWindowIcon(QIcon('fibre-segmenter.ico'))
        self.undo_stack = []  # for undoing merging and deletion
        self.show_overlay = True  # for turning on and off the ROI overlay 
        self.initUI()
        sys.stdout = OutputStream(self.append_output)
        sys.stderr = OutputStream(self.append_output)

    def initUI(self):
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(220, 220, 220))
        palette.setColor(QPalette.Base, QColor(240, 240, 240))
        palette.setColor(QPalette.Button, QColor(255, 255, 255))
        palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
        palette.setColor(QPalette.Text, QColor(0, 0, 0))
        palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
        self.setPalette(palette)

        self.fig = Figure(dpi=100, facecolor='white')
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.canvas = ZoomableCanvas(self.fig, self.ax)
        self.canvas.setFixedSize(1000, 1000)

        btn_load = QPushButton('load image')
        btn_load_roi = QPushButton('load ROI dict')
        btn_segment = QPushButton('run segmentation')
        self.param_inputs = {}
        grid = QGridLayout()
        param_names = [
            'clip-percentile',
            'MSER threshold', 
            'MSER max variation',
            'area', 
            'aspect ratio', 
            'solidity', 
            'eccentricity', 
            'thinness'
            ]
        defaults = [
            99,
            20, 
            1.0,
            100, 
            1.5, 
            0.7, 
            0.75, 
            0.5
            ]
        for i, (name, default) in enumerate(zip(param_names, defaults)):
            label = QLabel(name)
            inp = QLineEdit(str(default))
            self.param_inputs[name] = inp
            grid.addWidget(label, i, 0)
            grid.addWidget(inp, i, 1)

        left_layout = QVBoxLayout()
        left_layout.setAlignment(Qt.AlignTop)
        left_layout.addWidget(btn_load)
        left_layout.addWidget(btn_load_roi)
        left_layout.addLayout(grid)
        left_layout.addWidget(btn_segment)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setFixedWidth(400)

        canvas_layout = QVBoxLayout()
        canvas_layout.setAlignment(Qt.AlignCenter)
        canvas_layout.addWidget(self.canvas)
        canvas_widget = QWidget()
        canvas_widget.setLayout(canvas_layout)
        canvas_widget.setFixedWidth(1000)

        btn_delete = QPushButton('delete selected')
        btn_merge = QPushButton('merge selected')
        btn_undo = QPushButton('undo')
        btn_save = QPushButton('save ROI dict')
        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignTop)
        right_layout.addWidget(btn_delete)
        right_layout.addWidget(btn_merge)
        right_layout.addWidget(btn_undo)
        right_layout.addWidget(btn_save)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        right_widget.setFixedWidth(400)
        
        # output box
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        right_layout.addWidget(self.output_box)
        
        # overlay 
        self.overlay_toggle = QCheckBox('ROI overlay')
        self.overlay_toggle.setChecked(True)
        self.overlay_toggle.stateChanged.connect(self.toggle_overlay)
        right_layout.addWidget(self.overlay_toggle)

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

        # column widths redefined as percentage of window width, 18 Apr 2025
        full_layout.addLayout(left_col, 1)
        full_layout.addLayout(center_col, 3)
        full_layout.addLayout(right_col, 1)

        container = QWidget()
        container.setLayout(full_layout)
        self.setCentralWidget(container)

        btn_load.clicked.connect(self.load_image)
        btn_load_roi.clicked.connect(self.load_roi_dict)
        btn_segment.clicked.connect(self.run_segmentation)
        btn_delete.clicked.connect(self.delete_selected)
        btn_merge.clicked.connect(self.merge_selected)
        btn_undo.clicked.connect(self.undo_action)
        btn_save.clicked.connect(self.save_roi_dict)
        
        self.canvas.mpl_connect('button_press_event', self.on_click)

    def append_output(self, text):
        self.output_box.moveCursor(QTextCursor.End)
        self.output_box.insertPlainText(text)
        self.output_box.ensureCursorVisible()
    
    def delete_selected(self):
        if self.labelled is None:
            return
        for roi_id in self.selected:
            self.undo_stack.append(self.labelled.copy())
            self.labelled[self.labelled == roi_id] = 0
        self.selected.clear()
        self.update_roi_dict()
        self.plot_image(preserve_view=True)
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.show_overlay = not self.show_overlay
            self.overlay_toggle.setChecked(self.show_overlay)
            self.plot_image(preserve_view=True)
    
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open NPY Image', filter='*.npy')
        if path:
            self.ref_image_path = path
            self.recname = os.path.basename(path).split('_ref_mat')[0]
            self.ref_image = np.load(path)
            
            # clear any previously loaded ROIs
            self.roi_dict.clear()
            self.labelled = None
            self.selected.clear()
            
            self.plot_image()
            self.canvas.reset_view()
            self.append_output(f'{self.recname} loaded\n')
            
    def load_roi_dict(self):
        if self.ref_image is None:
            print('please load a reference image first')
            return
    
        path, _ = QFileDialog.getOpenFileName(self, 'load ROI Dict', filter='*.npy')
        if path:
            roi_dict = np.load(path, allow_pickle=True).item()
            if not isinstance(roi_dict, dict):
                print('invalid file format.')
                return
    
            self.roi_dict = roi_dict
            self.labelled = np.zeros_like(self.ref_image, dtype=np.int32)
            for roi_id, coords in self.roi_dict.items():
                self.labelled[coords['ypix'], coords['xpix']] = roi_id
            self.selected.clear()
            self.plot_image()
            self.canvas.reset_view()
            print(f'ROI dict loaded from {path}')
    
    def merge_selected(self):
        if self.labelled is None or len(self.selected) < 2:
            return
        target_id = min(self.selected)
        for roi_id in self.selected:
            self.undo_stack.append(self.labelled.copy())
            if roi_id != target_id:
                self.labelled[self.labelled == roi_id] = target_id
        self.selected = {target_id}
        self.update_roi_dict()
        self.plot_image(preserve_view=True)

    def on_click(self, event):
        if event.inaxes != self.ax or self.labelled is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        roi_id = self.labelled[y, x]
        if roi_id > 0:
            # check if shift is held down
            if event.guiEvent.modifiers() & Qt.ShiftModifier:
                if roi_id in self.selected:
                    self.selected.remove(roi_id)
                else:
                    self.selected.add(roi_id)
            else:
                self.selected = {roi_id}
            self.plot_image(preserve_view=True)

    def plot_image(self, preserve_view=False):
        try:
            params = {k: float(v.text()) for k, v in self.param_inputs.items()}
        except ValueError:
            print('invalid segmentation parameters')
            return
        
        if preserve_view:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
    
        self.ax.clear()
        self.ax.axis('off')
    
        if self.ref_image is None:
            self.canvas.draw()
            return
    
        self.ax.imshow(self.ref_image, cmap='gray',
                       vmin=np.percentile(self.ref_image, 0),
                       vmax=np.percentile(self.ref_image, params['clip-percentile']))
    
        if self.labelled is not None and self.show_overlay:  # overlay switch, 18 Apr 2025
            overlay = np.zeros((*self.labelled.shape, 4))
            ids = np.unique(self.labelled)
            ids = ids[ids > 0]
            colours = generate_distinct_colours(len(ids))
            for idx, roi_id in enumerate(ids):
                mask = self.labelled == roi_id
                overlay[mask, :3] = colours[idx]
                overlay[mask, 3] = 0.5
            self.ax.imshow(overlay)
            for roi_id in self.selected:
                coords = np.column_stack(np.where(self.labelled == roi_id))
                self.ax.plot(coords[:, 1], coords[:, 0], 'c.', markersize=1)
    
        if preserve_view:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
    
        self.canvas.draw()

    def run_segmentation(self):
        if self.ref_image is None:
            return
        try:
            params = {k: float(v.text()) for k, v in self.param_inputs.items()}
        except ValueError:
            print('invalid segmentation parameters')
            return

        ref_filtered = median_filter(self.ref_image, size=(3, 3))
        thresh_val = np.percentile(ref_filtered, params['MSER threshold'])
        binary_mask = ref_filtered > thresh_val
        ref_masked = ref_filtered.copy()
        ref_masked[~binary_mask] = 0
        ref_masked = np.clip(
            ref_masked, 
            np.percentile(ref_masked, 0), 
            np.percentile(ref_masked, params['clip-percentile'])
            )

        img_u8 = (ref_masked / ref_masked.max() * 255).astype(np.uint8)
        mser = cv2.MSER_create(5, 30, 500)
        mser.setMaxVariation(params['MSER max variation'])
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
                ecc > params['eccentricity'] and ar > params['aspect ratio'] and thin < params['thinness']):
                ypix, xpix = region.coords[:, 0], region.coords[:, 1]
                roi_dict[roi_id] = {'xpix': xpix, 'ypix': ypix}
                roi_id += 1

        self.roi_dict = roi_dict
        self.labelled = np.zeros_like(self.ref_image, dtype=np.int32)
        for i, roi in self.roi_dict.items():
            self.labelled[roi['ypix'], roi['xpix']] = i
        self.selected.clear()
        self.plot_image()
        self.canvas.reset_view()
    
    def save_roi_dict(self):
        if not hasattr(self, 'ref_image_path') or not hasattr(self, 'recname'):
            print('reference image not loaded, cannot auto-save.')
            return
        save_dir = os.path.dirname(self.ref_image_path)
        save_name = f'{self.recname}_ROI_dict.npy'
        save_path = os.path.join(save_dir, save_name)
        np.save(save_path, self.roi_dict)
        print(f'saved to {save_path}')
        
    def toggle_overlay(self):
        self.show_overlay = self.overlay_toggle.isChecked()
        self.plot_image(preserve_view=True)
    
    def undo_action(self):
        if self.undo_stack:
            self.labelled = self.undo_stack.pop()
            self.update_roi_dict()
            self.selected.clear()
            self.plot_image(preserve_view=True)

    def update_roi_dict(self):
        props = regionprops(self.labelled)
        self.roi_dict = {}
        for region in props:
            ypix, xpix = region.coords[:, 0], region.coords[:, 1]
            self.roi_dict[region.label] = {'xpix': xpix, 'ypix': ypix}


if __name__ == '__main__':
    app = QApplication(sys.argv)
    editor = ROIEditor()
    editor.show()
    sys.exit(app.exec_())
