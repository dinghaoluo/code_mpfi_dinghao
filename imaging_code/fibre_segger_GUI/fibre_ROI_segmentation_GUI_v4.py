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
        - added 'fix selected' button; fixed ROIs persist across segmentation runs
Updated on Thur 27 Nov 2025 
    patch note:
        - now includes a 'fix' functionality to fix ROI in the dict 

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


def enhance_contrast_u8(img, tophat_kernel=11, clahe_clip=2.0):
    """
    apply white top-hat (thin bright structures) and CLAHE.

    parameters:
    - tophat_kernel: odd int, structuring element size
    - clahe_clip: float, higher = stronger equalisation

    returns:
    - uint8 image scaled to [0,255]
    """
    img = img.astype(np.float32)
    # rescale to [0,1] robustly
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    if hi <= lo:
        hi = lo + 1.0
    img01 = np.clip((img - lo) / (hi - lo), 0, 1)

    k = int(tophat_kernel) if int(tophat_kernel) % 2 == 1 else int(tophat_kernel) + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx((img01 * 255).astype(np.uint8), cv2.MORPH_TOPHAT, se)

    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
    eq = clahe.apply(tophat)

    return eq


class ROIEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('fibre-segger v3.0')
        self.showMaximized()
        self.ref_image = None
        self.roi_dict = {}
        self.selected = set()
        self.labelled = None
        self.fixed_ids = set()  # ids of ROIs that are fixed across segmentation runs
        self.setWindowIcon(QIcon('fibre-segmenter.ico'))
        self.undo_stack = []  # for undoing merging and deletion (stores full state)
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
            'clip-percentile',         # 95–99
            'MSER threshold',          # percentile
            'MSER max variation',      # 0.5–2.0
            'MSER delta',              # 3–8
            'MSER min area',           # 20–200
            'MSER max area',           # 5000–20000
            'area min',                # 50–300
            'aspect ratio min',        # 1.2–1.8
            'solidity min',            # lower for more promiscuous 
            'eccentricity min',        # 0.65–0.9
            'thinness max',            # 0.6–0.9  (circularity; higher = more lenient)
            'tophat kernel',           # 3–17 (odd)
            'clahe clip'               # 1.0–3.0
        ]
        defaults = [
            99,
            80,
            1.2,
            5,
            30,
            15000,
            100,
            1.4,
            0.1,
            0.75,
            0.8,
            11,
            2.0
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
        btn_fix = QPushButton('fix selected')
        btn_remove_fixed = QPushButton('remove fixed')
        btn_undo = QPushButton('undo')
        btn_save = QPushButton('save ROI dict')
        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignTop)
        right_layout.addWidget(btn_delete)
        right_layout.addWidget(btn_merge)
        right_layout.addWidget(btn_fix)
        right_layout.addWidget(btn_remove_fixed)
        right_layout.addWidget(btn_undo)
        right_layout.addWidget(btn_save)
        
        # output box
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        right_layout.addWidget(self.output_box)
        
        # overlay 
        self.overlay_toggle = QCheckBox('ROI overlay')
        self.overlay_toggle.setChecked(True)
        self.overlay_toggle.stateChanged.connect(self.toggle_overlay)
        right_layout.addWidget(self.overlay_toggle)

        # clear board: remove all non-fixed rois (or all if none are fixed)
        btn_clear_board = QPushButton('clear board')
        right_layout.addWidget(btn_clear_board)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        right_widget.setFixedWidth(400)

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
        btn_fix.clicked.connect(self.fix_selected)
        btn_remove_fixed.clicked.connect(self.remove_fixed)
        btn_undo.clicked.connect(self.undo_action)
        btn_save.clicked.connect(self.save_roi_dict)
        btn_clear_board.clicked.connect(self.clear_board)
        
        self.canvas.mpl_connect('button_press_event', self.on_click)

    def append_output(self, text):
        self.output_box.moveCursor(QTextCursor.End)
        self.output_box.insertPlainText(text)
        self.output_box.ensureCursorVisible()
    
    def push_undo_state(self):
        # store full state: label map, roi dict, fixed ids
        if self.labelled is None:
            return
        roi_dict_copy = {
            k: {'xpix': v['xpix'].copy(), 'ypix': v['ypix'].copy()}
            for k, v in self.roi_dict.items()
        }
        fixed_ids_copy = set(self.fixed_ids)
        self.undo_stack.append((self.labelled.copy(), roi_dict_copy, fixed_ids_copy))
    
    def delete_selected(self):
        if self.labelled is None or not self.selected:
            return
        self.push_undo_state()
        for roi_id in self.selected:
            self.labelled[self.labelled == roi_id] = 0
            if roi_id in self.fixed_ids:
                self.fixed_ids.discard(roi_id)
        self.selected.clear()
        self.update_roi_dict()
        self.plot_image(preserve_view=True)
        
    def fix_selected(self):
        # mark currently selected rois as fixed so they persist across segmentation
        if self.labelled is None or not self.selected:
            return
        newly_fixed = []
        for roi_id in self.selected:
            if roi_id in self.roi_dict and roi_id not in self.fixed_ids:
                self.fixed_ids.add(roi_id)
                newly_fixed.append(roi_id)
        if newly_fixed:
            print(f'fixed ROI(s): {newly_fixed} (will persist across segmentation)')
        self.plot_image(preserve_view=True)

    def remove_fixed(self):
        # remove all fixed rois entirely
        if self.labelled is None or not self.fixed_ids:
            return
        self.push_undo_state()
        for fid in self.fixed_ids:
            self.labelled[self.labelled == fid] = 0
        self.fixed_ids.clear()
        self.selected.clear()
        self.update_roi_dict()
        self.plot_image(preserve_view=True)
        print('removed all fixed ROIs')
        
    def keyPressEvent(self, event):
        # use space to turn on and off view
        if event.key() == Qt.Key_Space:
            self.show_overlay = not self.show_overlay
            self.overlay_toggle.setChecked(self.show_overlay)
            self.plot_image(preserve_view=True)
        
        # use delete to delete selected roi
        if event.key() == Qt.Key_Delete:
            self.delete_selected()
            
        # use backspace to undo 
        if event.key() == Qt.Key_Backspace:
            self.undo_action()
        
        # use ctrl + m to merge
        if event.key() == Qt.Key_M and (event.modifiers() & Qt.ControlModifier):
            self.merge_selected()
    
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open NPY Image', filter='*.npy')
        if path:
            self.ref_image_path = path
            self.recname = os.path.basename(path).split('_ref_mat')[0]
            self.ref_image = np.load(path)
            
            # clear any previously loaded rois
            self.roi_dict.clear()
            self.labelled = None
            self.selected.clear()
            self.fixed_ids.clear()
            self.undo_stack.clear()
            
            self.plot_image()
            self.canvas.reset_view()
            self.append_output(f'{self.recname} loaded\n')
            
            # auto-check for roi dict, 15 sept 2025 
            roi_dict_path = os.path.join(os.path.dirname(path), f'{self.recname}_ROI_dict.npy')
            if os.path.exists(roi_dict_path):
                try:
                    roi_dict = np.load(roi_dict_path, allow_pickle=True).item()
                    if isinstance(roi_dict, dict):
                        self.roi_dict = roi_dict
                        self.labelled = np.zeros_like(self.ref_image, dtype=np.int32)
                        for roi_id, coords in self.roi_dict.items():
                            self.labelled[coords['ypix'], coords['xpix']] = roi_id
                        self.selected.clear()
                        self.fixed_ids.clear()
                        self.undo_stack.clear()
                        self.plot_image()
                        self.canvas.reset_view()
                        self.append_output(f'ROI dict loaded automatically from {roi_dict_path}\n')
                except Exception as e:
                    self.append_output(f'failed to load ROI dict: {e}\n')
            else:  # if no roi dict exists
                self.run_segmentation()
            
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
            self.fixed_ids.clear()
            self.undo_stack.clear()
            self.plot_image()
            self.canvas.reset_view()
            print(f'ROI dict loaded from {path}')
    
    def merge_selected(self):
        if self.labelled is None or len(self.selected) < 2:
            return
        self.push_undo_state()
        target_id = min(self.selected)
        for roi_id in self.selected:
            if roi_id != target_id:
                self.labelled[self.labelled == roi_id] = target_id
        # update fixed ids: if any merged roi was fixed, keep target fixed
        if any(roi_id in self.fixed_ids for roi_id in self.selected):
            self.fixed_ids.add(target_id)
        # remove merged-away ids from fixed set
        self.fixed_ids = {
            fid for fid in self.fixed_ids
            if fid == target_id or fid not in self.selected
        }
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
        
        lo = np.percentile(self.ref_image, 1)
        hi = np.percentile(self.ref_image, params['clip-percentile'])
        self.ax.imshow(self.ref_image, cmap='gray',
                       vmin=lo,
                       vmax=hi)
    
        if self.labelled is not None and self.show_overlay:  # overlay switch, 18 apr 2025
            overlay = np.zeros((*self.labelled.shape, 4))
            ids = np.unique(self.labelled)
            ids = ids[ids > 0]
            colours = generate_distinct_colours(len(ids))
            for idx, roi_id in enumerate(ids):
                mask = self.labelled == roi_id
                overlay[mask, :3] = colours[idx]
                alpha = 0.5
                if roi_id in self.fixed_ids:
                    alpha = 0.8  # fixed rois drawn slightly more solid
                overlay[mask, 3] = alpha
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
            p = {k: float(v.text()) for k, v in self.param_inputs.items()}
        except ValueError:
            print('invalid segmentation parameters')
            return

        # cache current fixed rois (coords) before overwriting roi_dict
        fixed_rois = {}
        if self.roi_dict and self.fixed_ids:
            for fid in list(self.fixed_ids):
                coords = self.roi_dict.get(fid, None)
                if coords is not None:
                    fixed_rois[fid] = {
                        'xpix': coords['xpix'].copy(),
                        'ypix': coords['ypix'].copy()
                    }
                else:
                    # fixed id no longer exists in roi_dict
                    self.fixed_ids.discard(fid)
    
        # denoise whilst preserving ridges
        ref_f = median_filter(self.ref_image, size=(3, 3))
    
        # contrast boost for thin bright fibres
        img_u8 = enhance_contrast_u8(
            ref_f,
            tophat_kernel=p['tophat kernel'],
            clahe_clip=p['clahe clip']
        )
    
        # soft gate: keep pixels above a low percentile (keeps faint axons)
        thr = np.percentile(img_u8, p['MSER threshold'])
        soft = img_u8.copy()
        soft[soft < thr] = 0  # zero background without crushing mid-high values
    
        # mser with tunable size + stability
        delta = int(max(1, round(p['MSER delta'])))
        min_area = int(max(5, round(p['MSER min area'])))
        max_area = int(max(min_area + 1, round(p['MSER max area'])))
        mser = cv2.MSER_create(delta, min_area, max_area)
        mser.setMaxVariation(p['MSER max variation'])
    
        regions, _ = mser.detectRegions(soft)
    
        # initial label map from mser regions
        lab = np.zeros_like(img_u8, dtype=np.int32)
        for i, reg in enumerate(regions):
            lab[reg[:, 1], reg[:, 0]] = i + 1
    
        # region filters (more lenient)
        props = regionprops(lab)
        seg_rois = []
        for r in props:
            area = r.area
            if area < p['area min']:
                continue
    
            # geometry
            ecc = r.eccentricity if np.isfinite(r.eccentricity) else 0.0
            sol = r.solidity if np.isfinite(r.solidity) else 0.0
            maj = r.major_axis_length
            minax = r.minor_axis_length if r.minor_axis_length > 1e-6 else 1e-6
            ar = maj / minax
            perim = r.perimeter if r.perimeter > 1e-6 else 1e-6
            thin = 4 * np.pi * area / (perim ** 2)
    
            # relaxed accept criteria:
            if sol < p['solidity min']:
                continue
            if ecc < p['eccentricity min']:
                continue
            if ar < p['aspect ratio min']:
                continue
            if thin > p['thinness max']:  # too round/fat
                continue
    
            ypix, xpix = r.coords[:, 0], r.coords[:, 1]
            seg_rois.append({'xpix': xpix, 'ypix': ypix})
    
        # build combined roi_dict: fixed first, then new segmentation (non-overlapping)
        self.labelled = np.zeros_like(self.ref_image, dtype=np.int32)
        self.roi_dict = {}
        new_fixed_ids = set()
        roi_id = 1

        # add fixed rois with priority
        for _, coords in fixed_rois.items():
            ypix = coords['ypix']
            xpix = coords['xpix']
            self.roi_dict[roi_id] = {'xpix': xpix, 'ypix': ypix}
            self.labelled[ypix, xpix] = roi_id
            new_fixed_ids.add(roi_id)
            roi_id += 1

        # add new segmentation rois (skip overlaps with existing, i.e., fixed)
        for roi in seg_rois:
            ypix = roi['ypix']
            xpix = roi['xpix']
            if np.any(self.labelled[ypix, xpix] > 0):
                # overlap with an existing roi (likely fixed); skip to give priority to fixed
                continue
            self.roi_dict[roi_id] = {'xpix': xpix, 'ypix': ypix}
            self.labelled[ypix, xpix] = roi_id
            roi_id += 1

        self.fixed_ids = new_fixed_ids
        self.selected.clear()
        self.plot_image()
        self.canvas.reset_view()
        print(f'MSER regions: {len(regions)} | kept ROIs: {len(self.roi_dict)} (including {len(self.fixed_ids)} fixed)')
    
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

    def clear_board(self):
        # remove all detected rois except the fixed ones
        if self.labelled is None:
            return

        # if no fixed rois, just clear everything
        if not self.fixed_ids:
            self.push_undo_state()
            self.labelled = np.zeros_like(self.labelled, dtype=np.int32)
            self.roi_dict = {}
            self.selected.clear()
            self.fixed_ids.clear()
            self.plot_image(preserve_view=True)
            print('cleared all ROIs (no fixed ROIs present)')
            return

        # keep only fixed rois
        self.push_undo_state()
        new_labelled = np.zeros_like(self.labelled, dtype=np.int32)
        for fid in self.fixed_ids:
            new_labelled[self.labelled == fid] = fid
        self.labelled = new_labelled
        self.update_roi_dict()
        self.fixed_ids = {fid for fid in self.fixed_ids if fid in self.roi_dict}
        self.selected.clear()
        self.plot_image(preserve_view=True)
        print(f'cleared non-fixed ROIs; kept {len(self.fixed_ids)} fixed ROI(s)')
    
    def undo_action(self):
        if self.undo_stack:
            self.labelled, self.roi_dict, self.fixed_ids = self.undo_stack.pop()
            self.selected.clear()
            self.plot_image(preserve_view=True)

    def update_roi_dict(self):
        props = regionprops(self.labelled)
        self.roi_dict = {}
        for region in props:
            ypix, xpix = region.coords[:, 0], region.coords[:, 1]
            self.roi_dict[region.label] = {'xpix': xpix, 'ypix': ypix}
        # drop any fixed ids that no longer exist (e.g. after deletion)
        self.fixed_ids = {fid for fid in self.fixed_ids if fid in self.roi_dict}


if __name__ == '__main__':
    app = QApplication(sys.argv)
    editor = ROIEditor()
    editor.show()
    sys.exit(app.exec_())
