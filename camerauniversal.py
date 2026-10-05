import sys
import cv2
import numpy as np
import os
import glob
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox, 
                             QFileDialog, QDialog, QLineEdit, QCheckBox)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QUrl
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

# --- PYINSTALLER RESOURCE HELPER ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- THEMES (CSS / QSS) ---
THEME_DEFAULT = ""

THEME_NEON_FUTURISTIC = """
QWidget { background-color: #0d0d12; color: #00ffff; font-family: 'Segoe UI', sans-serif; }
QPushButton { background-color: transparent; border: 2px solid #ff00ff; border-radius: 8px; padding: 5px; font-weight: bold; }
QPushButton:hover { background-color: #ff00ff; color: #0d0d12; }
QComboBox, QLineEdit { background-color: #1a1a24; border: 1px solid #00ffff; color: #00ffff; border-radius: 4px; padding: 2px; }
QLabel { color: #00ffff; }
"""

THEME_FRUTIGER_AERO = """
QWidget { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #a8d5e2); color: #003366; font-family: 'Segoe UI', sans-serif; }
QPushButton { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #c0e8f0); border: 1px solid #82b5c4; border-radius: 8px; padding: 5px; font-weight: bold; }
QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e0f7fa, stop:1 #82b5c4); }
QComboBox, QLineEdit { background-color: rgba(255, 255, 255, 200); border: 1px solid #a8d5e2; border-radius: 4px; color: #003366; }
QLabel { font-weight: bold; text-shadow: 1px 1px 2px white; }
"""

# --- CUSTOM DROP-DOWN (Plays sound on click) ---
class SoundComboBox(QComboBox):
    popup_opened = pyqtSignal()
    
    def showPopup(self):
        self.popup_opened.emit()
        super().showPopup()

# --- HARDWARE DETECTION ---
def get_camera_names():
    cameras = {}
    if sys.platform.startswith('linux'):
        for path in sorted(glob.glob('/sys/class/video4linux/video*')):
            try:
                index = int(path.replace('/sys/class/video4linux/video', ''))
                with open(f"{path}/name", 'r') as f:
                    name = f.read().strip()
                is_internal = "integrated" in name.lower() or "webcam" in name.lower() or index == 0
                tag = "[Built-in Webcam]" if is_internal else "[External Camera]"
                cameras[index] = f"{tag} {name} (video{index})"
            except Exception:
                pass
                
    if not cameras:
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_ANY)
            if cap.isOpened():
                tag = "[Built-in Webcam]" if i == 0 else "[External Camera]"
                cameras[i] = f"{tag} Camera Device {i}"
                cap.release()
    return cameras

# --- SAFE CAMERA THREAD ---
class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.camera_index = 0
        self.resolution = (640, 480)
        self.effect = "None"
        self.cap = None
        self.is_recording = False
        self.video_writer = None
        
        # Pending updates applied safely inside the loop
        self._pending_res = None
        self._pending_cam_idx = None

    def set_resolution_safe(self, w, h):
        self._pending_res = (w, h)
        
    def set_camera_safe(self, idx):
        self._pending_cam_idx = idx

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_ANY)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        while self._run_flag:
            # Handle resolution change without killing camera
            if self._pending_res is not None:
                self.resolution = self._pending_res
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self._pending_res = None
                
            # Handle camera index switch
            if self._pending_cam_idx is not None:
                self.cap.release()
                self.camera_index = self._pending_cam_idx
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_ANY)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self._pending_cam_idx = None

            ret, frame = self.cap.read()
            if ret:
                if self.effect == "Grayscale":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif self.effect == "Sepia":
                    kernel = np.array([[0.272, 0.534, 0.131],
                                       [0.349, 0.686, 0.168],
                                       [0.393, 0.769, 0.189]])
                    frame = cv2.transform(frame, kernel)
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                elif self.effect == "Invert":
                    frame = cv2.bitwise_not(frame)

                if self.is_recording and self.video_writer is not None:
                    self.video_writer.write(frame)

                self.change_pixmap_signal.emit(frame)
                
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()
        if self.cap is not None:
            self.cap.release()

    def start_recording(self, path, format_ext):
        codec_str = 'mp4v' if format_ext == '.mp4' else 'XVID'
        fourcc = cv2.VideoWriter_fourcc(*codec_str)
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 30 
        self.video_writer = cv2.VideoWriter(path, fourcc, fps, self.resolution)
        self.is_recording = True

    def stop_recording(self):
        self.is_recording = False
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def stop(self):
        self._run_flag = False
        self.wait()

# --- SAVE CONFIRMATION DIALOG ---
class CustomSaveDialog(QDialog):
    def __init__(self, parent, frame, folder, default_name, mode="image"):
        super().__init__(parent)
        self.setWindowTitle(f"Save {mode.capitalize()}")
        self.folder = folder
        self.mode = mode
        self.final_path = ""
        
        self.file_size_kb = 0
        if mode == "image" and frame is not None:
            success, encoded_image = cv2.imencode('.jpg', frame)
            if success: self.file_size_kb = len(encoded_image.tobytes()) / 1024

        self.initUI(default_name)
        
    def initUI(self, default_name):
        layout = QVBoxLayout()
        size_txt = f"<b>Est. Size:</b> {self.file_size_kb:.2f} KB<br>" if self.mode == "image" else "<b>Est. Size:</b> Depends on duration<br>"
        self.info_label = QLabel(f"{size_txt}<b>Path:</b> {self.folder}")
        layout.addWidget(self.info_label)

        name_layout = QHBoxLayout()
        self.name_input = QLineEdit(default_name)
        
        self.type_combo = QComboBox()
        if self.mode == "image":
            self.type_combo.addItems([".jpg", ".png", ".bmp"])
        else:
            self.type_combo.addItems([".mp4", ".avi"])
            
        name_layout.addWidget(QLabel("Filename:"))
        name_layout.addWidget(self.name_input)
        name_layout.addWidget(self.type_combo)
        layout.addLayout(name_layout)
        
        btn_layout = QHBoxLayout()
        
        btn_accept = QPushButton("Accept Save" if self.mode == "image" else "Accept Record")
        btn_accept.clicked.connect(self.process_accept)
        
        btn_change = QPushButton("Change Folder")
        btn_change.clicked.connect(self.change_folder)
        
        btn_decline = QPushButton("Decline")
        btn_decline.clicked.connect(self.reject)
        
        btn_layout.addWidget(btn_accept)
        btn_layout.addWidget(btn_change)
        btn_layout.addWidget(btn_decline)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def change_folder(self):
        new_dir = QFileDialog.getExistingDirectory(self, "Select Folder", self.folder)
        if new_dir:
            self.folder = new_dir
            size_txt = f"<b>Est. Size:</b> {self.file_size_kb:.2f} KB<br>" if self.mode == "image" else "<b>Est. Size:</b> Depends on duration<br>"
            self.info_label.setText(f"{size_txt}<b>Path:</b> {self.folder}")

    def process_accept(self):
        name = self.name_input.text().strip()
        ext = self.type_combo.currentText()
        if not name.endswith(ext):
            name += ext
            
        self.final_path = os.path.join(self.folder, name)
        self.accept()

# --- MAIN APP ---
class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CameraUniversalPC")
        self.resize(800, 600)
        
        self.save_folder = os.path.expanduser("~")
        self.current_frame = None
        self.fx_enabled = True
        self.show_hitmarker = False

        # Audio setup
        self.audio_out_click = QAudioOutput()
        self.player_click = QMediaPlayer()
        self.player_click.setAudioOutput(self.audio_out_click)
        click_path = resource_path(os.path.join("AudioFolder", "click.mp3"))
        self.player_click.setSource(QUrl.fromLocalFile(click_path))

        self.audio_out_cam = QAudioOutput()
        self.player_cam = QMediaPlayer()
        self.player_cam.setAudioOutput(self.audio_out_cam)
        cam_path = resource_path(os.path.join("AudioFolder", "camera.mp3"))
        self.player_cam.setSource(QUrl.fromLocalFile(cam_path))

        self.initUI()
        self.start_camera()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # 1. Video Feed (Top)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: black;")
        main_layout.addWidget(self.image_label)

        # 2. Controls Layout (Middle)
        control_layout = QHBoxLayout()

        self.cam_combo = SoundComboBox()
        self.detect_cameras()
        self.cam_combo.popup_opened.connect(lambda: self.play_sound("click"))
        self.cam_combo.currentIndexChanged.connect(self.on_cam_changed)
        control_layout.addWidget(self.cam_combo)

        self.res_combo = SoundComboBox()
        self.res_combo.addItems(["640x480", "1280x720", "1920x1080"])
        self.res_combo.popup_opened.connect(lambda: self.play_sound("click"))
        self.res_combo.currentIndexChanged.connect(self.on_res_changed)
        control_layout.addWidget(self.res_combo)

        self.theme_combo = SoundComboBox()
        self.theme_combo.addItems(["Default", "Neon Futuristic", "Frutiger Aero"])
        self.theme_combo.popup_opened.connect(lambda: self.play_sound("click"))
        self.theme_combo.currentIndexChanged.connect(self.on_theme_changed)
        control_layout.addWidget(self.theme_combo)

        self.effect_combo = SoundComboBox()
        self.effect_combo.addItems(["None", "Grayscale", "Sepia", "Invert"])
        self.effect_combo.popup_opened.connect(lambda: self.play_sound("click"))
        self.effect_combo.currentIndexChanged.connect(self.on_effect_changed)
        control_layout.addWidget(self.effect_combo)

        main_layout.addLayout(control_layout)

        # 3. Action Buttons Layout (Bottom)
        btn_layout = QHBoxLayout()
        
        self.btn_capture = QPushButton("📷 Capture Image")
        self.btn_capture.clicked.connect(self.capture_image)
        
        self.btn_record = QPushButton("⏺️ Start Recording (Beta)")
        self.btn_record.clicked.connect(self.toggle_recording)
        
        self.btn_folder = QPushButton("📁 Set Save Folder")
        self.btn_folder.clicked.connect(self.set_save_folder)

        self.toggle_fx = QCheckBox("Sound & Hitmarker FX")
        self.toggle_fx.setChecked(True)
        self.toggle_fx.stateChanged.connect(self.on_fx_toggled)

        btn_layout.addWidget(self.btn_capture)
        btn_layout.addWidget(self.btn_record)
        btn_layout.addWidget(self.btn_folder)
        btn_layout.addWidget(self.toggle_fx)

        main_layout.addLayout(btn_layout)

    def detect_cameras(self):
        self.cam_map = get_camera_names()
        self.cam_combo.clear()
        for idx, name in self.cam_map.items():
            self.cam_combo.addItem(name, idx)

    def start_camera(self):
        self.thread = CameraThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def play_sound(self, sound_type="click"):
        if not self.fx_enabled:
            return
        if sound_type == "camera":
            self.player_cam.stop()
            self.player_cam.play()
        else:
            self.player_click.stop()
            self.player_click.play()

    def play_hitmarker_fx(self):
        if self.fx_enabled:
            self.show_hitmarker = True
            QTimer.singleShot(250, self.clear_hitmarker)

    def clear_hitmarker(self):
        self.show_hitmarker = False

    def update_image(self, cv_img):
        self.current_frame = cv_img.copy()
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)

        if self.show_hitmarker:
            painter = QPainter(pixmap)
            pen = QPen(QColor(255, 0, 0), 3) 
            painter.setPen(pen)
            cx, cy = pixmap.width() // 2, pixmap.height() // 2
            size = 20
            painter.drawLine(cx - size, cy - size, cx - 5, cy - 5)
            painter.drawLine(cx + size, cy - size, cx + 5, cy - 5)
            painter.drawLine(cx - size, cy + size, cx - 5, cy + 5)
            painter.drawLine(cx + size, cy + size, cx + 5, cy + 5)
            painter.end()

        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def capture_image(self):
        self.play_sound("click")
        if self.current_frame is None:
            return

        default_name = f"Capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dlg = CustomSaveDialog(self, self.current_frame, self.save_folder, default_name, mode="image")
        
        if dlg.exec():
            save_path = dlg.final_path
            cv2.imwrite(save_path, self.current_frame)
            self.play_sound("camera")
            self.play_hitmarker_fx()

    def toggle_recording(self):
        self.play_sound("click")
        if not self.thread.is_recording:
            default_name = f"Record_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            dlg = CustomSaveDialog(self, None, self.save_folder, default_name, mode="video")
            
            if dlg.exec():
                save_path = dlg.final_path
                ext = dlg.type_combo.currentText()
                self.thread.start_recording(save_path, ext)
                self.btn_record.setText("⏹️ Stop Recording")
                self.play_sound("camera")
                self.play_hitmarker_fx()
        else:
            self.thread.stop_recording()
            self.btn_record.setText("⏺️ Start Recording (Beta)")
            self.play_sound("camera")
            self.play_hitmarker_fx()

    def set_save_folder(self):
        self.play_sound("click")
        folder = QFileDialog.getExistingDirectory(self, "Select Save Directory", self.save_folder)
        if folder:
            self.save_folder = folder

    def on_cam_changed(self, idx):
        self.play_sound("click")
        cam_idx = self.cam_combo.itemData(idx)
        if cam_idx is not None and self.thread.camera_index != cam_idx:
            # Safely request the running thread to change cameras
            self.thread.set_camera_safe(cam_idx)

    def on_res_changed(self):
        self.play_sound("click")
        res_str = self.res_combo.currentText()
        w, h = map(int, res_str.split('x'))
        # Safely request the running thread to alter the resolution properties
        self.thread.set_resolution_safe(w, h)

    def on_theme_changed(self):
        self.play_sound("click")
        theme = self.theme_combo.currentText()
        if theme == "Neon Futuristic":
            QApplication.instance().setStyleSheet(THEME_NEON_FUTURISTIC)
        elif theme == "Frutiger Aero":
            QApplication.instance().setStyleSheet(THEME_FRUTIGER_AERO)
        else:
            QApplication.instance().setStyleSheet(THEME_DEFAULT)

    def on_effect_changed(self):
        self.play_sound("click")
        self.thread.effect = self.effect_combo.currentText()

    def on_fx_toggled(self, state):
        self.fx_enabled = (state == 2)
        if self.fx_enabled:
            self.play_sound("click")

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec())
