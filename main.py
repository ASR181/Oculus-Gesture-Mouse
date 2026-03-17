import sys
import os
import cv2
import numpy as np
import time
import autopy
import pyautogui

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QSlider, QCheckBox, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QIcon

import HandTrackingModule as htm

# Fix for PyInstaller file paths
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# drop pyautogui delay to prevent lag stacking in the background
pyautogui.PAUSE = 0 

class TrackingThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    stats_signal = pyqtSignal(int, float)

    def __init__(self):
        super().__init__()
        self.running = False
        
        # camera and screen settings
        self.w_cam, self.h_cam = 640, 480
        self.w_scr, self.h_scr = autopy.screen.size()
        self.frame_reduc = 100
        self.smoothen = 7
        
        # user toggles
        self.enable_movement = True
        self.enable_click = True
        self.enable_scroll = True
        
        # visual theme defaults
        self.skeleton_color = (255, 229, 0) # default cyan for dark mode (BGR)
        
        self.detector = htm.HandDetector(max_hands=1, detection_con=0.7)

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(0)
        cap.set(3, self.w_cam)
        cap.set(4, self.h_cam)
        
        p_time = 0
        pre_loc_x, pre_loc_y = 0, 0
        click_handled = False
        
        # smart scroll variables
        edge_margin = 80
        scroll_speed = 150

        while self.running:
            success, img = cap.read()
            if not success:
                time.sleep(0.01)
                continue

            # mirror feed to feel natural
            img = cv2.flip(img, 1) 
            
            # process hands
            img = self.detector.find_hands(img, draw=True, color=self.skeleton_color)
            lm_list, bbox = self.detector.find_position(img, draw=False)

            # frame boundary for precise movement
            cv2.rectangle(img, (self.frame_reduc, self.frame_reduc), 
                          (self.w_cam - self.frame_reduc, self.h_cam - self.frame_reduc), 
                          (200, 200, 200), 1)

            if len(lm_list) != 0:
                x1, y1 = lm_list[8][1:]   # index
                x2, y2 = lm_list[12][1:]  # middle
                
                fingers = self.detector.fingers_up()

                # state 1: movement & scrolling (index up, middle down)
                if fingers[1] == 1 and fingers[2] == 0:
                    click_handled = False 
                    
                    if self.enable_movement:
                        x3 = np.interp(x1, (self.frame_reduc, self.w_cam - self.frame_reduc), (0, self.w_scr))
                        y3 = np.interp(y1, (self.frame_reduc, self.h_cam - self.frame_reduc), (0, self.h_scr))

                        cru_loc_x = pre_loc_x + (x3 - pre_loc_x) / self.smoothen
                        cru_loc_y = pre_loc_y + (y3 - pre_loc_y) / self.smoothen

                        move_x = max(0, min(int(cru_loc_x), int(self.w_scr - 1)))
                        move_y = max(0, min(int(cru_loc_y), int(self.h_scr - 1)))

                        try:
                            autopy.mouse.move(move_x, move_y)
                        except ValueError:
                            pass

                        pre_loc_x, pre_loc_y = cru_loc_x, cru_loc_y
                        
                        # smart edge scrolling
                        if self.enable_scroll:
                            if move_y <= edge_margin:
                                pyautogui.scroll(scroll_speed)
                            elif move_y >= self.h_scr - edge_margin:
                                pyautogui.scroll(-scroll_speed)

                        cv2.circle(img, (x1, y1), 12, self.skeleton_color, cv2.FILLED) 

                # state 2: clicking (index & middle pinched together)
                elif fingers[1] == 1 and fingers[2] == 1:
                    length, img, line_info = self.detector.find_distance(8, 12, img, draw=False)
                    
                    if length < 40:
                        cx, cy = line_info[4], line_info[5]
                        cv2.circle(img, (cx, cy), 12, (0, 255, 0), cv2.FILLED)
                        
                        if self.enable_click and not click_handled:
                            autopy.mouse.click()
                            click_handled = True
                    else:
                        click_handled = False
                else:
                    click_handled = False

            c_time = time.time()
            fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
            latency = (c_time - p_time) * 1000
            p_time = c_time

            self.stats_signal.emit(int(fps), latency)

            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.change_pixmap_signal.emit(qt_img)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        self.setFixedSize(400, 300)
        self.setStyleSheet("background-color: #0F171E; border: 2px solid #00E5FF; border-radius: 10px;")
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.logo_label = QLabel()
        # USE RESOURCE PATH FIX FOR SPLASH SCREEN LOGO
        pixmap = QPixmap(resource_path("logo.png")).scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.logo_label.setPixmap(pixmap)
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.logo_label.setStyleSheet("border: none;")
        
        self.text_label = QLabel("Built by Abdullah Alotaibi")
        self.text_label.setStyleSheet("color: #00E5FF; font-size: 15px; font-weight: bold; font-family: 'Times New Roman'; border: none; margin-top: 15px;")
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(self.logo_label)
        layout.addWidget(self.text_label)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Oculus Gesture Mouse")
        # USE RESOURCE PATH FIX FOR TASKBAR ICON
        self.setWindowIcon(QIcon(resource_path("logo.png")))
        self.setMinimumSize(1000, 650)
        
        self.is_dark_mode = True
        
        self.thread = TrackingThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.stats_signal.connect(self.update_stats)

        self.init_ui()
        self.apply_theme()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(330)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(25, 25, 25, 25)
        sidebar_layout.setSpacing(20)

        header_layout = QHBoxLayout()
        header_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        self.title_label = QLabel("OCULUS GESTURE")
        self.title_label.setFont(QFont("Times New Roman", 15, QFont.Weight.Bold))
        
        self.btn_theme = QPushButton("☀️")
        self.btn_theme.setFixedSize(35, 35)
        self.btn_theme.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_theme.clicked.connect(self.toggle_theme)
        
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_theme)
        
        sidebar_layout.addLayout(header_layout)

        self.btn_start = QPushButton("Start Tracking")
        self.btn_start.setFixedHeight(45)
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.setFont(QFont("Times New Roman", 12, QFont.Weight.Bold))
        self.btn_start.clicked.connect(self.start_tracking)

        self.btn_stop = QPushButton("Stop Tracking")
        self.btn_stop.setFixedHeight(45)
        self.btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop.setFont(QFont("Times New Roman", 12, QFont.Weight.Bold))
        self.btn_stop.clicked.connect(self.stop_tracking)
        self.btn_stop.setEnabled(False)

        sidebar_layout.addWidget(self.btn_start)
        sidebar_layout.addWidget(self.btn_stop)

        self.lbl_calib = QLabel("CALIBRATION")
        self.lbl_calib.setFont(QFont("Times New Roman", 11, QFont.Weight.Bold))
        sidebar_layout.addWidget(self.lbl_calib)

        self.slider_smooth = self.create_slider("Cursor Smoothness", 1, 15, self.thread.smoothen, sidebar_layout)
        self.slider_reduc = self.create_slider("Frame Reduction", 50, 200, self.thread.frame_reduc, sidebar_layout)

        self.lbl_gestures = QLabel("GESTURES")
        self.lbl_gestures.setFont(QFont("Times New Roman", 11, QFont.Weight.Bold))
        sidebar_layout.addWidget(self.lbl_gestures)

        self.toggle_move = QCheckBox("Cursor Movement")
        self.toggle_move.setChecked(True)
        self.toggle_move.setFont(QFont("Times New Roman", 11))
        self.toggle_move.stateChanged.connect(self.update_toggles)
        
        self.toggle_click = QCheckBox("Left Click (Pinch)")
        self.toggle_click.setChecked(True)
        self.toggle_click.setFont(QFont("Times New Roman", 11))
        self.toggle_click.stateChanged.connect(self.update_toggles)
        
        self.toggle_scroll = QCheckBox("Smart Edge Scroll")
        self.toggle_scroll.setChecked(True)
        self.toggle_scroll.setFont(QFont("Times New Roman", 11))
        self.toggle_scroll.stateChanged.connect(self.update_toggles)

        sidebar_layout.addWidget(self.toggle_move)
        sidebar_layout.addWidget(self.toggle_click)
        sidebar_layout.addWidget(self.toggle_scroll)
        sidebar_layout.addStretch()

        self.main_area = QFrame()
        main_area_layout = QVBoxLayout(self.main_area)
        main_area_layout.setContentsMargins(30, 30, 30, 30)

        self.video_label = QLabel("Camera Feed Offline")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFont(QFont("Times New Roman", 16))
        
        status_layout = QHBoxLayout()
        self.lbl_fps = QLabel("FPS: 0")
        self.lbl_latency = QLabel("Latency: 0.0 ms")
        self.lbl_fps.setFont(QFont("Times New Roman", 11, QFont.Weight.Bold))
        self.lbl_latency.setFont(QFont("Times New Roman", 11, QFont.Weight.Bold))
        
        self.lbl_github = QLabel('<a href="https://github.com/ASR181">GitHub: ASR181</a>')
        self.lbl_github.setFont(QFont("Times New Roman", 11, QFont.Weight.Bold))
        self.lbl_github.setOpenExternalLinks(True)
        
        status_layout.addWidget(self.lbl_fps)
        status_layout.addSpacing(15)
        status_layout.addWidget(self.lbl_latency)
        status_layout.addStretch()
        status_layout.addWidget(self.lbl_github)

        main_area_layout.addWidget(self.video_label, stretch=1)
        main_area_layout.addLayout(status_layout)

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.main_area, stretch=1)

    def create_slider(self, text, min_val, max_val, init_val, layout):
        wrapper = QVBoxLayout()
        lbl = QLabel(text)
        lbl.setFont(QFont("Times New Roman", 10))
        
        slider_layout = QHBoxLayout()
        lbl_l = QLabel("L")
        lbl_l.setFont(QFont("Times New Roman", 9, QFont.Weight.Bold))
        
        lbl_h = QLabel("H")
        lbl_h.setFont(QFont("Times New Roman", 9, QFont.Weight.Bold))
        
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(init_val)
        slider.setCursor(Qt.CursorShape.PointingHandCursor)
        slider.valueChanged.connect(self.update_calibration)
        
        slider_layout.addWidget(lbl_l)
        slider_layout.addWidget(slider)
        slider_layout.addWidget(lbl_h)
        
        wrapper.addWidget(lbl)
        wrapper.addLayout(slider_layout)
        layout.addLayout(wrapper)
        
        return slider

    def apply_theme(self):
        if self.is_dark_mode:
            bg_color = "#0F171E"
            panel_color = "#16202A"
            text_main = "#E0E0E0"
            accent_primary = "#00E5FF" 
            accent_btn = "#00B3CC"
            
            self.thread.skeleton_color = (255, 229, 0) 
            self.btn_theme.setText("☀️")
            self.btn_theme.setStyleSheet("background: transparent; border: none; font-size: 18px;")
            self.video_label.setStyleSheet("background-color: #0A1016; border: 2px solid #23303D; border-radius: 10px; color: #506070;")
            
            check_style = f"""
                QCheckBox {{ color: {text_main}; border: none; }}
                QCheckBox::indicator {{ width: 40px; height: 20px; border-radius: 10px; }}
                QCheckBox::indicator:unchecked {{ background-color: #2A3B4C; }}
                QCheckBox::indicator:checked {{ background-color: {accent_primary}; }}
            """
            slider_style = f"""
                QSlider::groove:horizontal {{ height: 4px; background: #23303D; border-radius: 2px; }}
                QSlider::handle:horizontal {{ background: {accent_primary}; width: 14px; margin: -5px 0; border-radius: 7px; }}
            """
        else:
            bg_color = "#F0F4F8"
            panel_color = "#FFFFFF"
            text_main = "#1A202C"
            accent_primary = "#0066FF" 
            accent_btn = "#005CE6"
            
            self.thread.skeleton_color = (255, 102, 0)
            self.btn_theme.setText("🌙")
            self.btn_theme.setStyleSheet("background: transparent; border: none; font-size: 18px;")
            self.video_label.setStyleSheet("background-color: #E2E8F0; border: 2px solid #CBD5E1; border-radius: 10px; color: #64748B;")
            
            check_style = f"""
                QCheckBox {{ color: {text_main}; border: none; }}
                QCheckBox::indicator {{ width: 40px; height: 20px; border-radius: 10px; }}
                QCheckBox::indicator:unchecked {{ background-color: #CBD5E1; }}
                QCheckBox::indicator:checked {{ background-color: {accent_primary}; }}
            """
            slider_style = f"""
                QSlider::groove:horizontal {{ height: 4px; background: #CBD5E1; border-radius: 2px; }}
                QSlider::handle:horizontal {{ background: {accent_primary}; width: 14px; margin: -5px 0; border-radius: 7px; }}
            """

        self.setStyleSheet(f"background-color: {bg_color}; color: {text_main}; font-family: 'Times New Roman';")
        self.sidebar.setStyleSheet(f"background-color: {panel_color}; border-right: 1px solid {'#23303D' if self.is_dark_mode else '#E2E8F0'};")
        self.main_area.setStyleSheet("border: none;")
        
        self.title_label.setStyleSheet(f"color: {accent_primary}; border: none;")
        self.lbl_calib.setStyleSheet(f"color: {accent_primary}; border: none;")
        self.lbl_gestures.setStyleSheet(f"color: {accent_primary}; border: none;")
        
        self.lbl_fps.setStyleSheet(f"color: {accent_primary};")
        self.lbl_latency.setStyleSheet(f"color: {accent_primary};")
        
        self.lbl_github.setStyleSheet(f"""
            QLabel {{ color: {accent_primary}; }}
            QLabel a {{ color: {accent_primary}; text-decoration: none; }}
            QLabel a:hover {{ text-decoration: underline; }}
        """)

        for checkbox in[self.toggle_move, self.toggle_click, self.toggle_scroll]:
            checkbox.setStyleSheet(check_style)
            
        for slider in[self.slider_smooth, self.slider_reduc]:
            slider.setStyleSheet(slider_style)

        if not self.thread.running:
            self.btn_start.setStyleSheet(f"background-color: {accent_btn}; color: white; border-radius: 5px; font-weight: bold; border: none;")
            self.btn_stop.setStyleSheet(f"background-color: {'#2A3B4C' if self.is_dark_mode else '#94A3B8'}; color: white; border-radius: 5px; font-weight: bold; border: none;")
        else:
            self.btn_start.setStyleSheet(f"background-color: {'#15404A' if self.is_dark_mode else '#94A3B8'}; color: {'#507080' if self.is_dark_mode else '#FFFFFF'}; border-radius: 5px; font-weight: bold; border: none;")
            self.btn_stop.setStyleSheet(f"background-color: #E53935; color: white; border-radius: 5px; font-weight: bold; border: none;")

    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        self.apply_theme()

    def start_tracking(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.apply_theme() 
        self.thread.start(QThread.Priority.HighestPriority)

    def stop_tracking(self):
        self.btn_stop.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.thread.stop()
        self.apply_theme() 
        
        self.video_label.clear()
        self.video_label.setText("Camera Feed Offline")
        self.lbl_fps.setText("FPS: 0")
        self.lbl_latency.setText("Latency: 0.0 ms")

    def update_image(self, qt_img):
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    def update_stats(self, fps, latency):
        self.lbl_fps.setText(f"FPS: {fps}")
        self.lbl_latency.setText(f"Latency: {latency:.1f} ms")

    def update_calibration(self):
        self.thread.smoothen = self.slider_smooth.value()
        self.thread.frame_reduc = self.slider_reduc.value()

    def update_toggles(self):
        self.thread.enable_movement = self.toggle_move.isChecked()
        self.thread.enable_click = self.toggle_click.isChecked()
        self.thread.enable_scroll = self.toggle_scroll.isChecked()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

def launch_app(splash, window):
    splash.close()
    window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # USE RESOURCE PATH FIX
    app.setWindowIcon(QIcon(resource_path("logo.png")))
    
    splash = SplashScreen()
    splash.show()
    
    window = MainWindow()
    
    QTimer.singleShot(2500, lambda: launch_app(splash, window))
    
    sys.exit(app.exec())