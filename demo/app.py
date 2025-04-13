import sys
import os
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QLineEdit, 
    QComboBox, QHBoxLayout, QMessageBox, QFileDialog, QSlider
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from time import time, sleep
from threading import Lock
from resample_queue import ResampleQueue
from queue import Queue, Empty
from collections import deque
from random import randint
from model_onnx import YoloONNX, get_providers
import numpy as np


class VideoThread(QThread):
    # change_pixmap_signal = pyqtSignal(QImage)
    position_signal = pyqtSignal(int)

    def __init__(self, stream, source, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._run_flag = True
        self.source = source
        self.deque = stream
        self.stream_lock = Lock()
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.frame = None

    def run(self):
        self.cap = cv2.VideoCapture(self.source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        delay = 1.0 / self.fps

        #preallocate arrays
        self.frame = np.zeros(shape=(cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH, 3), dtype=np.uint8)
        self.rgb_frame = np.zeros(shape=(cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH, 3), dtype=np.uint8)

        cnt = 0
        while self._run_flag:
            with self.stream_lock:
                ret, frame = self.cap.read(self.frame)
                if ret:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, self.rgb_frame)
                    self.deque.append(image_rgb)

            if cnt % (self.fps // 2) == 0:
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.position_signal.emit(int(current_frame / self.total_frames * 1000))
            sleep(delay)

        self.cap.release()


    def stop(self):
        self._run_flag = False
        
    def restart(self):
        self._run_flag = True


    def set_position(self, new_pos):
        if self.cap is not None:
            with self.stream_lock:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, (new_pos / 1000 * self.total_frames))

    def duration(self):
        #in seconds
        return (self.total_frames // self.fps)

class OnnxRunner(QThread):

    def __init__(self, stream, output_stream, onnx_model=None, device_mode='cpu', confidence=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not onnx_model:
            raise ValueError('Provide parameter onnx_model as model path')

        self.stop = False
        self.input_stream = stream
        self.output = output_stream
        self.frames_to_process = []
        if device_mode == 'cpu':
            self.batch = max(4, os.cpu_count() - 4) #остальная система тоже хочет ресурсов
        else:
            self.batch = 8  #8images batch for gpu
        self.model = YoloONNX(onnx_model, device=device_mode, batch=self.batch, confidence=confidence)

        self.fps_q = deque(maxlen=10)

    def fps(self):
        fps = '0' if not len(self.fps_q) else f'{1 / (sum(self.fps_q) / len(self.fps_q)):.2f}'
        return fps

    def run(self):

        while not self.stop:

            if self.input_stream.qsize() >= self.batch:
                self.frames_to_process = self.input_stream.get_batch()
                
                start = time()
                frames = self.model(self.frames_to_process)
                self.fps_q.append((time() - start) / self.batch)

                for frame in frames:
                    self.output.put_nowait(frame)
                self.frames_to_process.clear()
            elif self.input_stream.qsize() == 0:
                sleep(0.1)
            else:
                sleep(0.02)



class Renderer(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    def __init__(self, frame_queue, width, height, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.source = frame_queue
        self.width = width
        self.height = height

        self.update_interval = 30

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.setInterval(self.update_interval)
        self.timer.start()  
        self.fifo_fill_level = 20
        self.cnt = 0
        self.frame_fps = deque(maxlen=10)

    def reset_buf(self):
        self.source.queue.clear()

    def fps(self):
        if len(self.frame_fps):
            return sum(self.frame_fps) / len(self.frame_fps)
        
        return 0

    def update_frame(self):
        try:
            frame = self.source.get_nowait()
             
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(self.width , self.height)
            self.change_pixmap_signal.emit(p)

        except Empty:
            return

        mean_fill = self.source.qsize()
        self.cnt += 1
        
        if self.cnt % 12 == 0:
            error = mean_fill - self.fifo_fill_level
            err = (error * 0.01)
            self.update_interval *= (1 - err)
            self.update_interval = int(self.update_interval)

            if self.update_interval < 5:
                self.update_interval = 5

            if self.update_interval > 120:
                self.update_interval = 120

            self.frame_fps.append(1000 / self.update_interval)

            self.timer.setInterval(self.update_interval)

        sleep(0.01)


    def run(self):
        while True:
            sleep(0.1)


class cQLineEdit(QLineEdit):
    clicked = pyqtSignal()

    def __init__(self, widget):
        super().__init__(widget)

    def mousePressEvent(self, QMouseEvent):
        self.clicked.emit()


class VideoWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Onnx real view runner")
        self.setGeometry(100, 100, 1920 // 2, 1080 // 2)
        self.width = 800
        self.height = 600

        self.timer = QTimer()

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.file_label = QLabel(self)
        self.file_label.setText('Видеофайл')
        self.video_source = cQLineEdit(self.central_widget)
        self.video_source.clicked.connect(self.choose_sorce)

        self.rtsp_label = QLabel(self)
        self.rtsp_label.setText('RTSP ссылка')
        self.rtsp_source = cQLineEdit(self.central_widget)
        self.rtsp_source.setText('Вставьте RTSP ссылку')
        self.rtsp_source.clicked.connect(self.clear_rtsp)

        self.models_label = QLabel(self)
        self.models_label.setText('Модели')
        self.model_folder = cQLineEdit(self.central_widget)
        self.model_folder.setText('.')


        self.device_label = QLabel(self)
        self.model_file = QComboBox()

        self.model_file.addItems([i for i in os.listdir(self.model_folder.text()) if i.endswith('.onnx')])

        self.conf_value = QLabel(self)
        self.conf_value.setText('Confidence %')
        
        self.conf = QSlider(self)
        self.conf.setOrientation(Qt.Horizontal)
        self.conf.setMaximum(100)
        self.conf.setValue(50)
        self.conf.setMinimum(10)
        self.conf.setTickInterval(1)

        self.conf.valueChanged.connect(lambda value: self.conf_value.setText(f'Confidence {self.conf.value()} %'))


        self.device_label.setText('Устройство (CUDA 11.8)')
        self.perf_mode = QComboBox()
        self.perf_mode.addItems(get_providers())

        self.start_button = QPushButton('Старт')
        self.start_button.clicked.connect(self.start_video)
        self.stop_button = QPushButton('Стоп')
        self.stop_button.clicked.connect(self.stop_video)

        self.perf_label = QLabel(self)
        self.perf_label.setText('0 FPS')

        self.video_fps = QLabel(self)
        self.video_fps.setText('0 FPS')

        control_layout = QVBoxLayout()
        control_layout.addWidget(self.file_label)
        control_layout.addWidget(self.video_source)
        control_layout.addWidget(self.rtsp_label)
        control_layout.addWidget(self.rtsp_source)
        control_layout.addWidget(self.models_label)
        control_layout.addWidget(self.model_folder)
        control_layout.addWidget(self.model_file)
        control_layout.addWidget(self.device_label)
        control_layout.addWidget(self.perf_mode)
        control_layout.addWidget(self.conf_value)

        control_layout.addWidget(self.conf)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.perf_label)
        control_layout.addWidget(self.video_fps)

        self.time_slider = QSlider(Qt.Horizontal, self.central_widget)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(1000)
        self.time_slider.valueChanged.connect(self.on_time_changed)

        self.time_label = QLabel("00:00 / 00:00", self.central_widget)
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("QLabel { padding: 2px; }")
        self.time_label.setFixedHeight(20)


#------------------------------------------------
        control_widget = QWidget()
        control_widget.setLayout(control_layout)
        control_widget.setMaximumWidth(200)

        video_layout = QVBoxLayout()
        video_layout.addWidget(self.image_label)

        time_layout = QHBoxLayout()
        time_layout.addWidget(self.time_slider)
        time_layout.addWidget(self.time_label)

        video_layout.addLayout(time_layout)

        central_layout = QHBoxLayout()
        central_layout.addLayout(video_layout)
        central_layout.addWidget(control_widget)

        # Устанавливаем растяжку для метки с видео
        central_layout.setStretchFactor(video_layout, 1)
        central_layout.setStretchFactor(control_widget, 0)

        self.central_widget.setLayout(central_layout)

        self.qt_img = None

        self.frame_queue = ResampleQueue()
        self.render_source = Queue()
        self.rtsp_thread = None
        self.onnx_thread = None


    def update_slider_position(self, pos):
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(pos)
        self.time_slider.blockSignals(False)

        total_duration = self.rtsp_thread.duration()

        total_secs = total_duration * pos / 1000
        mins = int(total_secs // 60)
        secs = int(total_secs % 60)

        total_mins = int(total_duration // 60)
        total_secs = int(total_duration % 60)

        self.time_label.setText(f'{mins}:{str(secs).zfill(2)} / {total_mins}:{total_secs}') 

    def on_time_changed(self, pos):
        self.rtsp_thread.set_position(pos)
        self.render.reset_buf()

    def start_video(self):
        
        if self.rtsp_thread is not None:
            self.rtsp_thread.restart()
            return
        
        if not self.video_source.text():
            QMessageBox.warning(self, "Warning", "Нужно выбрать файл")
            return

        if not self.model_file.currentText():
            QMessageBox.warning(self, "Warning", "Нужно выбрать файл модели")
            return


        onnx_cfg = {}
        onnx_cfg['onnx_model'] = f'{self.model_folder.text()}/{self.model_file.currentText()}'
        onnx_cfg['device_mode'] = self.perf_mode.currentText()
        onnx_cfg['confidence'] = self.conf.value() / 100


        self.rtsp_thread = VideoThread(self.frame_queue, self.video_source.text())
        self.rtsp_thread.position_signal.connect(self.update_slider_position)
        self.rtsp_thread.start()

        self.onnx_thread = OnnxRunner(self.frame_queue, self.render_source, **onnx_cfg)
        self.onnx_thread.start()

        self.render = Renderer(self.render_source, 1920 // 2, 1080 // 2)
        self.render.change_pixmap_signal.connect(self.update_image)
        
        self.render.start()

        self.timer.timeout.connect(self.print_fps)
        self.timer.start(1000)  #время обновления FPS


    def stop_video(self):
        if self.rtsp_thread is not None and self.rtsp_thread.isRunning():
            self.rtsp_thread.stop()
            self.clear_rtsp()

    def clear_rtsp(self):
        self.rtsp_source.clear()   

    def choose_model_folder(self):
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", options=options)
        if folder_path:
            self.model_folder.setText(folder_path)
            self.model_file.addItems([i for i in os.listdir(folder_path) if i.endswith('.onnx')])

    def choose_sorce(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)

        if dlg.exec_():
            if len(dlg.selectedFiles()):
                filepath = dlg.selectedFiles()[0]
                self.video_source.setText(filepath)

    def print_fps(self):
        if self.onnx_thread and self.render:
            self.perf_label.setText(f'NN: {self.onnx_thread.fps()} FPS')
            self.video_fps.setText(f'R: {self.render.fps():.2f} FPS')

            
    def update_image(self, cv_img):
        self.qt_img = QPixmap.fromImage(cv_img)
        self.image_label.setPixmap(self.qt_img)
        original_size = self.qt_img.size()

        self.scale_image()
        self.setMinimumSize(original_size.width(), original_size.height())

    def scale_image(self):
        if self.qt_img is not None:
            scaled_pixmap = self.qt_img.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        self.scale_image()
        super().resizeEvent(event)

    def closeEvent(self, event):
        if self.rtsp_thread:
            self.rtsp_thread.stop()
            self.rtsp_thread.quit()
            self.rtsp_thread.wait()
        
        if self.onnx_thread:
            self.onnx_thread.stop = True
            self.onnx_thread.quit()
            self.onnx_thread.wait()

        event.accept()

if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_())