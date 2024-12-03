import sys
import os
import logging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QTextEdit, QGridLayout, QVBoxLayout, QWidget, QComboBox, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QFont, QPalette, QBrush, QImage
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from ultralytics import RTDETR
import warnings
import io
import contextlib
import cv2

warnings.filterwarnings('ignore')

class RedirectOutput(logging.Handler):
    """将日志重定向到QTextEdit"""
    def __init__(self, log_widget):
        super().__init__()
        self.log_widget = log_widget

    def emit(self, record):
        msg = self.format(record)
        self.log_widget.moveCursor(self.log_widget.textCursor().End)  # 滚动到最后一行
        self.log_widget.insertPlainText(msg + '\n')


class CameraThread(QThread):
    """线程类，用于处理摄像头实时画面"""
    update_frame = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.capture = cv2.VideoCapture(0)
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, c = rgb_frame.shape
                bytes_per_line = c * w
                qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.update_frame.emit(qt_img)
            cv2.waitKey(1)

    def stop(self):
        """停止摄像头线程"""
        self.running = False
        self.capture.release()
        self.wait()


class DetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于改进RT-DETR的轻量化遥感图像目标识别系统")
        self.setGeometry(200, 200, 1200, 900)

        # 初始化参数
        self.test_folder = None
        self.test_image = None
        self.weights_path = None
        self.save_folder = None
        self.detection_mode = "文件夹"  # 默认检测模式
        self.camera_thread = None  # 摄像头线程
        self.captured_frame = None  # 捕获的图片

        # 初始化UI
        self.init_ui()

        # 配置日志
        self.configure_logging()

    def configure_logging(self):
        """设置日志系统，捕获所有输出到 QTextEdit"""
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        log_handler = RedirectOutput(self.log_output)
        log_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(log_handler)

        # 过滤不相关的日志
        logging.getLogger("requests").setLevel(logging.WARNING)  # 过滤 requests 库的日志
        logging.getLogger("urllib3").setLevel(logging.WARNING)  # 过滤 urllib3 库的日志
        logging.getLogger("google").setLevel(logging.WARNING)  # 过滤 google 的日志

        sys.stdout = self  # 重定向标准输出
        sys.stderr = self  # 重定向标准错误

    def init_ui(self):
        # 设置遥感背景图片
        self.set_background("background.jpg")

        # 主界面布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QGridLayout()
        self.central_widget.setLayout(self.layout)

        # 添加标题
        self.title_label = QLabel("基于改进RT-DETR的轻量化遥感图像目标识别系统")
        self.title_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title_label, 0, 0, 1, 2)

        # 第一行：权重文件选择
        self.weights_label = QLabel("选择权重文件：")
        self.layout.addWidget(self.weights_label, 1, 0)

        self.weights_btn = QPushButton("选择文件")
        self.weights_btn.clicked.connect(self.select_weights)
        self.layout.addWidget(self.weights_btn, 1, 1)

        # 第二行：检测模式选择
        self.mode_label = QLabel("选择检测模式：")
        self.layout.addWidget(self.mode_label, 2, 0)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["文件夹", "单张图片", "摄像头检测"])  # 添加摄像头检测模式
        self.mode_combo.setStyleSheet("""
            QComboBox {
                text-align: center;
            }
            QComboBox QAbstractItemView {
                text-align: center;
            }
        """)
        self.mode_combo.currentTextChanged.connect(self.change_mode)
        self.layout.addWidget(self.mode_combo, 2, 1)

        # 第三行：测试文件/文件夹选择
        self.test_label = QLabel("选择测试数据：")
        self.layout.addWidget(self.test_label, 3, 0)

        self.test_btn = QPushButton("选择文件夹" if self.detection_mode == "文件夹" else "选择文件")
        self.test_btn.clicked.connect(self.select_test_data)
        self.layout.addWidget(self.test_btn, 3, 1)

        # 第四行：保存路径选择
        self.save_label = QLabel("选择保存路径：")
        self.layout.addWidget(self.save_label, 4, 0)

        self.save_btn = QPushButton("选择文件夹")
        self.save_btn.clicked.connect(self.select_save_folder)
        self.layout.addWidget(self.save_btn, 4, 1)

        # 第五行：运行检测按钮
        self.run_btn = QPushButton("运行检测")
        self.run_btn.clicked.connect(self.run_detection)
        self.layout.addWidget(self.run_btn, 5, 0, 1, 2)

        # 第六行：日志输出
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.layout.addWidget(self.log_output, 6, 0, 1, 2)

        # 第七行：检测结果显示（左右分栏）
        self.result_layout = QHBoxLayout()

        # 左侧：原始图像显示
        self.result_original_layout = QVBoxLayout()
        self.result_original_title = QLabel("原始图像")  # 只添加标题
        self.result_original_layout.addWidget(self.result_original_title, alignment=Qt.AlignCenter)
        self.result_original = QLabel()  # 清空文本
        self.result_original.setAlignment(Qt.AlignCenter)
        self.result_original.setFixedSize(400, 400)
        self.result_original_layout.addWidget(self.result_original)
        self.result_layout.addLayout(self.result_original_layout)

        # 右侧：检测结果显示
        self.result_detected_layout = QVBoxLayout()
        self.result_detected_title = QLabel("检测结果")  # 只添加标题
        self.result_detected_layout.addWidget(self.result_detected_title, alignment=Qt.AlignCenter)
        self.result_detected = QLabel()  # 清空文本
        self.result_detected.setAlignment(Qt.AlignCenter)
        self.result_detected.setFixedSize(400, 400)
        self.result_detected_layout.addWidget(self.result_detected)
        self.result_layout.addLayout(self.result_detected_layout)

        self.layout.addLayout(self.result_layout, 8, 0, 1, 2)

        # 新增：摄像头相关UI
        self.capture_btn = QPushButton("拍摄")
        self.capture_btn.clicked.connect(self.capture_image)
        self.layout.addWidget(self.capture_btn, 9, 0)

        self.retake_btn = QPushButton("重拍")
        self.retake_btn.clicked.connect(self.retake_image)
        self.layout.addWidget(self.retake_btn, 9, 1)

    def set_background(self, image_path):
        if os.path.exists(image_path):
            background = QImage(image_path)
            palette = QPalette()
            palette.setBrush(QPalette.Window, QBrush(background))
            self.setPalette(palette)

    def select_weights(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择权重文件", "", "PyTorch Files (*.pt)")
        if file_path:
            self.weights_path = file_path
            self.logger.info(f"已选择权重文件：{file_path}")

    def change_mode(self, mode):
        self.detection_mode = mode
        self.test_btn.setText("选择文件夹" if mode == "文件夹" else "选择文件")
        self.result_original.clear()
        self.result_detected.clear()
        self.logger.info(f"检测模式已切换为：{mode}")

        # 启动或停止摄像头线程
        # 启动或停止摄像头线程
        if mode == "摄像头检测":
            if self.camera_thread is None:
                self.camera_thread = CameraThread()
                self.camera_thread.update_frame.connect(self.update_camera_frame)
                self.camera_thread.start()
        else:
            if self.camera_thread is not None:
                self.camera_thread.stop()
                self.camera_thread = None

    def update_camera_frame(self, frame):
        """更新实时摄像头画面"""
        if self.captured_frame is None:  # 仅显示实时画面时
            self.result_original.setPixmap(
                QPixmap.fromImage(frame).scaled(self.result_original.size(), Qt.KeepAspectRatio))

    def capture_image(self):
        """捕获当前摄像头画面"""
        self.captured_frame = self.result_original.pixmap()  # 捕获图像
        self.result_original.setPixmap(self.captured_frame)  # 显示捕获的图像
        self.logger.info("已捕获图像")

    def retake_image(self):
        """重新拍摄图像"""
        self.captured_frame = None
        self.logger.info("重新拍摄图像")
        self.result_original.clear()  # 清空显示

    def select_test_data(self):
        if self.detection_mode == "文件夹":
            folder_path = QFileDialog.getExistingDirectory(self, "选择测试文件夹")
            if folder_path:
                self.test_folder = folder_path
                self.logger.info(f"已选择测试文件夹：{folder_path}")

            # 统计文件夹内图片的数量
            supported_formats = (".jpg", ".jpeg", ".png", ".bmp")  # 支持的图片格式
            image_files = [f for f in os.listdir(self.test_folder) if f.lower().endswith(supported_formats)]
            total_images = len(image_files)
            self.logger.info(f"所选文件夹包含图片数量: {total_images} 张")

        else:
            file_path, _ = QFileDialog.getOpenFileName(self, "选择测试图片", "", "Images (*.jpg *.png)")
            if file_path:
                self.test_image = file_path
                self.display_image(file_path, self.result_original)
                self.logger.info(f"已选择测试图片：{file_path}")

    def select_save_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择保存文件夹")
        if folder_path:
            self.save_folder = folder_path
            self.logger.info(f"已设置保存路径：{folder_path}")

    def run_detection(self):
        if not self.weights_path:
            self.logger.info("请先选择权重文件！")
            return

        if not self.save_folder:
            self.logger.info("请先设置保存路径！")
            return

        try:
            # 执行检测操作并捕获标准输出
            model = RTDETR(self.weights_path)  # 初始化模型
            if self.detection_mode == "单张图片" and self.test_image:
                with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                    results = model.predict(source=self.test_image, project=self.save_folder, save=True)
                    for result in results:
                        # 获取并格式化信息
                        boxes = result.boxes.xyxy
                        names = result.names
                        orig_shape = result.orig_shape
                        speed = result.speed

                        # 格式化输出
                        output = (
                            f"\n{'=' * 50}\n"
                            f"【检测结果】\n"
                            f"图片路径: {self.test_image}\n"
                            f"原始图像尺寸: {orig_shape}\n"
                            f"检测框(Boxes)位置:\n{boxes}\n"
                            f"类别名称列表:" 
                            f"   \n{names}\n"
                            f"检测速度: \n"
                            f"   预处理时间: {speed['preprocess']:.2f}ms\n"
                            f"   推理时间: {speed['inference']:.2f}ms\n"
                            f"   后处理时间: {speed['postprocess']:.2f}ms\n"
                            f"{'=' * 50}\n"
                        )
                    self.logger.info(output)  # 打印到日志区域

                latest_result_path = self.get_latest_exp_path(self.save_folder)
                detected_image_path = self.get_detected_image_path(latest_result_path, self.test_image)
                if detected_image_path:
                    self.display_image(self.test_image, self.result_original)
                    self.display_image(detected_image_path, self.result_detected)
                    self.logger.info(f"检测完成！检测结果已保存至：{detected_image_path}")

            elif self.detection_mode == "文件夹" and self.test_folder:
                with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                    results = model.predict(source=self.test_folder, project=self.save_folder, save=True)
                    for result in results:
                        boxes = result.boxes.xyxy
                        names = result.names
                        orig_shape = result.orig_shape
                        speed = result.speed

                        # 格式化输出
                        output = (
                            f"\n{'=' * 50}\n"  
                            f"【检测结果】\n"
                            f"图片路径: {self.test_image}\n"
                            f"原始图像尺寸: {orig_shape}\n"
                            f"检测框(Boxes)位置:\n{boxes}\n"
                            f"类别名称列表:"
                            f"   \n{names}\n"
                            f"检测速度: \n"
                            f"   预处理时间: {speed['preprocess']:.2f}ms\n"
                            f"   推理时间: {speed['inference']:.2f}ms\n"
                            f"   后处理时间: {speed['postprocess']:.2f}ms\n"
                            f"{'=' * 50}\n"
                        )
                        self.logger.info(output)

                latest_result_path = self.get_latest_exp_path(self.save_folder)
                if latest_result_path:
                    self.logger.info(f"检测完成！检测结果文件夹路径为：{latest_result_path}")
                else:
                    self.logger.error("未能找到检测结果文件夹！")

            elif self.detection_mode == "摄像头检测" and self.captured_frame:
                # 保存拍摄的图像进行检测
                file_path = os.path.join(self.save_folder, "captured_image.jpg")
                self.captured_frame.save(file_path)
                self.test_image = file_path

                with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                    results = model.predict(source=self.test_image, project=self.save_folder, save=True)
                    for result in results:
                        # 获取并格式化信息
                        boxes = result.boxes.xyxy
                        names = result.names
                        orig_shape = result.orig_shape
                        speed = result.speed

                        # 格式化输出
                        output = (
                            f"\n{'=' * 50}\n"
                            f"【检测结果】\n"
                            f"图片路径: {self.test_image}\n"
                            f"原始图像尺寸: {orig_shape}\n"
                            f"检测框(Boxes)位置:\n{boxes}\n"
                            f"类别名称列表:"
                            f"   \n{names}\n"
                            f"检测速度: \n"
                            f"   预处理时间: {speed['preprocess']:.2f}ms\n"
                            f"   推理时间: {speed['inference']:.2f}ms\n"
                            f"   后处理时间: {speed['postprocess']:.2f}ms\n"
                            f"{'=' * 50}\n"
                        )
                    self.logger.info(output)

                latest_result_path = self.get_latest_exp_path(self.save_folder)
                detected_image_path = self.get_detected_image_path(latest_result_path, self.test_image)
                if detected_image_path:
                    self.display_image(self.test_image, self.result_original)
                    self.display_image(detected_image_path, self.result_detected)
                    self.logger.info(f"检测完成！检测结果已保存至：{detected_image_path}")

            else:
                self.logger.info("请先选择测试数据！")

        except Exception as e:
            self.logger.error(f"检测失败：{e}")

    def get_latest_exp_path(self, folder):
        """获取保存目录下最新的exp文件夹"""
        subfolders = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        return max(subfolders, key=os.path.getmtime)

    def get_detected_image_path(self, folder, original_image_path):
        """根据原始图片路径获取检测结果图片路径"""
        original_name = os.path.basename(original_image_path)
        return os.path.join(folder, original_name)

    def display_image(self, image_path, label_widget):
        """在指定的QLabel上显示图片"""
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(label_widget.width(), label_widget.height(), Qt.KeepAspectRatio)
        label_widget.setPixmap(pixmap)
        label_widget.setAlignment(Qt.AlignCenter)

    def write(self, text):
        """重定向sys.stdout的输出"""
        self.logger.info(text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DetectionApp()
    window.show()
    sys.exit(app.exec_())
