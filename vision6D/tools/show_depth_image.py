"""
@author:Hanagumori-B <yjp.y@foxmail.com>
@license: (C) Copyright.
@contact: yjp.y@foxmail.com
@software: Vision6D
@file: show_depth_image.py
@time: 2025-10-14 17:08
@desc: show a 16bit depth image in colormap
"""

import os
import pathlib
import sys

from PyQt5 import QtWidgets, QtCore, QtGui
import cv2 as cv
import numpy as np
from PyQt5.QtGui import QPixmap

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# noinspection PyUnresolvedReferences
from vision6D import STYLES_FILE
# noinspection PyUnresolvedReferences
from vision6D.path import ICON_PATH


class DepthImageWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        self.setWindowTitle('16bit Depth Image Viewer')
        self.setWindowIcon(QtGui.QIcon(str(ICON_PATH / 'logo.ico')))
        self.setMinimumSize(640, 500)

        self.setMenuBars()
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)

        self.image_path: pathlib.Path | None = None

        layout = QtWidgets.QGridLayout(self.centralWidget)
        self.imageLabel = ImageLabel()
        layout.addWidget(self.imageLabel, 0, 0)

        self.statusbar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusbar)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_scale)
        self.timer.start(1000)

    def setMenuBars(self):
        mainMenu = self.menuBar()
        mainMenu.addAction('Open', self.open_image)

        imageMenu = mainMenu.addMenu('Image')
        imageMenu.addAction('Center', self.set_center)
        imageMenu.addAction('Adapt', self.set_adapt)

        imageMenu.addAction('Next', self.next_image)
        imageMenu.addAction('Last', self.last_image)

    def open_image(self):
        image_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        # self.imageLabel.set_image(QPixmap.fromImage(QtGui.QImage(image_path)))
        if not image_path:
            return
        self.image_path = pathlib.Path(image_path)
        self.set_image(self.image_path)

    def set_image(self, image_path):
        depth_image_processor = DepthImageProcessor.read_16bit_image(image_path)
        d_img = depth_image_processor.apply_colormap()
        # print(d_img.shape)

        d_img = cv.cvtColor(d_img, cv.COLOR_BGR2RGB)
        d_img = np.ascontiguousarray(d_img)
        height, width, bytes_per_component = d_img.shape
        bytes_per_line = bytes_per_component * width
        q_img = QtGui.QImage(d_img.data.tobytes(), width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.imageLabel.set_image(QPixmap.fromImage(q_img))
        # self.imageLabel.set_image(QPixmap.fromImage(QtGui.QImage.fromData(depth_image_processor.apply_colormap())))

    def set_center(self):
        if not self.imageLabel.m_rectPixmap.isEmpty():
            self.imageLabel.center_image()
            self.imageLabel.update()

    def set_adapt(self):
        if not self.imageLabel.m_rectPixmap.isEmpty():
            self.imageLabel.adapt_scale()
            self.imageLabel.center_image()
            self.imageLabel.update()

    def update_scale(self):
        if not self.imageLabel.m_rectPixmap.isEmpty():
            self.statusbar.showMessage(f"scale: {self.imageLabel.m_scaleValue * 100:.02f}%")

    def next_image(self):
        if not self.imageLabel.m_rectPixmap.isEmpty() and self.image_path:
            images_in_dir = [d for d in self.image_path.parent.glob('*') if d.suffix in ('.png', '.jpg', '.bmp', '.jpeg')]
            index = images_in_dir.index(self.image_path)
            next_index = index + 1 if index + 1 < len(images_in_dir) else 0
            self.image_path = images_in_dir[next_index]
            self.set_image(self.image_path)

    def last_image(self):
        if not self.imageLabel.m_rectPixmap.isEmpty() and self.image_path:
            images_in_dir = [d for d in self.image_path.parent.glob('*') if d.suffix in ('.png', '.jpg', '.bmp', '.jpeg')]
            index = images_in_dir.index(self.image_path)
            next_index = index - 1 if index - 1 > 0 else len(images_in_dir) - 1
            self.image_path = images_in_dir[next_index]
            self.set_image(self.image_path)


class ImageLabel(QtWidgets.QLabel):
    # 缩放比例的最小值和最大值
    SCALE_MIN_VALUE = 0.05
    SCALE_MAX_VALUE = 10.0

    def __init__(self, parent=None):
        super(ImageLabel, self).__init__(parent)
        self.setMouseTracking(True)
        self.m_scaleValue = 1.0  # 缩放比例初始值
        self.m_rectPixmap = QtCore.QRectF()
        self.m_scalePixmap = QtCore.QRectF()
        self.m_drawPoint = QtCore.QPointF(0.0, 0.0)
        self.m_pressed = False
        self.m_lastPos = QtCore.QPoint()
        self.old_width = self.width()
        self.old_height = self.height()

    def set_image(self, img: QPixmap):
        self.setPixmap(img)
        self.adapt_scale()
        self.center_image()
        self.update()

    def center_image(self):
        scaled_width = self.pixmap().size().width() * self.m_scaleValue
        scaled_height = self.pixmap().size().height() * self.m_scaleValue
        center_x = (self.width() - scaled_width) / 2
        center_y = (self.height() - scaled_height) / 2
        self.m_drawPoint.setX(center_x)
        self.m_drawPoint.setY(center_y)

    def adapt_scale(self):
        w, h = self.size().width(), self.size().height()
        img_w, img_h = self.pixmap().size().width(), self.pixmap().size().height()
        self.m_scaleValue = float(min(w / img_w, h / img_h))
        self.m_rectPixmap = QtCore.QRectF(self.pixmap().rect())

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.m_pressed = True
            self.m_lastPos = event.pos()

    def mouseMoveEvent(self, event):
        if self.m_pressed:
            delta = event.pos() - self.m_lastPos
            self.m_lastPos = event.pos()
            self.m_drawPoint += delta
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.m_pressed = False

    def wheelEvent(self, event):
        if not self.m_rectPixmap.isEmpty():
            oldScale = self.m_scaleValue
            if event.angleDelta().y() > 0:
                self.m_scaleValue *= 1.1
            else:
                self.m_scaleValue *= 0.9
            if self.m_scaleValue > self.SCALE_MAX_VALUE:
                self.m_scaleValue = self.SCALE_MAX_VALUE
            if self.m_scaleValue < self.SCALE_MIN_VALUE:
                self.m_scaleValue = self.SCALE_MIN_VALUE
            # 鼠标位在图片范围内，则以鼠标位置为中心缩放
            if self.m_scalePixmap.contains(event.pos()):
                x = self.m_drawPoint.x() - (event.pos().x() - self.m_drawPoint.x()) * (self.m_scaleValue / oldScale - 1)
                y = self.m_drawPoint.y() - (event.pos().y() - self.m_drawPoint.y()) * (self.m_scaleValue / oldScale - 1)
                self.m_drawPoint = QtCore.QPointF(x, y)
            # 鼠标位置不在图片范围内，则以图片中心缩放
            else:
                x = self.m_drawPoint.x() - (self.m_rectPixmap.width() * (self.m_scaleValue - oldScale)) / 2
                y = self.m_drawPoint.y() - (self.m_rectPixmap.height() * (self.m_scaleValue - oldScale)) / 2
                self.m_drawPoint = QtCore.QPointF(x, y)
            self.update()

    def paintEvent(self, event):
        if not self.m_rectPixmap.isEmpty():
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
            scaled_pixmap = self.pixmap().scaled(self.pixmap().size() * self.m_scaleValue, QtCore.Qt.KeepAspectRatio,
                                                 QtCore.Qt.SmoothTransformation)
            self.m_scalePixmap = QtCore.QRectF(scaled_pixmap.rect())
            painter.drawPixmap(self.m_drawPoint, scaled_pixmap)

    def resizeEvent(self, a0):
        if not self.m_rectPixmap.isEmpty():
            center_x = ((self.m_drawPoint.x() + self.m_rectPixmap.width() * self.m_scaleValue / 2) /
                        self.old_width * self.width() -
                        self.m_rectPixmap.width() * self.m_scaleValue / 2)
            center_y = ((self.m_drawPoint.y() + self.m_rectPixmap.height() * self.m_scaleValue / 2) /
                        self.old_height * self.height() -
                        self.m_rectPixmap.height() * self.m_scaleValue / 2)
            self.m_drawPoint.setX(center_x)
            self.m_drawPoint.setY(center_y)

            self.update()
            self.old_width = self.width()
            self.old_height = self.height()


class DepthImageProcessor:
    def __init__(self, image):
        self.depth_image: np.ndarray = image
        self.original_min = self.depth_image.min()
        self.original_max = self.depth_image.max()

    @classmethod
    def read_16bit_image(cls, image_path):
        image = cv.imread(image_path, cv.IMREAD_ANYDEPTH)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        return cls(image)

    def get_depth_statistics(self):
        return {
            'min': self.depth_image.min(),
            'max': self.depth_image.max(),
            'mean': self.depth_image.mean(),
            'std': self.depth_image.std()
        }

    def apply_colormap(self, colormap_name='JET', min_val=None, max_val=None):
        colormap_dict = {
            'JET': cv.COLORMAP_JET,
            'HOT': cv.COLORMAP_HOT,
            'COOL': cv.COLORMAP_COOL,
            'SPRING': cv.COLORMAP_SPRING,
            'SUMMER': cv.COLORMAP_SUMMER,
            'AUTUMN': cv.COLORMAP_AUTUMN,
            'WINTER': cv.COLORMAP_WINTER,
            'BONE': cv.COLORMAP_BONE,
            'RAINBOW': cv.COLORMAP_RAINBOW
        }

        if min_val is None:
            min_val = self.original_min
        if max_val is None:
            max_val = self.original_max

        # 处理无效值
        valid_mask = (self.depth_image >= min_val) & (self.depth_image <= max_val)
        processed = np.copy(self.depth_image)
        processed[~valid_mask] = min_val

        # 归一化
        normalized = cv.normalize(processed, None, 0, 255, cv.NORM_MINMAX)
        normalized_8bit = normalized.astype(np.uint8)

        # 应用色彩映射
        pseudocolor = cv.applyColorMap(normalized_8bit, colormap_dict[colormap_name])

        return pseudocolor


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    with open(STYLES_FILE, "r") as f: app.setStyleSheet(f.read())
    win = DepthImageWindow()
    win.show()
    sys.exit(app.exec_())
