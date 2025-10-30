"""
@author:Hanagumori-B
@license: (C) Copyright.
@contact: yjp.y@foxmail.com
@software: Vision6D
@file: show_depth_image.py
@time: 2025-10-30 21:21
@desc: show a 16bit depth image in colormap
"""


from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt
from ..path import ICON_PATH


class ExportBopDataSettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Bop Dataset Settings")
        self.setModal(True)
        self.setFixedSize(640, 500)
        self.setWindowIcon(QtGui.QIcon(str(ICON_PATH / 'logo.ico')))
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        layout = QtWidgets.QGridLayout()

        self.setLayout(layout)
        layout.addItem(QtWidgets.QSpacerItem(100, 35), 0, 0, 1, 1)
        layout.addItem(QtWidgets.QSpacerItem(100, 35), 1, 0, 1, 1)
        layout.addItem(QtWidgets.QSpacerItem(100, 35), 2, 0, 1, 1)
        layout.addItem(QtWidgets.QSpacerItem(100, 35), 3, 0, 1, 1)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setFixedSize(100, 35)
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button, 3, 1, 1, 1)
        layout.addItem(QtWidgets.QSpacerItem(100, 35), 3, 2, 1, 1)
        self.export_button = QtWidgets.QPushButton("Export")
        self.export_button.setFixedSize(100, 35)
        self.export_button.clicked.connect(self.accept)
        layout.addWidget(self.export_button, 3, 3, 1, 1)
        layout.addItem(QtWidgets.QSpacerItem(100, 35), 3, 4, 1, 1)
