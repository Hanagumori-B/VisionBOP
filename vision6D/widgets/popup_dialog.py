"""
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: popup_dialog.py
@time: 2023-07-03 20:32
@desc: create the dialog for popup window
"""

# General import
import numpy as np

# Qt5 import
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt

# self defined package import
np.set_printoptions(suppress=True)


class PopUpDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, on_button_click=None):
        super().__init__(parent)

        self.setFixedSize(250, 200)

        self.setWindowTitle("Vision6D - Colors")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)  # Disable the question mark

        button_grid = QtWidgets.QGridLayout()
        colors = [
            "nocs", "wheat", "magenta", "yellow", "white",
            "select", "dodgerblue", "cyan", "lime", "black",]

        button_count = 0
        # two columns
        for i in range(2):
            for j in range(int(len(colors) // 2)):
                name = f"{colors[button_count]}"
                button = QtWidgets.QPushButton(name)
                button.setMaximumSize(120, 35)
                button.setMinimumSize(120, 35)
                if name == 'select':
                    button.clicked.connect(lambda: on_button_click(str(self.pick_color())))
                elif name == 'nocs':
                    gradient_str = """
                                background-color: qlineargradient(
                                    spread:pad, x1:0, y1:0, x2:1, y2:1,
                                    stop:0 red, stop:0.17 orange, stop:0.33 yellow,
                                    stop:0.5 green, stop:0.67 blue, stop:0.83 indigo, stop:1 violet);
                                """
                    button.setStyleSheet(gradient_str)
                    button.clicked.connect(lambda _, idx=name: on_button_click(str(idx)))
                else:
                    button.setStyleSheet(f"background-color: {name};")
                    button.clicked.connect(lambda _, idx=name: on_button_click(str(idx)))
                button_grid.addWidget(button, j, i)
                button_count += 1

        self.setLayout(button_grid)

    def pick_color(self):
        """打开颜色对话框，让用户选择颜色。"""
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor('black'), self)
        if color.isValid():
            return color.name()
        return 'nocs'

