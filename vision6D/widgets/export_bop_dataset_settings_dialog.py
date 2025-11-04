"""
@author:Hanagumori-B
@license: (C) Copyright.
@contact: yjp.y@foxmail.com
@software: Vision6D
@file: show_depth_image.py
@time: 2025-10-30 21:21
@desc: show a 16bit depth image in colormap
"""
import json

from PyQt5 import QtWidgets, QtGui, QtCore
from ..path import ICON_PATH


class ExportBopDataSettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Bop Dataset Settings")
        self.setModal(True)
        self.setFixedSize(640, 500)
        self.setWindowIcon(QtGui.QIcon(str(ICON_PATH / 'logo.ico')))
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

        main_layout = QtWidgets.QVBoxLayout()
        first_layout = QtWidgets.QHBoxLayout()
        resolution_layout = QtWidgets.QGridLayout()
        choose_workspace_layout = QtWidgets.QVBoxLayout()
        button_layout = QtWidgets.QHBoxLayout()

        resolution_label = QtWidgets.QLabel("Resolution:")
        resolution_layout.addWidget(resolution_label, 0, 0, 1, 1)
        resolution_layout.addWidget(QtWidgets.QLabel("Width"), 0, 1, 1, 1)
        resolution_layout.addWidget(QtWidgets.QLabel("Height"), 0, 3, 1, 1)
        resolution_layout.addWidget(QtWidgets.QLabel("Original:"), 1, 0, 1, 1)
        self.original_width_label = QtWidgets.QLabel(f"{parent.scene.canvas_width}")
        self.original_height_label = QtWidgets.QLabel(f"{parent.scene.canvas_height}")
        resolution_layout.addWidget(self.original_width_label, 1, 1, 1, 1)
        resolution_layout.addWidget(QtWidgets.QLabel("X"), 1, 2, 1, 1)
        resolution_layout.addWidget(self.original_height_label, 1, 3, 1, 1)
        self.resized_width_label = QtWidgets.QLabel(f"{parent.scene.canvas_width}")
        self.resized_height_label = QtWidgets.QLabel(f"{parent.scene.canvas_height}")
        resolution_layout.addWidget(QtWidgets.QLabel("Resized:"), 2, 0, 1, 1)
        resolution_layout.addWidget(self.resized_width_label, 2, 1, 1, 1)
        resolution_layout.addWidget(QtWidgets.QLabel("X"), 2, 2, 1, 1)
        resolution_layout.addWidget(self.resized_height_label, 2, 3, 1, 1)
        self.choose_resize_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.choose_resize_slider.setMinimum(1)
        self.choose_resize_slider.setMaximum(10)
        self.choose_resize_slider.setPageStep(1)
        self.choose_resize_slider.setTickInterval(1)
        self.choose_resize_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        resolution_layout.addWidget(self.choose_resize_slider, 3, 0, 1, 4)
        resolution_layout.addWidget(QtWidgets.QLabel("Reduce in Scale:"), 4, 0, 1, 2)
        self.reduce_label = QtWidgets.QLabel("1")
        resolution_layout.addWidget(self.reduce_label, 4, 2, 1, 1)
        self.choose_resize_slider.valueChanged.connect(self.resize_slider_changed)
        resolution_layout.addItem(QtWidgets.QSpacerItem(100, 35), 4, 3, 1, 1)

        self.choose_all_button = QtWidgets.QPushButton("Choose All")
        self.choose_all_button.setFixedSize(100, 35)
        self.choose_all_button.clicked.connect(self.click_choose_all_workspaces)
        choose_workspace_layout.addWidget(self.choose_all_button)
        self.cancel_all_button = QtWidgets.QPushButton("Cancel All")
        self.cancel_all_button.setFixedSize(100, 35)
        self.cancel_all_button.clicked.connect(self.click_cancel_all_workspaces)
        choose_workspace_layout.addWidget(self.cancel_all_button)
        self.workspace_list_view = QtWidgets.QListView()
        workspace_list = parent.workspaces
        self.workspace_list_model = QtCore.QStringListModel(workspace_list)
        self.workspace_list_view.setModel(self.workspace_list_model)
        self.workspace_list_view.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.workspace_list_view.entered.connect(self.on_workspace_list_view_entered)
        self.workspace_list_view.selectionModel().select(self.workspace_list_model.index(parent.current_workspace_index), QtCore.QItemSelectionModel.Select)
        self.workspace_list_view.setMouseTracking(True)
        choose_workspace_layout.addWidget(self.workspace_list_view)

        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setFixedSize(100, 35)
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addItem(QtWidgets.QSpacerItem(100, 35))
        button_layout.addWidget(self.cancel_button)
        button_layout.addItem(QtWidgets.QSpacerItem(100, 35))
        self.export_button = QtWidgets.QPushButton("Export")
        self.export_button.setFixedSize(100, 35)
        self.export_button.clicked.connect(self.accept)
        button_layout.addWidget(self.export_button)
        button_layout.addItem(QtWidgets.QSpacerItem(100, 35))

        resolution_widget = QtWidgets.QWidget()
        resolution_widget.setLayout(resolution_layout)
        resolution_widget.setMaximumSize(300, 150)
        resolution_widget.setMinimumSize(300, 150)
        first_layout.addWidget(resolution_widget)
        first_layout.addLayout(choose_workspace_layout)
        main_layout.addLayout(first_layout)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    # --slots--

    def resize_slider_changed(self):
        if self.choose_resize_slider.value() == 1:
            self.reduce_label.setText("1")
        else:
            self.reduce_label.setText(f"1/{self.choose_resize_slider.value()}")
        self.resized_width_label.setStyleSheet("color: #b1b1b1;")
        self.resized_height_label.setStyleSheet("color: #b1b1b1;")
        width = int(self.original_width_label.text())
        height = int(self.original_height_label.text())
        self.resized_width_label.setText(str(width // self.choose_resize_slider.value()))
        self.resized_height_label.setText(str(height // self.choose_resize_slider.value()))
        if width % self.choose_resize_slider.value() != 0:
            self.resized_width_label.setStyleSheet("color: red;")
        if height % self.choose_resize_slider.value() != 0:
            self.resized_height_label.setStyleSheet("color: red;")

    def click_choose_all_workspaces(self):
        self.workspace_list_view.selectAll()

    def click_cancel_all_workspaces(self):
        self.workspace_list_view.selectionModel().clearSelection()

    def on_workspace_list_view_entered(self, index: QtCore.QModelIndex):
        # with open(str(self.workspace_list_model.stringList()[index.row()]), 'r') as f:
        with open(str(index.data()), 'r') as f:
            self.resized_width_label.setStyleSheet("color: #b1b1b1;")
            self.resized_height_label.setStyleSheet("color: #b1b1b1;")
            workspace = json.load(f)
            height = workspace["camera"]["canvas_height"]
            width = workspace["camera"]["canvas_width"]
            self.original_width_label.setText(str(width))
            self.original_height_label.setText(str(height))
            self.resized_width_label.setText(str(width // self.choose_resize_slider.value()))
            self.resized_height_label.setText(str(height // self.choose_resize_slider.value()))
            if width % self.choose_resize_slider.value() != 0:
                self.resized_width_label.setStyleSheet("color: red;")
            if height % self.choose_resize_slider.value() != 0:
                self.resized_height_label.setStyleSheet("color: red;")

    # --slots end--

    def get_workspace_list(self):
        return [index.data() for index in self.workspace_list_view.selectedIndexes()]

    def get_reduced_scale_list(self):
        """
        return: list[list[reduced_scale, (resized_width, resized_height), (actual_width_scale, actual_height_scale)]]
        """
        reduced_scale_list = []
        for index in self.workspace_list_view.selectedIndexes():
            reduced_scale = self.choose_resize_slider.value()
            with open(str(index.data()), 'r') as f:
                workspace = json.load(f)
                height = workspace["camera"]["canvas_height"]
                width = workspace["camera"]["canvas_width"]
                resized_width = width // reduced_scale
                resized_height = height // reduced_scale
                actual_width_scale = width / resized_width
                actual_height_scale = height / resized_height
            reduced_scale_list.append([reduced_scale, (resized_width, resized_height), (actual_width_scale, actual_height_scale)])
        return reduced_scale_list
