"""
@author:Hanagumori-B <yjp.y@foxmail.com>
@license: (C) Copyright.
@contact: yjp.y@foxmail.com
@software: Vision6D
@file: export_models_info_json.py
@time: 2025-10-14 17:08
@desc: export models_info.json
"""

import os
import pathlib
import json
import trimesh
import numpy as np

from scipy.spatial.distance import pdist
from PyQt5 import QtWidgets, QtCore, QtGui
from ..path import ICON_PATH


class PlyListWidget(QtWidgets.QListWidget):
    def allItemsText(self):
        return [self.item(i).text() for i in range(self.count())]


class ExportModelsInfo(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export models_info.json")
        self.setFixedSize(640, 500)
        self.setWindowIcon(QtGui.QIcon(str(ICON_PATH / 'logo.ico')))

        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)

        main_layout = QtWidgets.QVBoxLayout(self.centralWidget)
        button_layout = QtWidgets.QHBoxLayout()

        self.ply_list = PlyListWidget()
        self.ply_list.setFixedSize(620, 400)
        self.ply_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ply_list.customContextMenuRequested.connect(self.show_context_menu)
        main_layout.addWidget(self.ply_list)

        self.save_ply_files_by_naming_rule = QtWidgets.QCheckBox(self)
        self.save_ply_files_by_naming_rule.setText("Save PLY Files by Naming Rule as the Same Time")
        self.save_ply_files_by_naming_rule.setChecked(True)
        main_layout.addWidget(self.save_ply_files_by_naming_rule)

        self.choose_ply_button = QtWidgets.QPushButton("Choose PLY Files")
        self.choose_ply_button.setFixedSize(120, 35)
        self.export_button = QtWidgets.QPushButton("Export")
        self.export_button.setFixedSize(120, 35)
        button_layout.addItem(QtWidgets.QSpacerItem(100, 35))
        button_layout.addWidget(self.choose_ply_button)
        button_layout.addItem(QtWidgets.QSpacerItem(100, 35))
        button_layout.addWidget(self.export_button)
        button_layout.addItem(QtWidgets.QSpacerItem(100, 35))
        main_layout.addLayout(button_layout)

        self.choose_ply_button.clicked.connect(self.choose_ply_files)
        self.export_button.clicked.connect(self.generate_models_info_json)

    @staticmethod
    def get_model_info(ply_file_path):
        mesh = trimesh.load(ply_file_path)
        vertices = mesh.vertices  # 获取顶点

        # 计算直径（顶点之间的最大距离）
        if len(vertices) > 1:
            diameter = np.max(pdist(vertices))
        else:
            diameter = 0.0

        # 计算3D边界框
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        sizes = max_coords - min_coords

        model_info = {
            'diameter': float(diameter),
            'min_x': float(min_coords[0]),
            'min_y': float(min_coords[1]),
            'min_z': float(min_coords[2]),
            'size_x': float(sizes[0]),
            'size_y': float(sizes[1]),
            'size_z': float(sizes[2]),
        }

        return model_info, mesh

    def choose_ply_files(self):
        models_paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select PLY Files", "", "ply Files (*.ply)")
        for model_path in models_paths:
            if model_path not in self.ply_list.allItemsText():
                self.ply_list.addItem(model_path)

    def generate_models_info_json(self):
        models_paths = self.ply_list.allItemsText()
        output_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory")
        models_info = {}

        # 遍历目录中的所有.ply文件
        for file_path in sorted(models_paths):
            if file_path.endswith('.ply'):
                obj_id = str(int(''.join([d for d in file_path if d.isdigit()])))  # 提取对象ID
                info, mesh = self.get_model_info(file_path)
                models_info[obj_id] = info
                if self.save_ply_files_by_naming_rule.isChecked():
                    mesh.export(pathlib.Path(output_path) / f"obj_{int(obj_id):06d}.ply")

        # 将信息写入json文件
        output_file = pathlib.Path(output_path) / "models_info.json"
        with open(output_file, 'w') as f:
            json.dump(models_info, f, indent=4)

        self.close()
        self.parent().output_text.append(f'Successfully exported {output_file}')

    def show_context_menu(self, pos):
        # 获取当前鼠标位置对应的项
        index = self.ply_list.indexAt(pos)
        if not index.isValid():
            return

        menu = QtWidgets.QMenu(self)
        remove_action = QtWidgets.QAction('Remove', self)
        remove_action.triggered.connect(lambda: self.ply_list.takeItem(index.row()))
        menu.addAction(remove_action)
        menu.exec_(self.mapToGlobal(pos))
