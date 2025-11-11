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
import sys
import json
import trimesh
import numpy as np

from scipy.spatial.distance import pdist
from PyQt5 import QtWidgets, QtCore, QtGui
from ..path import ICON_PATH


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

        self.ply_list = QtWidgets.QListWidget()
        self.ply_list.setFixedSize(620, 400)
        main_layout.addWidget(self.ply_list)

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

        self.choose_ply_button.clicked.connect()
        self.export_button.clicked.connect()

    @staticmethod
    def get_model_info(ply_file_path):
        mesh = trimesh.load(ply_file_path)

        vertices = mesh.vertices  # 获取顶点

        # 计算直径（顶点之间的最大距离）
        # 对于顶点数量较多的模型，此计算可能较慢
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

        return model_info

    def generate_models_info_json(self, models_dir, output_file='models_info.json'):
        models_info = {}

        # 遍历目录中的所有.ply文件
        for filename in sorted(os.listdir(models_dir)):
            if filename.endswith('.ply'):
                # 提取对象ID（假设文件名格式为obj_xxxxx.ply）
                obj_id = str(int(filename.split('_')[1].split('.')[0]))

                file_path = os.path.join(models_dir, filename)

                print(f"正在处理模型: {filename} (ID: {obj_id})")

                # 获取模型信息
                info = self.get_model_info(file_path)
                models_info[obj_id] = info

        # 将信息写入json文件
        output_path = os.path.join(models_dir, output_file)
        with open(output_path, 'w') as f:
            json.dump(models_info, f, indent=4)
