"""
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: custom_qt_interactor.py
@time: 2023-07-03 20:30
@desc: custom overwrite default qt interactor to include the mouse press and release related events.
"""
import os
import vtk
from PyQt5.QtCore import pyqtSignal

from ..tools import utils
from pyvistaqt import QtInteractor


class CustomQtInteractor(QtInteractor):
    pose_changed_by_drag = pyqtSignal()

    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        self.set_background("4c4c4c")
        self.selected_actor = None  # Initialize the attribute here
        self._is_dragging = False
        self._is_right_dragging = None
        self.last_mouse_pos = []
        self.tz_sensitivity = 0.5

    def mousePressEvent(self, event):
        # Call superclass method for left and middle buttons
        if event.button() in (1, 4):
            super().mousePressEvent(event)
            self.press_callback(self.iren.interactor)
            self._is_dragging = True

        elif event.button() == 2:
            self.press_callback(self.iren.interactor)
            picked_actor = self.selected_actor

            # 只有当点中的是当前选中的模型时，才开始Z轴拖拽
            if self.main_window.scene.mesh_container.reference is not None:
                selected_actor = self.main_window.scene.mesh_container.get_mesh_by_index(
                    self.main_window.scene.mesh_container.reference).actor
                # print(self.main_window.scene.mesh_container.get_index_by_actor(picked_actor))
                # print(self.main_window.scene.mesh_container.get_index_by_actor(selected_actor))
                # print(picked_actor == selected_actor)
                if picked_actor == selected_actor:
                    self._is_right_dragging = True
                    self.last_mouse_pos = [event.pos().x(), event.pos().y()]
                    # 不调用 super().mousePressEvent(event)，以阻止VTK的默认缩放
        # Always call press_callback

    def mouseReleaseEvent(self, event):
        if event.button() in (1, 4):  # Left, and middle mouse buttons
            self.release_callback()
            if self._is_dragging:
                self.pose_changed_by_drag.emit()
                self._is_dragging = False
        if event.button() == 2:
            self.release_callback()
            if self._is_right_dragging:
                self._is_right_dragging = False
                self.pose_changed_by_drag.emit()
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        """
        核心修复：重写鼠标移动事件，实现自定义的右键拖拽逻辑。
        """
        # 如果是右键拖拽模式
        if self._is_right_dragging:
            # 获取当前鼠标位置和垂直移动量
            current_pos = [event.pos().x(), event.pos().y()]
            dy = current_pos[1] - self.last_mouse_pos[1]

            # 获取当前选中的模型和其变换矩阵
            mesh_index = self.main_window.scene.mesh_container.reference
            if isinstance(mesh_index, int) and mesh_index >= 0:
                mesh_model = self.main_window.scene.mesh_container.get_mesh_by_index(self.main_window.scene.mesh_container.reference)
                matrix = utils.get_actor_user_matrix(mesh_model)

                # 将鼠标垂直移动量转换为Tz的变化量
                # 鼠标向上拖拽（dy为负）-> 模型远离相机（Tz增加）
                dTz = -dy * self.tz_sensitivity

                # 直接修改变换矩阵的Tz分量
                matrix[2, 3] += dTz
                mesh_model.actor.user_matrix = matrix

                # 实时更新UI中的Tz输入框
                self.main_window.block_value_change_signal(self.main_window.camera_tz_control.spin_box, matrix[2, 3])

                # 更新上一次的鼠标位置
                self.last_mouse_pos = current_pos
            return  # 阻止事件传递给VTK

        # 如果不是右键拖拽，则执行VTK的默认行为（例如左键旋转）
        super().mouseMoveEvent(event)

    def press_callback(self, obj, *args):
        x, y = obj.GetEventPosition()
        prop_picker = vtk.vtkPropPicker()
        if prop_picker.Pick(x, y, 0, self.renderer):
            self.selected_actor = prop_picker.GetActor()  # Use a different attribute name

    def release_callback(self):
        if self.selected_actor:
            index = self.main_window.scene.mesh_container.get_index_by_actor(self.selected_actor)
            # name = self.selected_actor.name
            if index != -1:
                self.main_window.check_mesh_button(index, True)
            # if name in self.main_window.scene.mesh_container.meshes:
            #     self.main_window.check_mesh_button(name, output_text=True)
            # elif name in self.main_window.scene.mask_container.masks:
            #     self.main_window.check_mask_button(name)
        self.selected_actor = None  # Reset the attribute
        #  * very important to sync the poses if the link_mesh_button is checked
        self.main_window.on_link_mesh_button_toggle(checked=self.main_window.link_mesh_button.isChecked(),
                                                    clicked=False)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        mesh_paths = []
        image_paths = []
        mask_paths = []
        unsupported_files = []

        mask_indicators = ['_mask', '_seg', '_label']
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            # Load mesh files
            if file_path.endswith(('.mesh', '.ply', '.stl', '.obj', '.off', '.dae', '.fbx', '.3ds', '.x3d')):
                mesh_paths.append(file_path)
            # Load image/mask files
            elif file_path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp', '.ico')):
                filename = os.path.basename(file_path).lower()
                if any(indicator in filename for indicator in mask_indicators):
                    mask_paths.append(file_path)
                else:
                    image_paths.append(file_path)
            else:
                unsupported_files.append(file_path)

        self.main_window.add_mesh_file(mesh_paths=mesh_paths)
        self.main_window.add_image_file(image_paths=image_paths)
        self.main_window.add_mask_file(mask_paths=mask_paths)
        if len(unsupported_files) > 0: utils.display_warning(f"File {unsupported_files} format is not supported!")
