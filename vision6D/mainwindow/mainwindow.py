"""
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: mainwindow.py
@time: 2023-07-03 20:33
@desc: the mainwindow to run application
"""
# General import
import os
import ast
import math
import json
import pickle
import time

import trimesh
import pathlib
import functools
import vtk

import PIL.Image
import numpy as np
import pyvista as pv
import cv2 as cv

# Qt5 import
from PyQt5 import QtWidgets, QtGui
from pyvistaqt import MainWindow
from PyQt5.QtCore import Qt

# self defined package import
from ..widgets import CustomQtInteractor
from ..widgets import SearchBar
from ..widgets import PnPWindow
from ..widgets import CustomImageButtonWidget
from ..widgets import CustomMeshButtonWidget
from ..widgets import CustomMaskButtonWidget
from ..widgets import GetPoseDialog
from ..widgets import GetMaskDialog
from ..widgets import CalibrationDialog
from ..widgets import DistanceInputDialog
from ..widgets import CameraPropsInputDialog
from ..widgets import MaskWindow
from ..widgets import LiveWireWindow
from ..widgets import SamWindow
from ..widgets import CustomGroupBox
from ..widgets import CameraControlWidget
from ..widgets import SaveChangesDialog
from ..widgets import SquareButton
from ..widgets import ExportBopDialog

from ..tools import utils
from ..tools import exception
from ..tools import BopJsonEncoder
from ..tools import DepthImageWindow
from ..containers import Scene

from ..path import ICON_PATH, SAVE_ROOT

os.environ["QT_API"] = "pyqt5"  # Setting the Qt bindings for QtPy
np.set_printoptions(suppress=True)


# noinspection PyAttributeOutsideInit
class MyMainWindow(MainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)

        # Set up the main window layout
        self.setWindowTitle("Vision6D")
        self.setWindowIcon(QtGui.QIcon(str(ICON_PATH / 'logo.ico')))
        # the vision6D window is maximized by default
        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)
        self.setAcceptDrops(True)

        # Initialize
        self.image_button_group_actors = QtWidgets.QButtonGroup(self)
        self.mesh_button_group_actors = QtWidgets.QButtonGroup(self)
        self.mask_button_group_actors = QtWidgets.QButtonGroup(self)
        self.workspace_button_group_actors = QtWidgets.QButtonGroup(self)
        self.mesh_button_group_actors.setExclusive(True)
        self.workspaces = []  # List to store workspace file paths
        self.current_workspace_index = -1  # Index of the currently active workspace
        self.is_workspace_dirty = False

        self.background_plane_color = QtGui.QColor('#808080')
        self.background_plane_actor: pv.Actor | None = None
        # Create the plotter
        self.create_plotter()

        self.output_text = QtWidgets.QTextEdit()

        self.scene = Scene(self.plotter, self.output_text)

        self.toggle_hide_meshes_flag = False

        # Set bars
        self.set_panel_bar()
        self.set_menu_bars()

        # Set up the main layout with the left panel and the render window using QSplitter
        self.main_layout = QtWidgets.QHBoxLayout(self.main_widget)
        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(self.panel_widget)
        self.splitter.addWidget(self.plotter)
        self.splitter.setStretchFactor(0, 1)  # for self.panel_widget
        self.splitter.setStretchFactor(1, 5)  # for self.plotter

        self.main_layout.addWidget(self.splitter)

        # Show the plotter
        self.show_plot()

        # Shortcut key bindings
        self.key_bindings()

    @property
    def workspace_path(self):
        try:
            path = self.workspaces[self.current_workspace_index]
        except IndexError:
            path = ''
        return path

    def key_bindings(self):
        # Camera related key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("c"), self).activated.connect(self.reset_camera)
        QtWidgets.QShortcut(QtGui.QKeySequence("z"), self).activated.connect(self.scene.zoom_out)
        QtWidgets.QShortcut(QtGui.QKeySequence("x"), self).activated.connect(self.scene.zoom_in)

        # Mask related key bindings
        QtWidgets.QShortcut(QtGui.QKeySequence("t"), self).activated.connect(self.reset_mask)

        # Mesh related key bindings 
        QtWidgets.QShortcut(QtGui.QKeySequence("k"), self).activated.connect(self.reset_gt_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("l"), self).activated.connect(self.update_gt_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+z"), self).activated.connect(self.undo_actor_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+y"), self).activated.connect(self.redo_actor_pose)
        QtWidgets.QShortcut(QtGui.QKeySequence("y"), self).activated.connect(
            lambda up=True: self.toggle_surface_opacity(up))
        QtWidgets.QShortcut(QtGui.QKeySequence("u"), self).activated.connect(
            lambda up=False: self.toggle_surface_opacity(up))

        # todo: create the swith button for mesh and ct "ctrl + tap"
        QtWidgets.QShortcut(QtGui.QKeySequence("Tab"), self).activated.connect(self.scene.tap_toggle_opacity)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Tab"), self).activated.connect(self.scene.ctrl_tap_opacity)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+w"), self).activated.connect(self.clear_plot)

        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+s"), self).activated.connect(self.save_current_workspace)
        QtWidgets.QShortcut(QtGui.QKeySequence("Up"), self).activated.connect(
            lambda up=True: self.key_next_workspace(up))
        QtWidgets.QShortcut(QtGui.QKeySequence("Down"), self).activated.connect(
            lambda up=False: self.key_next_workspace(up))

    def reset_camera(self):
        self.plotter.camera = self.scene.camera.copy()
        self.on_link_mesh_button_toggle(checked=self.link_mesh_button.isChecked(), clicked=False)

    def handle_transformation_matrix(self, index, transformation_matrix):
        self.mesh_register(index, transformation_matrix)
        self.update_gt_pose()

    def mesh_register(self, index, pose):
        mesh_model = self.scene.mesh_container.get_mesh_by_index(index)
        if not mesh_model.undo_poses or not np.allclose(mesh_model.undo_poses[-1], pose):
            mesh_model.actor.user_matrix = pose
            mesh_model.undo_poses.append(pose)
            mesh_model.undo_poses = mesh_model.undo_poses[-20:]
            mesh_model.redo_poses.clear()
        self.is_workspace_dirty = True

    @utils.require_attributes([('scene.mesh_container.reference', "Please load a mesh first!")])
    def update_gt_pose(self, input_pose=None):
        if self.link_mesh_button.isChecked():
            for mesh_model in self.scene.mesh_container.meshes:
                mesh_model.initial_pose = mesh_model.actor.user_matrix if input_pose is None else input_pose
                mesh_model.undo_poses.clear()  # reset the undo_poses after updating the gt pose of a mesh object
                mesh_model.undo_poses.append(mesh_model.initial_pose)
                matrix = utils.get_actor_user_matrix(mesh_model)
                text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
                    matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3],
                    matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3],
                    matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3],
                    matrix[3, 0], matrix[3, 1], matrix[3, 2], matrix[3, 3])
                self.output_text.append(f"-> Update the {mesh_model.name} GT pose to:")
                self.output_text.append(text)
        else:
            mesh_model = self.scene.mesh_container.get_mesh_by_index(self.scene.mesh_container.reference)
            mesh_model.initial_pose = mesh_model.actor.user_matrix if input_pose is None else input_pose
            mesh_model.undo_poses.clear()  # reset the undo_poses after updating the gt pose of a mesh object
            mesh_model.undo_poses.append(mesh_model.initial_pose)
            matrix = utils.get_actor_user_matrix(mesh_model)
            text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
                matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3],
                matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3],
                matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3],
                matrix[3, 0], matrix[3, 1], matrix[3, 2], matrix[3, 3])
            self.output_text.append(f"-> Update the {mesh_model.name} GT pose to:")
            self.output_text.append(text)

    @utils.require_attributes([('scene.mesh_container.reference', "Please load a mesh first!")])
    def reset_gt_pose(self):
        if self.link_mesh_button.isChecked():
            for mesh_name, mesh_model in self.scene.mesh_container.meshes:
                self.mesh_register(mesh_name, mesh_model.initial_pose)
                matrix = utils.get_actor_user_matrix(mesh_model)
                text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
                    matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3],
                    matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3],
                    matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3],
                    matrix[3, 0], matrix[3, 1], matrix[3, 2], matrix[3, 3])
                self.output_text.append(f"-> Reset the {mesh_model.name} GT pose to:")
                self.output_text.append(text)
        else:
            mesh_model = self.scene.mesh_container.get_mesh_by_index(self.scene.mesh_container.reference)
            self.mesh_register(self.scene.mesh_container.reference, mesh_model.initial_pose)
            matrix = utils.get_actor_user_matrix(mesh_model)
            text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
                matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3],
                matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3],
                matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3],
                matrix[3, 0], matrix[3, 1], matrix[3, 2], matrix[3, 3])
            self.output_text.append(f"-> Reset the {mesh_model.name} GT pose to:")
            self.output_text.append(text)

        self.reset_camera()

        # todo: 将up和down键改为切换space

    def key_next_workspace(self, up=False):
        if not self.workspaces:
            return

        num_workspaces = len(self.workspaces)
        if up:  # Up arrow key, move to previous workspace in the list
            next_index = (self.current_workspace_index - 1 + num_workspaces) % num_workspaces
        else:  # Down arrow key, move to next workspace in the list
            next_index = (self.current_workspace_index + 1) % num_workspaces

        self.switch_workspace(next_index)

    def key_next_image_button(self, up=False):
        buttons = self.image_button_group_actors.buttons()
        checked_button = self.image_button_group_actors.checkedButton()
        if checked_button is not None:
            checked_button.setChecked(False)
            current_button_index = buttons.index(checked_button)
            if up:
                current_button_index = (current_button_index + 1) % len(buttons)
            else:
                current_button_index = (current_button_index - 1) % len(buttons)
            next_button = buttons[current_button_index]
            next_button.click()
            self.images_actors_group.scroll_area.ensureWidgetVisible(next_button)

    def showMaximized(self):
        super(MyMainWindow, self).showMaximized()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    # Camera related
    @utils.require_attributes([('scene.image_container.reference', "Please load an image first!")])
    def camera_calibrate(self):
        original_image = np.array(
            PIL.Image.open(self.scene.image_container.images[self.scene.image_container.reference].path), dtype='uint8')
        # make the original image shape is [h, w, 3] to match with the rendered calibrated_image
        original_image = original_image[..., :3]
        if len(original_image.shape) == 2: original_image = original_image[..., None]
        if original_image.shape[-1] == 1: original_image = np.dstack((original_image, original_image, original_image))
        calibrated_image = np.array(self.scene.image_container.render_image(self.scene.image_container.reference,
                                                                            self.plotter.camera.copy()),
                                    dtype='uint8')
        if original_image.shape != calibrated_image.shape:
            utils.display_warning("Original image shape is not equal to calibrated image shape!")
        else:
            CalibrationDialog(calibrated_image, original_image).exec_()

    def set_camera(self):
        dialog = CameraPropsInputDialog(
            line1=("Fx", self.scene.fx),
            line2=("Fy", self.scene.fy),
            line3=("Cx", self.scene.cx),
            line4=("Cy", self.scene.cy),
            line5=("Canvas Height", self.scene.canvas_height),
            line6=("Canvas Width", self.scene.canvas_width),
            line7=("Camera View Up", self.scene.cam_viewup))
        if dialog.exec():
            fx, fy, cx, cy, canvas_height, canvas_width, cam_viewup = dialog.getInputs()
            pre_fx = self.scene.fx
            pre_fy = self.scene.fy
            pre_cx = self.scene.cx
            pre_cy = self.scene.cy
            pre_canvas_height = self.scene.canvas_height
            pre_canvas_width = self.scene.canvas_width
            pre_cam_viewup = self.scene.cam_viewup
            if not (fx == '' or fy == '' or cx == '' or cy == '' or cam_viewup == ''):
                try:
                    self.scene.fx = ast.literal_eval(fx)
                    self.scene.fy = ast.literal_eval(fy)
                    self.scene.cx = ast.literal_eval(cx)
                    self.scene.cy = ast.literal_eval(cy)
                    self.scene.canvas_height = ast.literal_eval(canvas_height)
                    self.scene.canvas_width = ast.literal_eval(canvas_width)
                    self.scene.cam_viewup = ast.literal_eval(cam_viewup)
                    self.scene.set_camera_intrinsics(self.scene.fx, self.scene.fy, self.scene.cx, self.scene.cy,
                                                     self.scene.canvas_height)
                    self.scene.set_camera_extrinsics(self.scene.cam_viewup)
                    if self.scene.image_container.reference is not None:
                        self.scene.handle_image_click(self.scene.image_container.reference)
                    self.reset_camera()
                except:
                    self.scene.fx = pre_fx
                    self.scene.fy = pre_fy
                    self.scene.cx = pre_cx
                    self.scene.cy = pre_cy
                    self.scene.canvas_height = pre_canvas_height
                    self.scene.canvas_width = pre_canvas_width
                    self.scene.cam_viewup = pre_cam_viewup
                    utils.display_warning("Error occured, check the format of the input values")

    def add_image_file(self, image_paths: list[pathlib.Path | str] = None, prompt=False):
        if prompt:
            image_paths, _ = QtWidgets.QFileDialog().getOpenFileNames(None, "Open file", "",
                                                                      "Files (*.png *.jpg *.jpeg *.tiff *.bmp *.webp *.ico)")
        if image_paths:
            # 限制一个workspace只能加载一张图片
            # 1. 检查是否已经存在图片
            if len(self.scene.image_container.images) > 0:
                utils.display_warning(
                    "A workspace can only contain one image at a time. Please clear the existing image first.")
                return  # 终止函数执行

            # 2. 如果用户选择了多张图片，也只加载第一张并发出警告
            if len(image_paths) > 1:
                utils.display_warning("You can only load one image. Only the first selected image will be loaded.")
                image_paths = [image_paths[0]]

            # Set up the camera
            self.scene.set_camera_intrinsics(self.scene.fx, self.scene.fy, self.scene.cx, self.scene.cy,
                                             self.scene.canvas_height)
            self.scene.set_camera_extrinsics(self.scene.cam_viewup)
            try:
                for image_path in image_paths:
                    image_model = self.scene.image_container.add_image_attributes(image_path)
                    button_widget = CustomImageButtonWidget(image_model.name, image_path=image_model.path)
                    button_widget.removeButtonClicked.connect(self.remove_image_button)
                    image_model.opacity_spinbox = button_widget.double_spinbox
                    button = button_widget.button
                    button.setCheckable(True)
                    button.clicked.connect(lambda _, name=image_model.name: self.scene.handle_image_click(name))
                    self.image_button_group_actors.addButton(button_widget.button)
                    self.images_actors_group.widget_layout.insertWidget(0, button_widget)
                self.check_image_button(image_model.name)
            except exception as e:
                utils.display_warning(str(e))
            self.reset_camera()

    def add_mask_file(self, mask_paths: list[pathlib.Path | str] = None, prompt=False):
        if prompt:
            mask_paths, _ = QtWidgets.QFileDialog().getOpenFileNames(None, "Open file", "",
                                                                     "Files (*.npy *.png *.jpg *.jpeg *.tiff *.bmp *.webp *.ico)")
        if mask_paths:
            for mask_path in mask_paths:
                mask_model = self.scene.mask_container.add_mask(mask_source=mask_path,
                                                                fy=self.scene.fy,
                                                                cx=self.scene.cx,
                                                                cy=self.scene.cy)
                self.add_mask_button(mask_model.name)

    def set_mask(self):
        get_mask_dialog = GetMaskDialog()
        res = get_mask_dialog.exec_()
        if res == QtWidgets.QDialog.Accepted:
            if get_mask_dialog.mask_path:
                self.add_mask_file(get_mask_dialog.mask_path)
            else:
                user_text = get_mask_dialog.get_text()
                points = exception.set_data_format(user_text)
                if points is not None:
                    if points.shape[1] == 2:
                        os.makedirs(SAVE_ROOT / "output", exist_ok=True)
                        os.makedirs(SAVE_ROOT / "output" / "mask_points", exist_ok=True)
                        if self.scene.image_container.images[self.scene.image_container.reference]:
                            mask_path = SAVE_ROOT / "output" / "mask_points" / f"{pathlib.Path(self.scene.image_container.images[self.scene.image_container.reference].path).stem}.npy"
                        else:
                            mask_path = SAVE_ROOT / "output" / "mask_points" / "mask_points.npy"
                        np.save(mask_path, points)
                        self.add_mask_file(mask_path)
                    else:
                        utils.display_warning("It needs to be a n by 2 matrix")

    @utils.require_attributes([('scene.mask_container.reference', "Please load a mask first!")])
    def reset_mask(self):
        mask_model = self.scene.mask_container.masks[self.scene.mask_container.reference]
        mask_model.actor.user_matrix = np.eye(4)

    def add_mesh_file(self, mesh_paths: list[pathlib.Path | str] = None, prompt=False):
        if prompt:
            mesh_paths, _ = QtWidgets.QFileDialog().getOpenFileNames(None, "Open file", "",
                                                                     "Files (*.mesh *.ply *.stl *.obj *.off *.dae *.fbx *.3ds *.x3d)")
        if mesh_paths:
            # Set up the camera
            self.scene.set_camera_intrinsics(self.scene.fx, self.scene.fy, self.scene.cx, self.scene.cy,
                                             self.scene.canvas_height)
            self.scene.set_camera_extrinsics(self.scene.cam_viewup)
            try:
                for mesh_path in mesh_paths:
                    mesh_model, new_index = self.scene.mesh_container.add_mesh_actor(mesh_source=mesh_path,
                                                                                     transformation_matrix=np.array(
                                                                                         [[1, 0, 0, 0],
                                                                                          [0, 1, 0, 0],
                                                                                          [0, 0, 1, 1e+3],
                                                                                          [0, 0, 0, 1]]))
                    button_widget = CustomMeshButtonWidget(mesh_model.name)
                    button_widget.colorChanged.connect(
                        lambda color, index=new_index: self.scene.mesh_color_value_change(index, color))
                    button_widget.removeButtonClicked.connect(self.remove_mesh_button)
                    mesh_model.color_button = button_widget.square_button
                    mesh_model.color_button.setStyleSheet(f"background-color: {mesh_model.color}")
                    mesh_model.opacity_spinbox = button_widget.double_spinbox
                    mesh_model.opacity_spinbox.setValue(mesh_model.opacity)
                    mesh_model.opacity_spinbox.valueChanged.connect(
                        lambda value, name=new_index: self.scene.mesh_container.set_mesh_opacity(name, value))
                    button = button_widget.button
                    button.setCheckable(True)
                    button.clicked.connect(
                        lambda _, name=new_index, output_text=True: self.check_mesh_button(name, output_text))
                    self.mesh_button_group_actors.addButton(button_widget.button, new_index)
                    self.mesh_actors_group.widget_layout.insertWidget(0, button_widget)
                if self.scene.mesh_container.reference is None: self.set_camera_spinbox(indicator=True)
                if self.link_mesh_button.isChecked() and self.scene.mesh_container.reference is not None:
                    self.on_link_mesh_button_toggle(checked=self.link_mesh_button.isChecked(), clicked=True)
                else:
                    self.check_mesh_button(new_index, output_text=True)
            except Exception as e:
                utils.display_warning(str(e))
            self.reset_camera()

    @utils.require_attributes([('scene.mesh_container.reference', "Please load a mesh first!")])
    def set_spacing(self):
        mesh_model = self.scene.mesh_container.get_mesh_by_index(self.scene.mesh_container.reference)
        spacing, ok = QtWidgets.QInputDialog().getText(QtWidgets.QMainWindow(), 'Input', "Set Spacing",
                                                       text=str(mesh_model.spacing))
        if ok:
            scaled_spacing = np.array(exception.set_spacing(spacing)) / np.array(mesh_model.spacing)
            vertices, _ = utils.get_mesh_actor_vertices_faces(mesh_model.actor)
            centroid = np.mean(vertices, axis=0)  # Calculate the centroid
            offset = vertices - centroid
            scaled_offset = offset * scaled_spacing
            vertices = centroid + scaled_offset
            mesh_model.pv_obj.points = vertices

    def toggle_surface_opacity(self, up):
        checked_button = self.mesh_button_group_actors.checkedButton()
        if checked_button:
            name = checked_button.text()
            if name in self.scene.mesh_container.meshes:
                change = 0.05
                if not up: change *= -1
                mesh_model = self.scene.mesh_container.meshes[name]
                current_opacity = mesh_model.opacity_spinbox.value()
                current_opacity += change
                current_opacity = np.clip(current_opacity, 0, 1)
                mesh_model.opacity_spinbox.setValue(current_opacity)

    def handle_hide_meshes_opacity(self, flag):
        checked_button = self.mesh_button_group_actors.checkedButton()
        checked_name = checked_button.text() if checked_button else None
        for button in self.mesh_button_group_actors.buttons():
            name = button.text()
            if name not in self.scene.mesh_container.meshes: continue
            if len(self.scene.mesh_container.meshes) != 1 and name == checked_name: continue
            mesh_model = self.scene.mesh_container.meshes[name]
            if flag:
                self.scene.mesh_container.set_mesh_opacity(name, 0)
            else:
                self.scene.mesh_container.set_mesh_opacity(name, mesh_model.previous_opacity)

    @utils.require_attributes([('scene.mesh_container.reference', "Please load a mesh first!")])
    def toggle_hide_meshes_button(self):
        self.toggle_hide_meshes_flag = not self.toggle_hide_meshes_flag
        self.handle_hide_meshes_opacity(self.toggle_hide_meshes_flag)

    def add_pose_file(self, pose_path):
        if pose_path:
            if isinstance(pose_path, list):
                transformation_matrix = np.array(pose_path)
            else:
                transformation_matrix = np.load(pose_path)
            # set the initial pose of the mesh to the loaded transformation matrix
            self.scene.mesh_container.get_mesh_by_index(
                self.scene.mesh_container.reference).initial_pose = transformation_matrix
            self.reset_gt_pose()

    @utils.require_attributes([('scene.mesh_container.reference', "Please load a mesh first!")])
    def set_pose(self):
        mesh_model = self.scene.mesh_container.get_mesh_by_index(self.scene.mesh_container.reference)
        get_pose_dialog = GetPoseDialog(utils.get_actor_user_matrix(mesh_model))
        res = get_pose_dialog.exec_()
        if res == QtWidgets.QDialog.Accepted:
            user_text = get_pose_dialog.get_text()
            if "," not in user_text:
                user_text = user_text.replace(" ", ",")
                user_text = user_text.strip().replace("[,", "[")
            input_pose = exception.set_data_format(user_text)
            if input_pose is not None:
                if input_pose.shape == (4, 4):
                    # set the mesh to be the originally loaded mesh
                    mesh_model = self.scene.mesh_container.get_mesh_by_index(self.scene.mesh_container.reference)
                    mesh_name = mesh_model.name
                    transformation_matrix = utils.get_actor_user_matrix(mesh_model)
                    vertices, faces = mesh_model.source_obj.vertices, mesh_model.source_obj.faces
                    mesh_model.pv_obj = pv.wrap(trimesh.Trimesh(vertices, faces, process=False))
                    try:
                        mesh = self.plotter.add_mesh(mesh_model.pv_obj, color=mesh_model.color,
                                                     opacity=mesh_model.opacity, pickable=True, name=mesh_name)
                    except ValueError:
                        self.scene.mesh_container.set_color(self.scene.mesh_container.reference, mesh_model.color)
                    mesh_model.actor = mesh
                    mesh_model.actor.user_matrix = transformation_matrix
                    self.scene.mesh_container.meshes[self.scene.mesh_container.reference] = mesh_model
                    mesh_model.undo_poses.clear()
                    mesh_model.undo_poses.append(transformation_matrix)
                    self.update_gt_pose(input_pose=input_pose)
                    self.set_camera_control_values(input_pose)
                    self.reset_gt_pose()
                    self.is_workspace_dirty = True
                else:
                    utils.display_warning("It needs to be a 4 by 4 matrix")

    @utils.require_attributes([('scene.mesh_container.reference', "Please load a mesh first!")])
    def undo_actor_pose(self):
        # checked_button = self.mesh_button_group_actors.checkedButton()
        if self.scene.mesh_container.get_poses_from_undo():
            # mesh_name = self.scene.mesh_container.reference
            # mesh_model = self.scene.mesh_container.meshes[mesh_name]
            mesh_model = self.scene.mesh_container.get_mesh_by_index(self.scene.mesh_container.reference)
            mesh_name = mesh_model.name

            self.output_text.append(f"-> Undo pose for {mesh_name}")
            # self.check_mesh_button(name=mesh_name, output_text=True)
            current_pose = utils.get_actor_user_matrix(mesh_model)
            self.set_camera_control_values(current_pose)
            self.on_link_mesh_button_toggle(checked=self.link_mesh_button.isChecked(), clicked=False)
            self.is_workspace_dirty = True
        # checked_button.click()

    @utils.require_attributes([('scene.mesh_container.reference', "Please load a mesh first!")])
    def redo_actor_pose(self):
        # checked_button = self.mesh_button_group_actors.checkedButton()
        if self.scene.mesh_container.get_pose_from_redo():
            # mesh_name = self.scene.mesh_container.reference
            # mesh_model = self.scene.mesh_container.meshes[mesh_name]
            mesh_model = self.scene.mesh_container.get_mesh_by_index(self.scene.mesh_container.reference)
            mesh_name = mesh_model.name

            self.output_text.append(f"-> Redo pose for {mesh_name}")
            current_pose = utils.get_actor_user_matrix(mesh_model)
            self.set_camera_control_values(current_pose)
            self.on_link_mesh_button_toggle(checked=self.link_mesh_button.isChecked(), clicked=False)
            self.is_workspace_dirty = True
            # self.check_mesh_button(name=mesh_name, output_text=True)
        # checked_button.click()

    # ^Menu
    def set_menu_bars(self):
        mainMenu = self.menuBar()

        # allow to add files
        fileMenu = mainMenu.addMenu('File')
        fileMenu.addAction('Add Workspace', functools.partial(self.add_workspace_file, prompt=True))
        fileMenu.addSeparator()
        fileMenu.addAction('Add Image', functools.partial(self.add_image_file, prompt=True))
        fileMenu.addAction('Add Mask', self.set_mask)
        fileMenu.addAction('Add Mesh', functools.partial(self.add_mesh_file, prompt=True))

        # allow to export files
        exportMenu = mainMenu.addMenu('Export')
        exportMenu.addAction('BOP Dataset', self.export_bop_dataset)
        exportMenu.addSeparator()
        exportMenu.addAction('Workspace', self.export_workspace)
        exportMenu.addAction('Image', self.export_image)
        exportMenu.addAction('Mask', self.export_mask)
        exportMenu.addAction('Pose', self.export_pose)
        exportMenu.addAction('Mesh Render', self.export_mesh_render)
        # exportMenu.addAction('SegMesh Render', self.export_segmesh_render)
        exportMenu.addSeparator()
        exportMenu.addAction('Camera Info', self.export_camera_info)

        toolsMenu = mainMenu.addMenu('Tools')
        toolsMenu.addAction('Colored Depth Image Viewer', self.colored_viewer)

    # ^Panel
    def set_panel_bar(self):
        # Create a left panel layout
        self.panel_widget = QtWidgets.QWidget()
        self.panel_layout = QtWidgets.QVBoxLayout(self.panel_widget)

        # Create a top panel bar with a toggle button
        self.panel_bar = QtWidgets.QMenuBar()
        self.toggle_action = QtWidgets.QAction("Panel", self)
        self.toggle_action.triggered.connect(self.toggle_panel)
        self.panel_bar.addAction(self.toggle_action)
        self.setMenuBar(self.panel_bar)

        self.panel_workspaces()

        self.panel_console()
        self.camera_control_console()
        self.panel_images_actors()
        self.panel_mesh_actors()
        self.panel_background_control()
        # self.panel_mask_actors()
        self.panel_output()

    def toggle_panel(self):
        if self.panel_widget.isVisible():
            # self.panel_widget width changes when the panel is visiable or hiden
            self.panel_widget_width = self.panel_widget.width()
            self.panel_widget.hide()
        else:
            self.panel_widget.show()

    def panel_workspaces(self):
        self.workspaces_group = CustomGroupBox("Workspaces", self)

        self.workspaces_group.addButtonClicked.connect(
            lambda path='', prompt=True: self.add_workspace_file(path, prompt))

        self.workspaces_group.new_button = QtWidgets.QPushButton("New")
        self.workspaces_group.new_button.setFixedSize(20, 20)
        self.workspaces_group.add_button_to_header(self.workspaces_group.new_button)
        self.workspaces_group.new_button.clicked.connect(self.new_workspace_file)

        func_options_button = QtWidgets.QPushButton("Func")
        func_options_button.setFixedSize(20, 20)
        func_options_menu = QtWidgets.QMenu()
        func_options_menu.addAction("Clear", self.clear_workspaces)  # 添加 Clear 选项
        func_options_button.setMenu(func_options_menu)
        self.workspaces_group.add_button_to_header(func_options_button)

        self.panel_layout.insertWidget(0, self.workspaces_group)

    def set_panel_row_column(self, row, column):
        column += 1
        if column % 3 == 0:
            row += 1
            column = 0
        return row, column

    @utils.require_attributes([('scene.image_container.reference', 'Please load an image first!'),
                               ('scene.mesh_container.reference', 'Please load a mesh first!')])
    def pnp_register(self):
        image = utils.get_image_actor_scalars(
            self.scene.image_container.images[self.scene.image_container.reference].actor)
        pnp_window = PnPWindow(image_source=image,
                               mesh_model=self.scene.mesh_container.get_mesh_by_index(
                                   self.scene.mesh_container.reference),
                               camera_intrinsics=self.scene.camera_intrinsics.astype(np.float32))
        pnp_window.transformation_matrix_computed.connect(
            lambda transformation_matrix: self.handle_transformation_matrix(self.scene.mesh_container.reference,
                                                                            transformation_matrix))

    def on_pose_options_selection_change(self, option):
        if option == "Set Pose":
            self.set_pose()
        elif option == "PnP Register":
            self.pnp_register()
        elif option == "Reset GT Pose (k)":
            self.reset_gt_pose()
        elif option == "Update GT Pose (l)":
            self.update_gt_pose()
        elif option == "Undo Pose":
            self.undo_actor_pose()

    def on_camera_options_selection_change(self, option):
        if option == "Set Camera":
            self.set_camera()
        elif option == "Reset Camera (c)":
            self.reset_camera()
        elif option == "Zoom In (x)":
            self.scene.zoom_in()
        elif option == "Zoom Out (z)":
            self.scene.zoom_out()
        elif option == "Calibrate":
            self.camera_calibrate()

    def panel_console(self):
        console_group = QtWidgets.QGroupBox("Console")
        console_group.setObjectName("consoleGroupBox")
        display_layout = QtWidgets.QVBoxLayout()
        display_layout.setContentsMargins(10, 15, 10, 5)

        # Create the top widgets (layout)
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Create Grid layout for function buttons
        top_grid_layout = QtWidgets.QGridLayout()

        row, column = 0, 0

        # Create a QPushButton that will act as a drop-down button and QMenu to act as the drop-down menu
        camera_options_button = QtWidgets.QPushButton("Set Camera")
        camera_options_menu = QtWidgets.QMenu()
        camera_options_menu.addAction("Set Camera", lambda: self.on_camera_options_selection_change("Set Camera"))
        camera_options_menu.addAction("Reset Camera (c)",
                                      lambda: self.on_camera_options_selection_change("Reset Camera (c)"))
        camera_options_menu.addAction("Zoom In (x)", lambda: self.on_camera_options_selection_change("Zoom In (x)"))
        camera_options_menu.addAction("Zoom Out (z)", lambda: self.on_camera_options_selection_change("Zoom Out (z)"))
        camera_options_menu.addAction("Calibrate", lambda: self.on_camera_options_selection_change("Calibrate"))
        camera_options_button.setMenu(camera_options_menu)
        top_grid_layout.addWidget(camera_options_button, row, column)

        pose_options_button = QtWidgets.QPushButton("Set Pose")
        pose_options_menu = QtWidgets.QMenu()
        pose_options_menu.addAction("Set Pose", lambda: self.on_pose_options_selection_change("Set Pose"))
        pose_options_menu.addAction("PnP Register", lambda: self.on_pose_options_selection_change("PnP Register"))
        pose_options_menu.addAction("Reset GT Pose (k)",
                                    lambda: self.on_pose_options_selection_change("Reset GT Pose (k)"))
        pose_options_menu.addAction("Update GT Pose (l)",
                                    lambda: self.on_pose_options_selection_change("Update GT Pose (l)"))
        pose_options_menu.addAction("Undo Pose", lambda: self.on_pose_options_selection_change("Undo Pose"))
        pose_options_button.setMenu(pose_options_menu)
        row, column = self.set_panel_row_column(row, column)
        top_grid_layout.addWidget(pose_options_button, row, column)

        # Other buttons
        clear_all_button = QtWidgets.QPushButton("Clear All")
        clear_all_button.clicked.connect(self.clear_plot)
        row, column = self.set_panel_row_column(row, column)
        top_grid_layout.addWidget(clear_all_button, row, column)

        top_grid_widget = QtWidgets.QWidget()
        top_grid_widget.setLayout(top_grid_layout)
        top_layout.addWidget(top_grid_widget)
        display_layout.addLayout(top_layout)
        console_group.setLayout(display_layout)
        self.panel_layout.addWidget(console_group)

    def set_mesh_pose(self):
        euler_angles = np.array([self.camera_rx_control.spin_box.value(), self.camera_ry_control.spin_box.value(),
                                 self.camera_rz_control.spin_box.value()])
        translation_vector = np.array([self.camera_tx_control.spin_box.value(), self.camera_ty_control.spin_box.value(),
                                       self.camera_tz_control.spin_box.value()])
        camera_control_matrix = utils.compose_transform(euler_angles, translation_vector)
        actor_actual_matrix = utils.get_actor_user_matrix(
            self.scene.mesh_container.get_mesh_by_index(self.scene.mesh_container.reference))
        offset_matrix = camera_control_matrix @ np.linalg.inv(
            actor_actual_matrix)  # Compute the offset (change) matrix between the current camera control value and the true pose
        user_matrix = self.scene.mesh_container.get_mesh_by_index(self.scene.mesh_container.reference).actor.user_matrix
        new_matrix = offset_matrix @ user_matrix
        self.scene.mesh_container.get_mesh_by_index(self.scene.mesh_container.reference).actor.user_matrix = new_matrix
        self.scene.handle_mesh_click(self.scene.mesh_container.reference, output_text=True)
        self.reset_camera()
        self.is_workspace_dirty = True

    def set_camera_spinbox(self, indicator):
        self.block_value_change_signal(self.camera_rx_control.spin_box, 0)
        self.block_value_change_signal(self.camera_ry_control.spin_box, 0)
        self.block_value_change_signal(self.camera_rz_control.spin_box, 0)
        self.block_value_change_signal(self.camera_tx_control.spin_box, 0)
        self.block_value_change_signal(self.camera_ty_control.spin_box, 0)
        self.block_value_change_signal(self.camera_tz_control.spin_box, 0)
        self.camera_rx_control.spin_box.setEnabled(indicator)
        self.camera_ry_control.spin_box.setEnabled(indicator)
        self.camera_rz_control.spin_box.setEnabled(indicator)
        self.camera_tx_control.spin_box.setEnabled(indicator)
        self.camera_ty_control.spin_box.setEnabled(indicator)
        self.camera_tz_control.spin_box.setEnabled(indicator)

    def camera_control_console(self):
        console_group = QtWidgets.QGroupBox("Camera Control")
        console_group.setObjectName("consoleGroupBox")

        display_layout = QtWidgets.QVBoxLayout()
        display_layout.setContentsMargins(0, 15, 0, 5)
        top_layout = QtWidgets.QHBoxLayout()
        top_grid_layout = QtWidgets.QGridLayout()

        row, column = 0, 0
        self.camera_rx_control = CameraControlWidget("Rx", "(deg)", 180)
        self.camera_rx_control.spin_box.setSingleStep(1)
        self.camera_rx_control.spin_box.valueChanged.connect(self.set_mesh_pose)
        top_grid_layout.addWidget(self.camera_rx_control, row, column)

        row, column = self.set_panel_row_column(row, column)
        self.camera_ry_control = CameraControlWidget("Ry", "(deg)", 180)
        self.camera_ry_control.spin_box.setSingleStep(1)
        self.camera_ry_control.spin_box.valueChanged.connect(self.set_mesh_pose)
        top_grid_layout.addWidget(self.camera_ry_control, row, column)

        row, column = self.set_panel_row_column(row, column)
        self.camera_rz_control = CameraControlWidget("Rz", "(deg)", 180)
        self.camera_rz_control.spin_box.setSingleStep(1)
        self.camera_rz_control.spin_box.valueChanged.connect(self.set_mesh_pose)
        top_grid_layout.addWidget(self.camera_rz_control, row, column)

        row, column = self.set_panel_row_column(row, column)
        self.camera_tx_control = CameraControlWidget("Tx", "(mm)", 1e+4)
        self.camera_tx_control.spin_box.setSingleStep(1)
        self.camera_tx_control.spin_box.valueChanged.connect(self.set_mesh_pose)
        top_grid_layout.addWidget(self.camera_tx_control, row, column)

        row, column = self.set_panel_row_column(row, column)
        self.camera_ty_control = CameraControlWidget("Ty", "(mm)", 1e+4)
        self.camera_ty_control.spin_box.setSingleStep(1)
        self.camera_ty_control.spin_box.valueChanged.connect(self.set_mesh_pose)
        top_grid_layout.addWidget(self.camera_ty_control, row, column)

        row, column = self.set_panel_row_column(row, column)
        self.camera_tz_control = CameraControlWidget("Tz", "(mm)", 1e+4)
        self.camera_tz_control.spin_box.setSingleStep(10)
        self.camera_tz_control.spin_box.valueChanged.connect(self.set_mesh_pose)
        top_grid_layout.addWidget(self.camera_tz_control, row, column)

        # Disable the spinboxes before loading a mesh
        self.set_camera_spinbox(indicator=False)

        top_grid_widget = QtWidgets.QWidget()
        top_grid_widget.setLayout(top_grid_layout)
        top_layout.addWidget(top_grid_widget)
        display_layout.addLayout(top_layout)
        console_group.setLayout(display_layout)
        self.panel_layout.addWidget(console_group)

    # In your main class or wherever you're using panel_images_actors
    def panel_images_actors(self):
        func_options_button = QtWidgets.QPushButton("Func")
        func_options_button.setFixedSize(50, 20)
        func_options_menu = QtWidgets.QMenu()
        func_options_menu.addAction("Set Distance", self.set_distance2camera)
        mirror_menu = QtWidgets.QMenu("Mirror", func_options_menu)
        mirror_menu.addAction("x-axis", functools.partial(self.mirror_image, direction="x"))
        mirror_menu.addAction("y-axis", functools.partial(self.mirror_image, direction="y"))
        func_options_menu.addMenu(mirror_menu)
        func_options_menu.addAction("Clear", self.clear_image)
        func_options_button.setMenu(func_options_menu)

        self.images_actors_group = CustomGroupBox("Image", self)
        self.images_actors_group.addButtonClicked.connect(
            lambda image_path='', prompt=True: self.add_image_file(image_path, prompt))
        self.images_actors_group.add_button_to_header(func_options_button)
        self.panel_layout.addWidget(self.images_actors_group)

    def on_link_mesh_button_toggle(self, checked, clicked):
        if clicked and checked and self.scene.mesh_container.reference is not None:
            # First, compute the average translation and set an identify matrix for R
            ts = [mesh_model.actor.user_matrix[:3, 3] for mesh_model in self.scene.mesh_container.meshes]
            average_t = np.mean(ts, axis=0)
            new_rt = np.eye(4)
            new_rt[:3, 3] = average_t
            for i, mesh_model in enumerate(self.scene.mesh_container.meshes):
                # Compute the relative matrix for each mesh with respect to the new_rt
                matrix = mesh_model.actor.user_matrix
                relative_matrix = np.linalg.inv(new_rt) @ matrix
                vertices, faces = utils.get_mesh_actor_vertices_faces(mesh_model.actor)
                transformed_vertices = utils.transform_vertices(vertices, relative_matrix)
                mesh_model.pv_obj = pv.wrap(trimesh.Trimesh(transformed_vertices, faces, process=False))
                try:
                    mesh = self.plotter.add_mesh(mesh_model.pv_obj, color=mesh_model.color, opacity=mesh_model.opacity,
                                                 pickable=True, name=mesh_model.name)
                except ValueError:
                    self.scene.mesh_container.set_color(i, mesh_model.color)
                mesh_model.actor = mesh
                mesh_model.actor.user_matrix = new_rt
                self.scene.mesh_container.meshes[i] = mesh_model
                mesh_model.undo_poses.clear()
                mesh_model.undo_poses.append(new_rt)
        elif checked and not clicked:
            for i, mesh_model in enumerate(self.scene.mesh_container.meshes):
                if i == self.scene.mesh_container.reference: continue
                mesh_model.actor.user_matrix = self.scene.mesh_container.get_mesh_by_index(
                    self.scene.mesh_container.reference).actor.user_matrix

    def panel_mesh_actors(self):
        func_options_button = QtWidgets.QPushButton("Func")
        func_options_button.setFixedSize(50, 20)
        func_options_menu = QtWidgets.QMenu()
        # func_options_menu.addAction("Set Spacing", self.set_spacing)
        # func_options_menu.addAction("Toggle Meshes", self.toggle_hide_meshes_button)
        mirror_menu = QtWidgets.QMenu("Mirror", func_options_menu)
        mirror_menu.addAction("x-axis", functools.partial(self.mirror_mesh, direction="x"))
        mirror_menu.addAction("y-axis", functools.partial(self.mirror_mesh, direction="y"))
        func_options_menu.addMenu(mirror_menu)
        func_options_menu.addAction("Clear", self.clear_mesh)
        func_options_button.setMenu(func_options_menu)

        self.link_mesh_button = QtWidgets.QPushButton("Link")
        self.link_mesh_button.setFixedSize(20, 20)
        self.link_mesh_button.setCheckable(True)
        self.link_mesh_button.setChecked(False)
        self.link_mesh_button.toggled.connect(
            lambda checked, clicked=True: self.on_link_mesh_button_toggle(checked, clicked))
        self.mesh_actors_group = CustomGroupBox("Mesh", self)
        self.mesh_actors_group.addButtonClicked.connect(
            lambda mesh_path='', prompt=True: self.add_mesh_file(mesh_path, prompt))
        self.mesh_actors_group.add_button_to_header(self.link_mesh_button)
        self.mesh_actors_group.add_button_to_header(func_options_button)
        self.panel_layout.addWidget(self.mesh_actors_group)

    def panel_mask_actors(self):
        draw_mask_button = QtWidgets.QPushButton("Draw")
        draw_mask_button.setFixedSize(20, 20)
        draw_options_menu = QtWidgets.QMenu()
        draw_options_menu.addAction("Set Mask", self.set_mask)
        draw_mask_menu = QtWidgets.QMenu("Draw Mask", draw_options_menu)
        draw_mask_menu.addAction("Free Hand", functools.partial(self.draw_mask, live_wire=False, sam=False))
        draw_mask_menu.addAction("Live Wire", functools.partial(self.draw_mask, live_wire=True, sam=False))
        draw_mask_menu.addAction("SAM", functools.partial(self.draw_mask, live_wire=False, sam=True))
        draw_options_menu.addMenu(draw_mask_menu)
        draw_options_menu.addAction("Reset Mask (t)", self.reset_mask)
        draw_mask_button.setMenu(draw_options_menu)

        func_options_button = QtWidgets.QPushButton("Func")
        func_options_button.setFixedSize(50, 20)
        func_options_menu = QtWidgets.QMenu()
        mirror_menu = QtWidgets.QMenu("Mirror", func_options_menu)
        mirror_menu.addAction("x-axis", functools.partial(self.mirror_mask, direction="x"))
        mirror_menu.addAction("y-axis", functools.partial(self.mirror_mask, direction="y"))
        func_options_menu.addMenu(mirror_menu)
        func_options_menu.addAction("Clear", self.clear_mask)
        func_options_button.setMenu(func_options_menu)

        self.mask_actors_group = CustomGroupBox("Mask", self)
        self.mask_actors_group.checkbox.setChecked(False)
        self.mask_actors_group.addButtonClicked.connect(
            lambda mask_path='', prompt=True: self.add_mask_file(mask_path, prompt))
        self.mask_actors_group.add_button_to_header(draw_mask_button)
        self.mask_actors_group.add_button_to_header(func_options_button)
        self.panel_layout.addWidget(self.mask_actors_group)

    def panel_background_control(self):
        # 1. 创建一个标准的 QGroupBox，就像 camera_control_console 一样
        self.background_group = QtWidgets.QGroupBox("Background Control")
        self.background_group.setObjectName("consoleGroupBox")  # 保持样式一致

        # 2. 创建一个主垂直布局来容纳标题栏和内容
        main_v_layout = QtWidgets.QVBoxLayout()
        main_v_layout.setContentsMargins(0, 5, 0, 5)  # 调整边距以匹配

        # --- 3. 创建标题栏布局 ---
        header_h_layout = QtWidgets.QHBoxLayout()
        header_h_layout.setContentsMargins(10, 0, 10, 5)

        # a) 添加复选框
        self.bg_show_checkbox = QtWidgets.QCheckBox("Show")
        self.bg_show_checkbox.setChecked(False)

        # b) 添加颜色选择按钮
        # self.bg_color_button = QtWidgets.QPushButton()
        self.bg_color_button = SquareButton()
        # self.background_plane_color 应该在 __init__ 中被初始化
        self.bg_color_button.setStyleSheet(f"background-color: {self.background_plane_color.name()}")

        self.bg_reset_button = QtWidgets.QPushButton()
        self.bg_reset_button.setFixedSize(20, 20)
        self.bg_reset_button.setText('Reset')

        header_h_layout.addWidget(self.bg_show_checkbox)
        header_h_layout.addStretch(1)  # 添加一个弹簧，将颜色按钮推到右边
        header_h_layout.addWidget(self.bg_reset_button)
        header_h_layout.addWidget(self.bg_color_button)

        # 将标题栏添加到主布局
        main_v_layout.addLayout(header_h_layout)

        # --- 4. 创建可隐藏的内容区域 ---
        # a) 创建一个 QWidget 作为所有控件的容器
        self.bg_content_widget = QtWidgets.QWidget()

        # b) 为这个容器创建一个网格布局
        bg_grid_layout = QtWidgets.QGridLayout()
        bg_grid_layout.setContentsMargins(10, 5, 10, 5)  # 设置合适的边距

        # c) 创建并添加所有 CameraControlWidget
        self.bg_rx_control = CameraControlWidget("Rx", "(deg)", 180)
        self.bg_ry_control = CameraControlWidget("Ry", "(deg)", 180)
        self.bg_rz_control = CameraControlWidget("Rz", "(deg)", 180)
        self.bg_tx_control = CameraControlWidget("Tx", "(mm)", 50000)
        self.bg_ty_control = CameraControlWidget("Ty", "(mm)", 50000)
        self.bg_distance_control = CameraControlWidget("Dis", "(mm)", 50000)
        self.bg_width_control = CameraControlWidget("W", "(mm)", 50000)
        self.bg_height_control = CameraControlWidget("H", "(mm)", 50000)

        self.bg_distance_control.spin_box.setValue(1000)
        self.bg_width_control.spin_box.setValue(500)
        self.bg_height_control.spin_box.setValue(500)

        bg_grid_layout.addWidget(self.bg_rx_control, 0, 0)
        bg_grid_layout.addWidget(self.bg_ry_control, 0, 1)
        bg_grid_layout.addWidget(self.bg_rz_control, 0, 2)
        bg_grid_layout.addWidget(self.bg_tx_control, 1, 0)
        bg_grid_layout.addWidget(self.bg_ty_control, 1, 1)
        bg_grid_layout.addWidget(self.bg_distance_control, 1, 2)
        bg_grid_layout.addWidget(self.bg_width_control, 2, 1)
        bg_grid_layout.addWidget(self.bg_height_control, 2, 2)

        # d) 将网格布局应用到内容容器 QWidget
        self.bg_content_widget.setLayout(bg_grid_layout)

        # e) 默认情况下隐藏内容
        self.bg_content_widget.setVisible(False)

        # f) 将内容容器添加到主布局
        main_v_layout.addWidget(self.bg_content_widget)

        # 5. 将主布局应用到 QGroupBox
        self.background_group.setLayout(main_v_layout)

        # 6. 将整个 GroupBox 添加到主面板
        self.panel_layout.addWidget(self.background_group)

        # --- 7. 连接所有信号 ---
        self.bg_show_checkbox.toggled.connect(self.on_bg_show_toggled)
        self.bg_color_button.clicked.connect(self.pick_background_color)
        self.bg_reset_button.clicked.connect(self.reset_background_controls)
        for control in [self.bg_rx_control, self.bg_ry_control, self.bg_rz_control, self.bg_tx_control,
                        self.bg_ty_control,
                        self.bg_distance_control, self.bg_width_control, self.bg_height_control]:
            control.spin_box.valueChanged.connect(self.update_background_plane)

    # Panel Output
    def panel_output(self):
        # Add a spacer to the top of the main layout
        self.output = QtWidgets.QGroupBox("Output")
        self.output.setObjectName("consoleGroupBox")
        output_layout = QtWidgets.QVBoxLayout()
        output_layout.setContentsMargins(10, 20, 10, 5)

        # Create the top widgets (layout)
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Create Grid layout for function buttons
        grid_layout = QtWidgets.QGridLayout()

        # Create a SearchBar for search bar
        self.search_bar = SearchBar()
        self.search_bar.setPlaceholderText("Search...")

        # Add a signal to the QLineEdit object to connect to a function
        self.search_bar.textChanged.connect(self.handle_search)
        self.search_bar.returnPressedSignal.connect(self.find_next)

        # Add the search bar to the layout
        grid_layout.addWidget(self.search_bar, 0, 0)

        # Create the set camera button
        copy_pixmap = QtGui.QPixmap(str(ICON_PATH / "copy.png"))
        copy_icon = QtGui.QIcon(copy_pixmap)
        copy_text_button = QtWidgets.QPushButton()
        copy_text_button.setIcon(copy_icon)  # Set copy icon
        copy_text_button.clicked.connect(self.copy_output_text)
        grid_layout.addWidget(copy_text_button, 0, 1)

        # Create the actor pose button
        reset_pixmap = QtGui.QPixmap(str(ICON_PATH / "reset.png"))
        reset_icon = QtGui.QIcon(reset_pixmap)
        reset_text_button = QtWidgets.QPushButton()
        reset_text_button.setIcon(reset_icon)  # Set reset icon
        reset_text_button.clicked.connect(self.reset_output_text)
        grid_layout.addWidget(reset_text_button, 0, 2)

        grid_widget = QtWidgets.QWidget()
        grid_widget.setLayout(grid_layout)
        top_layout.addWidget(grid_widget)
        output_layout.addLayout(top_layout)

        # Access to the system clipboard
        self.clipboard = QtGui.QGuiApplication.clipboard()
        self.output_text.setReadOnly(True)
        output_layout.addWidget(self.output_text)
        self.output.setLayout(output_layout)
        self.panel_layout.addWidget(self.output)

    def on_bg_show_toggled(self, checked):
        """当'Show'复选框状态改变时，显示/隐藏内容控件和3D平面。"""
        self.bg_content_widget.setVisible(checked)
        self.toggle_background_plane_visibility(checked)

    def pick_background_color(self):
        """打开颜色对话框，让用户选择颜色。"""
        color = QtWidgets.QColorDialog.getColor(self.background_plane_color, self)
        if color.isValid():
            self.background_plane_color = color
            self.bg_color_button.setStyleSheet(f"background-color: {self.background_plane_color.name()}")
            # 如果平面当前可见，则立即更新其颜色
            if self.background_plane_actor and self.background_plane_actor.visibility:
                self.background_plane_actor.prop.color = self.background_plane_color.name()

    def toggle_background_plane_visibility(self, visible):
        """当'Show Background'复选框状态改变时调用。"""
        if visible:
            # --- 修改：不再需要检查RGB图像 ---
            if self.background_plane_actor is None:
                plane = pv.Plane()
                self.background_plane_actor = self.plotter.add_mesh(
                    plane,
                    color=self.background_plane_color.name(),  # 使用纯色
                    opacity=0.5,
                    lighting=True,  # 让平面有光照，看起来更立体
                    name="_background_plane"
                )

            self.update_background_plane()
            self.background_plane_actor.visibility = True
        else:
            if self.background_plane_actor is not None:
                self.background_plane_actor.visibility = False

    def update_background_plane(self):
        """读取所有UI控件的值，并更新可视化平面。"""
        if self.background_plane_actor is None or not self.background_plane_actor.visibility:
            return

        rx = self.bg_rx_control.spin_box.value()
        ry = self.bg_ry_control.spin_box.value()
        rz = self.bg_rz_control.spin_box.value()
        tx = self.bg_tx_control.spin_box.value()
        ty = self.bg_ty_control.spin_box.value()
        distance = self.bg_distance_control.spin_box.value()
        width = self.bg_width_control.spin_box.value()
        height = self.bg_height_control.spin_box.value()

        new_plane = pv.Plane(
            i_size=width,
            j_size=height,
            i_resolution=1,
            j_resolution=1
        )
        self.background_plane_actor.mapper.dataset = new_plane

        # --- 新增：确保颜色也更新 ---
        self.background_plane_actor.prop.color = self.background_plane_color.name()

        transform_matrix = utils.compose_transform(np.array([rx, ry, rz]), np.array([tx, ty, distance]))
        self.background_plane_actor.user_matrix = transform_matrix

        self.plotter.render()

    def update_background_controls_from_actor(self):
        """
        从3D背景平面actor读取状态，并更新UI控件。
        这是 update_background_plane 的反向操作。
        """
        # 检查actor是否存在且可见
        if self.background_plane_actor is None or not self.background_plane_actor.visibility:
            return

        # 1. 从actor获取变换矩阵并分解
        matrix = self.background_plane_actor.user_matrix
        euler_angles, translation = utils.decompose_transform(matrix)

        # 2. 从actor的几何数据中获取大小
        #    一个pv.Plane的边界(bounds)可以告诉我们它的尺寸
        poly_data = self.background_plane_actor.mapper.dataset
        width = poly_data.bounds[1] - poly_data.bounds[0]
        height = poly_data.bounds[3] - poly_data.bounds[2]

        # 3. 使用 block_value_change_signal 安全地更新UI控件
        #    这可以防止触发无限循环的回调
        self.block_value_change_signal(self.bg_rx_control.spin_box, euler_angles[0])
        self.block_value_change_signal(self.bg_ry_control.spin_box, euler_angles[1])
        self.block_value_change_signal(self.bg_rz_control.spin_box, euler_angles[2])
        self.block_value_change_signal(self.bg_tx_control.spin_box, translation[0])
        self.block_value_change_signal(self.bg_ty_control.spin_box, translation[1])
        self.block_value_change_signal(self.bg_distance_control.spin_box, translation[2])
        self.block_value_change_signal(self.bg_width_control.spin_box, width)
        self.block_value_change_signal(self.bg_height_control.spin_box, height)

    def reset_background_controls(self):
        """将背景控制面板的所有控件重置为默认值。"""
        # 使用 block_value_change_signal 避免触发不必要的回调
        self.block_value_change_signal(self.bg_rx_control.spin_box, 0)
        self.block_value_change_signal(self.bg_ry_control.spin_box, 0)
        self.block_value_change_signal(self.bg_rz_control.spin_box, 0)
        self.block_value_change_signal(self.bg_tx_control.spin_box, 0)
        self.block_value_change_signal(self.bg_ty_control.spin_box, 0)
        self.block_value_change_signal(self.bg_distance_control.spin_box, 1000)
        self.block_value_change_signal(self.bg_width_control.spin_box, 1000)
        self.block_value_change_signal(self.bg_height_control.spin_box, 1000)

        # 重置颜色
        self.background_plane_color = QtGui.QColor('#808080')
        self.bg_color_button.setStyleSheet(f"background-color: {self.background_plane_color.name()}")

        # 取消勾选并隐藏3D平面
        self.bg_show_checkbox.setChecked(False)

    def toggle_group_content(self, group, checked):
        for child in group.findChildren(QtWidgets.QWidget):
            child.setVisible(checked)
        self.panel_layout.update()

    def handle_search(self, text):
        # If there's text in the search bar
        if text:
            self.highlight_keyword(text)
        # If the search bar is empty
        else:
            self.clear_highlight()

    def highlight_keyword(self, keyword):
        # Get QTextDocument from QTextEdit
        doc = self.output_text.document()

        # Default text format
        default_format = QtGui.QTextCharFormat()

        # Text format for highlighted words
        highlight_format = QtGui.QTextCharFormat()
        highlight_format.setBackground(QtGui.QBrush(QtGui.QColor("yellow")))
        highlight_format.setForeground(QtGui.QBrush(QtGui.QColor("black")))

        # Clear all previous highlights
        cursor = QtGui.QTextCursor(doc)
        cursor.beginEditBlock()
        block_format = cursor.blockFormat()
        cursor.select(QtGui.QTextCursor.Document)
        cursor.setBlockFormat(block_format)
        cursor.setCharFormat(default_format)
        cursor.clearSelection()
        cursor.endEditBlock()
        cursor.setPosition(0)

        # Loop through each occurrence of the keyword
        occurrence_found = False
        while not cursor.isNull() and not cursor.atEnd():
            cursor = doc.find(keyword, cursor)
            if not cursor.isNull():
                if not occurrence_found:
                    self.output_text.setTextCursor(cursor)
                    occurrence_found = True
                cursor.mergeCharFormat(highlight_format)

        if not occurrence_found:
            cursor.setPosition(0)
            self.output_text.setTextCursor(cursor)  # Set QTextEdit cursor to the beginning if no match found

    def find_next(self):
        keyword = self.search_bar.text()
        # Get the QTextCursor from the QTextEdit
        cursor = self.output_text.textCursor()
        # Move the cursor to the position after the current selection
        cursor.setPosition(cursor.position() + cursor.selectionEnd() - cursor.selectionStart())
        # Use the QTextDocument's find method to find the next occurrence
        found_cursor = self.output_text.document().find(keyword, cursor)
        if not found_cursor.isNull():
            self.output_text.setTextCursor(found_cursor)

    def clear_highlight(self):
        doc = self.output_text.document()
        default_format = QtGui.QTextCharFormat()
        cursor = QtGui.QTextCursor(doc)
        cursor.beginEditBlock()
        block_format = cursor.blockFormat()
        cursor.select(QtGui.QTextCursor.Document)
        cursor.setBlockFormat(block_format)
        cursor.setCharFormat(default_format)
        cursor.clearSelection()
        cursor.endEditBlock()
        cursor.setPosition(0)
        self.output_text.setTextCursor(cursor)  # Set QTextEdit cursor to the beginning

    # ^ Plotter
    def create_plotter(self):
        self.frame = QtWidgets.QFrame()
        self.plotter = CustomQtInteractor(self.frame, self)
        self.plotter.pose_changed_by_drag.connect(self.on_pose_drag_finished)
        self.plotter.pose_changed_by_drag.connect(self.update_background_controls_from_actor)
        self.signal_close.connect(self.plotter.close)

    def show_plot(self):
        self.plotter.enable_trackball_actor_style()
        self.plotter.add_axes(color='white')
        self.plotter.add_camera_orientation_widget()
        self.plotter.show()
        self.show()

    def on_pose_drag_finished(self):
        self.is_workspace_dirty = True

    def check_image_button(self, name):
        button = next((btn for btn in self.image_button_group_actors.buttons() if btn.text() == name), None)
        if button: button.click()

    def block_value_change_signal(self, spinbox, value):
        spinbox.blockSignals(True)
        spinbox.setValue(value)
        spinbox.blockSignals(False)

    def set_camera_control_values(self, matrix):
        euler_angles, translation = utils.decompose_transform(matrix)
        self.block_value_change_signal(self.camera_rx_control.spin_box, euler_angles[0])
        self.block_value_change_signal(self.camera_ry_control.spin_box, euler_angles[1])
        self.block_value_change_signal(self.camera_rz_control.spin_box, euler_angles[2])
        self.block_value_change_signal(self.camera_tx_control.spin_box, translation[0])
        self.block_value_change_signal(self.camera_ty_control.spin_box, translation[1])
        self.block_value_change_signal(self.camera_tz_control.spin_box, translation[2])

    def check_mesh_button(self, index, output_text):
        if not (0 <= index < len(self.scene.mesh_container.meshes)):
            return
        mesh_model = self.scene.mesh_container.get_mesh_by_index(index)
        name = mesh_model.name
        # button = next((btn for btn in self.mesh_button_group_actors.buttons() if btn.text() == name), None)
        button = self.mesh_button_group_actors.button(index)
        if button:
            button.setChecked(True)
            self.scene.handle_mesh_click(index, output_text=output_text)
            self.set_camera_control_values(utils.get_actor_user_matrix(
                self.scene.mesh_container.get_mesh_by_index(self.scene.mesh_container.reference)))
            self.on_link_mesh_button_toggle(checked=self.link_mesh_button.isChecked(), clicked=False)

    def check_mask_button(self, name):
        button = next((btn for btn in self.mask_button_group_actors.buttons() if btn.text() == name), None)
        if button: button.click()

    @utils.require_attributes([('scene.image_container.reference', "Please load an image first!")])
    def mirror_image(self, direction):
        image_model = self.scene.image_container.images[self.scene.image_container.reference]
        if direction == 'x':
            image_model.source_obj = image_model.source_obj[:, ::-1, :]
        elif direction == 'y':
            image_model.source_obj = image_model.source_obj[::-1, :, :]
        self.check_image_button(image_model.name)

    @utils.require_attributes([('scene.mesh_container.reference', "Please load a mesh first!")])
    def mirror_mesh(self, direction):
        mesh_model = self.scene.mesh_container.get_mesh_by_index(self.scene.mesh_container.reference)
        if (mesh_model.initial_pose != np.eye(4)).all():
            mesh_model.initial_pose = mesh_model.actor.user_matrix
        transformation_matrix = mesh_model.actor.user_matrix
        if direction == 'x':
            transformation_matrix = np.array(
                [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
        elif direction == 'y':
            transformation_matrix = np.array(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
        mesh_model.actor.user_matrix = transformation_matrix
        self.check_mesh_button(mesh_model.name, output_text=True)
        self.is_workspace_dirty = True

    @utils.require_attributes([('scene.mask_container.reference', "Please load a mask first!")])
    def mirror_mask(self, direction):
        mask_model = self.scene.mask_container.masks[self.scene.mask_container.reference]
        transformation_matrix = mask_model.actor.user_matrix
        if direction == 'x':
            transformation_matrix = np.array(
                [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
        elif direction == 'y':
            transformation_matrix = np.array(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ transformation_matrix
        mask_model.actor.user_matrix = transformation_matrix
        self.check_mask_button(name=mask_model.name)
        self.is_workspace_dirty = True

    def add_mask_button(self, name):
        button_widget = CustomMaskButtonWidget(name)
        button_widget.colorChanged.connect(lambda color, name=name: self.scene.mask_color_value_change(name, color))
        button_widget.removeButtonClicked.connect(self.remove_mask_button)
        button = button_widget.button
        mask_model = self.scene.mask_container.masks[name]
        mask_model.opacity_spinbox = button_widget.double_spinbox
        mask_model.opacity_spinbox.setValue(mask_model.opacity)
        mask_model.opacity_spinbox.valueChanged.connect(
            lambda value, name=name: self.scene.mask_container.set_mask_opacity(name, value))
        mask_model.color_button = button_widget.square_button
        mask_model.color_button.setStyleSheet(f"background-color: {mask_model.color}")
        # check the button
        button.setCheckable(True)
        button.setChecked(False)
        button.clicked.connect(lambda _, name=name: self.handle_mask_click(name))
        self.mask_actors_group.widget_layout.insertWidget(0, button_widget)
        self.mask_button_group_actors.addButton(button)
        self.check_mask_button(name=name)

    def handle_mask_click(self, name):
        self.scene.mask_container.reference = name
        self.reset_camera()

    @utils.require_attributes([('scene.image_container.reference', "Please load an image first!")])
    def draw_mask(self, live_wire=False, sam=False):
        def handle_output_path_change(output_path):
            if output_path:
                self.scene.mask_container.add_mask(mask_source=output_path,
                                                   fy=self.scene.fy,
                                                   cx=self.scene.cx,
                                                   cy=self.scene.cy)
                self.add_mask_button(self.scene.mask_container.reference)

        image = utils.get_image_actor_scalars(
            self.scene.image_container.images[self.scene.image_container.reference].actor)
        if sam:
            self.mask_window = SamWindow(image)
        elif live_wire:
            self.mask_window = LiveWireWindow(image)
        else:
            self.mask_window = MaskWindow(image)
        self.mask_window.mask_label.output_path_changed.connect(handle_output_path_change)

    def copy_output_text(self):
        self.clipboard.setText(self.output_text.toPlainText())

    def reset_output_text(self):
        self.output_text.clear()

    """
    Copyright (C) 2025 Hanagumori-B <yjp.y@foxmail.com>
    To load and save workspaces in Json format.
    """

    def load_workspace_from_path(self, workspace_path=''):
        if workspace_path:
            self.clear_plot()  # clear out everything before loading a workspace
            self.current_workspace_index = self.workspaces.index(
                workspace_path) if workspace_path in self.workspaces else -1
            with open(str(self.workspace_path), 'r') as f:
                try:
                    workspace = json.load(f)
                except json.decoder.JSONDecodeError as e:
                    self.output_text.append(e.msg)
                    return False

            try:
                if 'image' in workspace and workspace['image'] is not None:
                    images = workspace['image']
                    for item in images:
                        image = images[item]
                        self.add_image_file(image_paths=[SAVE_ROOT / pathlib.Path(image)])
                # if 'mask_path' in workspace and workspace['mask_path'] is not None:
                #     masks = workspace['mask_path']
                #     for item in masks:
                #         mask = masks[item]
                #         self.add_mask_file(mask_paths=[SAVE_ROOT / pathlib.Path(mask)])
                if 'meshes' in workspace:
                    meshes = workspace['meshes']
                    for index in meshes:
                        mesh = meshes[index]
                        mesh_name = mesh['mesh_name']
                        file_path = mesh['file_path']
                        pose = mesh['pose']
                        self.add_mesh_file(mesh_paths=[SAVE_ROOT / pathlib.Path(file_path)])
                        self.add_pose_file(pose)
                if 'background' in workspace:
                    settings = workspace["background"]
                    (rx, ry, rz), (tx, ty, tz) = utils.decompose_transform(np.array(settings.get('matrix')))
                    self.block_value_change_signal(self.bg_rx_control.spin_box, rx)
                    self.block_value_change_signal(self.bg_ry_control.spin_box, ry)
                    self.block_value_change_signal(self.bg_rz_control.spin_box, rz)
                    self.block_value_change_signal(self.bg_tx_control.spin_box, tx)
                    self.block_value_change_signal(self.bg_ty_control.spin_box, ty)
                    self.block_value_change_signal(self.bg_distance_control.spin_box, tz)
                    self.block_value_change_signal(self.bg_width_control.spin_box, settings.get("width", 1000))
                    self.block_value_change_signal(self.bg_height_control.spin_box, settings.get("height", 1000))
                    self.background_plane_color = QtGui.QColor(settings.get("color", '#808080'))
                    self.bg_color_button.setStyleSheet(f"background-color: {self.background_plane_color.name()}")
                    self.bg_show_checkbox.setChecked(True)
                else:
                    self.reset_background_controls()
                self.update_background_plane()

                if 'camera' in workspace:
                    camera_settings = workspace['camera']
                    self.scene.fx = camera_settings.get("fx", 572.4114)
                    self.scene.fy = camera_settings.get("fy", 573.57043)
                    self.scene.cx = camera_settings.get("cx", 325.2611)
                    self.scene.cy = camera_settings.get("cy", 242.048990)
                    self.scene.canvas_height = camera_settings.get("canvas_height", 480)
                    self.scene.canvas_width = camera_settings.get("canvas_width", 640)
                    self.scene.cam_viewup = tuple(camera_settings.get("cam_viewup", (0, -1, 0)))
                    self.scene.set_camera_intrinsics(self.scene.fx, self.scene.fy, self.scene.cx, self.scene.cy,
                                                     self.scene.canvas_height)
                    self.scene.set_camera_extrinsics(self.scene.cam_viewup)
                    if self.scene.image_container.reference is not None:
                        self.scene.handle_image_click(self.scene.image_container.reference)
            except Exception as e:
                return False

            self.reset_camera()
            self.is_workspace_dirty = False
            return True
        return False

    def add_workspace_file(self, workspace_paths: list = None, prompt=False):
        if prompt:
            workspace_paths, _ = QtWidgets.QFileDialog().getOpenFileNames(None, "Open file", "", "Files (*.json)")

        if workspace_paths:
            for path in workspace_paths:
                if path not in self.workspaces:
                    self.workspaces.append(path)
                    index = len(self.workspaces) - 1

                    button = QtWidgets.QPushButton(path)
                    button.setCheckable(True)
                    button.clicked.connect(lambda _, i=index: self.switch_workspace(i))

                    button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                    button.customContextMenuRequested.connect(self.show_workspace_context_menu)

                    self.workspace_button_group_actors.addButton(button)
                    self.workspaces_group.widget_layout.addWidget(button)

            # If this is the first workspace being added, load it automatically
            if self.current_workspace_index == -1 and self.workspaces:
                self.switch_workspace(0)

    def show_workspace_context_menu(self, pos):
        button = self.sender()
        if not isinstance(button, QtWidgets.QPushButton):
            return

        menu = QtWidgets.QMenu()
        remove_action = menu.addAction("Remove")
        remove_action.triggered.connect(lambda: self.remove_workspace_button(button))

        menu.exec_(button.mapToGlobal(pos))

    def new_workspace_file(self):
        workspace_dir = pathlib.Path(self.workspace_path).parent
        workspace_path, _ = QtWidgets.QFileDialog().getSaveFileName(
            QtWidgets.QMainWindow(), "Save File", str(workspace_dir), "Mesh Files (*.json);;All Files (*)")
        if workspace_path != "":
            with open(workspace_path, 'w') as f:
                pass
        else:
            return
        self.add_workspace_file([workspace_path])

    def switch_workspace(self, index):
        if not (0 <= index < len(self.workspaces)):
            return

        if self.current_workspace_index == index:
            return  # Do nothing if already active

        if self.is_workspace_dirty:
            dialog = SaveChangesDialog(self, pathlib.Path(self.workspace_path).name)
            result = dialog.exec_()

            if result == SaveChangesDialog.Save:
                self.save_current_workspace()
            elif result == SaveChangesDialog.Cancel:
                buttons = self.workspace_button_group_actors.buttons()
                if 0 <= self.current_workspace_index < len(buttons):
                    buttons[self.current_workspace_index].setChecked(True)
                return

        self.current_workspace_index = index

        # Update UI button states
        buttons = self.workspace_button_group_actors.buttons()
        if 0 <= index < len(buttons):
            buttons[index].setChecked(True)

        # Get path and load the workspace
        path = self.workspaces[index]

        if not self.load_workspace_from_path(path):
            self.output_text.append(f"\n-> A empty workspace: {pathlib.Path(path).name}")
        self.output_text.append(f"\n-> Switched to workspace: {pathlib.Path(path).name}")

    @utils.require_attributes([('scene.image_container.reference', "Please load an image first!")])
    def export_workspace(self, default_directory=''):
        workspace_dict = {"meshes": {}, "image": {}}
        for index, mesh_model in enumerate(self.scene.mesh_container.meshes):
            mesh_info = {
                "mesh_name": mesh_model.name,
                "file_path": mesh_model.path,
                "pose": utils.get_actor_user_matrix(mesh_model).tolist()
            }
            workspace_dict["meshes"][index] = mesh_info
        # for mesh_model in self.scene.mesh_container.meshes.values():
        #     matrix = utils.get_actor_user_matrix(mesh_model)
        #     workspace_dict["mesh_path"][mesh_model.name] = (mesh_model.path, matrix.tolist())
        for image_model in self.scene.image_container.images.values():
            workspace_dict["image"][image_model.name] = image_model.path
        # for mask_model in self.scene.mask_container.masks.values():
        #     workspace_dict["mask_path"][mask_model.name] = mask_model.path

        if self.background_plane_actor:
            background_poly_data = self.background_plane_actor.mapper.dataset
            background_width = background_poly_data.bounds[1] - background_poly_data.bounds[0]
            background_height = background_poly_data.bounds[3] - background_poly_data.bounds[2]
            workspace_dict["background"] = {'matrix': self.background_plane_actor.user_matrix.tolist(),
                                            'color': self.background_plane_color.name(),
                                            'width': background_width,
                                            'height': background_height}
        camera_settings = {
            "fx": self.scene.fx,
            "fy": self.scene.fy,
            "cx": self.scene.cx,
            "cy": self.scene.cy,
            "canvas_height": self.scene.canvas_height,
            "canvas_width": self.scene.canvas_width,
            "cam_viewup": list(self.scene.cam_viewup)
        }
        workspace_dict["camera"] = camera_settings
        # write the dict to json file
        output_path, _ = QtWidgets.QFileDialog.getSaveFileName(QtWidgets.QMainWindow(), "Save File",
                                                               default_directory, "Mesh Files (*.json)")
        if output_path != "":
            with open(output_path, 'w') as f: json.dump(workspace_dict, f, indent=4)

    def save_current_workspace(self):
        if self.current_workspace_index == -1:
            return  # No active workspace to save

        workspace_path = self.workspaces[self.current_workspace_index]

        # Update the poses for meshes present in the scene
        workspace_dict = {"meshes": {}, "image": {}}
        for index, mesh_model in enumerate(self.scene.mesh_container.meshes):
            mesh_info = {
                "mesh_name": mesh_model.name,
                "file_path": mesh_model.path,
                "pose": utils.get_actor_user_matrix(mesh_model).tolist()
            }
            workspace_dict["meshes"][index] = mesh_info
        for image_model in self.scene.image_container.images.values():
            workspace_dict["image"][image_model.name] = image_model.path

        if self.background_plane_actor:
            background_poly_data = self.background_plane_actor.mapper.dataset
            background_width = background_poly_data.bounds[1] - background_poly_data.bounds[0]
            background_height = background_poly_data.bounds[3] - background_poly_data.bounds[2]
            workspace_dict["background"] = {'matrix': self.background_plane_actor.user_matrix.tolist(),
                                            'color': self.background_plane_color.name(),
                                            'width': background_width,
                                            'height': background_height}
        camera_settings = {
            "fx": self.scene.fx,
            "fy": self.scene.fy,
            "cx": self.scene.cx,
            "cy": self.scene.cy,
            "canvas_height": self.scene.canvas_height,
            "canvas_width": self.scene.canvas_width,
            "cam_viewup": list(self.scene.cam_viewup)
        }
        workspace_dict["camera"] = camera_settings

        # Write the updated data back to the file
        try:
            with open(workspace_path, 'w') as f:
                json.dump(workspace_dict, f, indent=4)
            self.output_text.append(f"-> Workspace '{pathlib.Path(workspace_path).name}' saved.")
            self.is_workspace_dirty = False  # Reset dirty flag after saving
        except Exception as e:
            self.output_text.append(f"-> Error saving workspace: {e}")

    def remove_workspace_button(self, button, dialog=True):
        try:
            # 找到按钮及其在列表中的索引
            index = self.workspace_button_group_actors.buttons().index(button)
        except ValueError:
            return  # 如果找不到按钮，则不执行任何操作

        if self.is_workspace_dirty and dialog:
            dialog = SaveChangesDialog(self, pathlib.Path(self.workspaces[index]).name, 'Remove Workspace')
            result = dialog.exec_()

            if result == SaveChangesDialog.Save:
                self.save_current_workspace()
            elif result == SaveChangesDialog.Cancel:
                buttons = self.workspace_button_group_actors.buttons()
                if 0 <= self.current_workspace_index < len(buttons):
                    buttons[self.current_workspace_index].setChecked(True)
                return

            # 核心逻辑：如果被移除的是当前活动的工作区，则清空场景
        if index == self.current_workspace_index:
            self.clear_plot()
            self.current_workspace_index = -1
            # 如果被移除的工作区在当前活动工作区之前，则更新索引
        elif index < self.current_workspace_index:
            self.current_workspace_index -= 1

            # 从数据模型和UI中移除
        del self.workspaces[index]
        self.workspace_button_group_actors.removeButton(button)
        button.deleteLater()

        # 更新剩余按钮的信号连接，因为它们的索引已经改变
        for i, btn in enumerate(self.workspace_button_group_actors.buttons()):
            try:
                btn.clicked.disconnect()
            except TypeError:
                pass
            # 重新连接，并赋予新的正确索引
            btn.clicked.connect(lambda _, new_index=i: self.switch_workspace(new_index))
        self.is_workspace_dirty = False

    """
    Part of workspaces ends.
    """

    def remove_image_button(self, button):
        # Get the index of the button before removal
        buttons = self.image_button_group_actors.buttons()
        checked_button = self.image_button_group_actors.checkedButton()
        index = buttons.index(checked_button)

        # Remove the associated actor
        actor = self.scene.image_container.images[button.text()].actor
        self.plotter.remove_actor(actor)
        del self.scene.image_container.images[button.text()]
        self.image_button_group_actors.removeButton(button)
        self.remove_image_button_widget(button)
        self.scene.image_container.reference = None
        button.deleteLater()
        buttons = self.image_button_group_actors.buttons()
        if buttons:
            if index < len(buttons):
                next_button = buttons[index]
                next_button.click()
            else:
                buttons[-1].click()

    def remove_mask_button(self, button):
        # Get the index of the button before removal
        buttons = self.mask_button_group_actors.buttons()
        checked_button = self.mask_button_group_actors.checkedButton()
        index = buttons.index(checked_button)

        actor = self.scene.mask_container.masks[button.text()].actor
        self.plotter.remove_actor(actor)
        del self.scene.mask_container.masks[button.text()]
        self.mask_button_group_actors.removeButton(button)
        self.remove_mask_button_widget(button)
        self.scene.mask_container.reference = None
        button.deleteLater()
        buttons = self.mask_button_group_actors.buttons()
        if buttons:
            if index < len(buttons):
                next_button = buttons[index]
                next_button.click()
            else:
                buttons[-1].click()

    def remove_mesh_button(self, button):
        # Get the index of the button before removal
        buttons = self.mesh_button_group_actors.buttons()
        checked_button = self.mesh_button_group_actors.checkedButton()
        index = buttons.index(checked_button)

        # Remove the associated actor
        actor = self.scene.mesh_container.get_mesh_by_index(index).actor
        self.plotter.remove_actor(actor)
        self.scene.mesh_container.meshes.pop(index)
        self.mesh_button_group_actors.removeButton(button)
        self.remove_mesh_button_widget(button)
        self.scene.mesh_container.reference = None
        button.deleteLater()
        for i, btn in enumerate(self.mesh_button_group_actors.buttons()):
            self.mesh_button_group_actors.setId(btn, i)
        buttons = self.mesh_button_group_actors.buttons()
        if buttons:
            if index < len(buttons):
                next_button = buttons[index]
                next_button.click()
            else:
                buttons[-1].click()
        else:
            self.set_camera_spinbox(indicator=False)

    def clear_workspaces(self):
        # 从后往前移除，以避免在循环中出现索引问题
        dialog = SaveChangesDialog(self, 'all', 'Remove ALL Workspace')
        result = dialog.exec_()

        if result == SaveChangesDialog.Save:
            for button in reversed(self.workspace_button_group_actors.buttons()):
                button.click()
                self.save_current_workspace()
        elif result == SaveChangesDialog.Cancel:
            buttons = self.workspace_button_group_actors.buttons()
            if 0 <= self.current_workspace_index < len(buttons):
                buttons[self.current_workspace_index].setChecked(True)
            return

        for button in reversed(self.workspace_button_group_actors.buttons()):
            self.remove_workspace_button(button, False)
        self.clear_plot()
        self.is_workspace_dirty = False

    def clear_image(self):
        for button in self.image_button_group_actors.buttons(): self.remove_image_button(button)

    def clear_mesh(self):
        for button in self.mesh_button_group_actors.buttons(): self.remove_mesh_button(button)
        self.link_mesh_button.setChecked(False)
        self.is_workspace_dirty = True

    def clear_mask(self):
        for button in self.mask_button_group_actors.buttons(): self.remove_mask_button(button)

    def clear_plot(self):
        self.clear_image()
        self.clear_mesh()
        self.clear_mask()
        self.reset_output_text()
        self.is_workspace_dirty = True

    def remove_image_button_widget(self, button):
        for i in range(self.images_actors_group.widget_layout.count()):
            widget = self.images_actors_group.widget_layout.itemAt(i).widget()
            if widget is not None and hasattr(widget, 'button') and widget.button == button:
                self.images_actors_group.widget_layout.removeWidget(widget)
                widget.deleteLater()
                break

    def remove_mask_button_widget(self, button):
        for i in range(self.mask_actors_group.widget_layout.count()):
            widget = self.mask_actors_group.widget_layout.itemAt(i).widget()
            if widget is not None and hasattr(widget, 'button') and widget.button == button:
                self.mask_actors_group.widget_layout.removeWidget(widget)
                widget.deleteLater()
                break

    def remove_mesh_button_widget(self, button):
        for i in range(self.mesh_actors_group.widget_layout.count()):
            widget = self.mesh_actors_group.widget_layout.itemAt(i).widget()
            if widget is not None and hasattr(widget, 'button') and widget.button == button:
                self.mesh_actors_group.widget_layout.removeWidget(widget)
                widget.deleteLater()
                break

    def colored_viewer(self):
        viewer = DepthImageWindow(self)
        viewer.show()

    @utils.require_attributes([('scene.image_container.reference', "Please load an image first!")])
    def set_distance2camera(self):
        image_model = self.scene.image_container.images[self.scene.image_container.reference]
        dialog = DistanceInputDialog(title='Input', label='Set Objects distance to camera:',
                                     value=str(image_model.distance2camera), default_value=str(self.scene.fy))
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            distance = float(dialog.get_value())
            if self.scene.image_container.reference is not None:
                image_model = self.scene.image_container.images[self.scene.image_container.reference]
                image_model.distance2camera = distance
                pv_obj = pv.ImageData(dimensions=(image_model.width, image_model.height, 1), spacing=[1, 1, 1],
                                      origin=(0.0, 0.0, 0.0))
                pv_obj.point_data["values"] = image_model.source_obj.reshape(
                    (image_model.width * image_model.height, image_model.channel))  # order = 'C
                pv_obj = pv_obj.translate(-1 * np.array(pv_obj.center), inplace=False)  # center the image at (0, 0)
                pv_obj = pv_obj.translate(-np.array([0, 0, pv_obj.center[-1]]),
                                          inplace=False)  # very important, re-center it to [0, 0, 0]
                pv_obj = pv_obj.translate(np.array([0, 0, image_model.distance2camera]), inplace=False)
                if image_model.channel == 1:
                    image_actor = self.plotter.add_mesh(pv_obj, cmap='gray', opacity=image_model.opacity,
                                                        name=image_model.name)
                else:
                    image_actor = self.plotter.add_mesh(pv_obj, rgb=True, opacity=image_model.opacity, pickable=False,
                                                        name=image_model.name)
                image_model.actor = image_actor
                self.scene.image_container.images[image_model.name] = image_model
            if len(self.scene.mask_container.masks) > 0:
                for mask_name, mask_model in self.scene.mask_container.masks.items():
                    mask_model = self.scene.mask_container.masks[mask_name]
                    mask_model.pv_obj = mask_model.pv_obj.translate(-np.array([0, 0, mask_model.pv_obj.center[-1]]),
                                                                    inplace=False)  # very important, re-center it to [0, 0, 0]
                    mask_model.pv_obj = mask_model.pv_obj.translate(np.array([0, 0, distance]), inplace=False)
                    mask_mesh = self.plotter.add_mesh(mask_model.pv_obj, color=mask_model.color, style='surface',
                                                      opacity=mask_model.opacity, pickable=True, name=mask_name)
                    mask_model.actor = mask_mesh
                    self.scene.mask_container.masks[mask_name] = mask_model
            self.reset_camera()

    @utils.require_attributes([('scene.mesh_container.reference', "Please load a mesh first!")])
    def export_pose(self):
        output_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose a folder to save BOP images.",
                                                                str(SAVE_ROOT))
        if not output_dir:
            return
        save_root = pathlib.Path(output_dir)
        os.makedirs(save_root / "export_pose", exist_ok=True)
        for mesh_name, mesh_model in self.scene.mesh_container.meshes.items():
            matrix = utils.get_actor_user_matrix(mesh_model)
            output_path = save_root / "export_pose" / (mesh_name + '.npy')
            np.save(output_path, matrix)
            self.output_text.append(f"Export {mesh_name} mesh pose to:\n {output_path}")

    @utils.require_attributes([('scene.mesh_container.reference', "Please load a mesh first!")])
    def export_mesh_render(self, save_render=True):
        output_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose a folder to save BOP images.",
                                                                str(SAVE_ROOT))
        if not output_dir:
            return None
        save_root = pathlib.Path(output_dir)
        os.makedirs(save_root / "export_mesh_render", exist_ok=True)
        image = self.scene.mesh_container.render_mesh(name=self.scene.mesh_container.reference,
                                                      camera=self.plotter.camera.copy(), width=self.scene.canvas_width,
                                                      height=self.scene.canvas_height)
        if save_render:
            output_name = "export_" + self.scene.mesh_container.reference
            output_path = save_root / "export_mesh_render" / (output_name + '.png')
            while output_path.exists():
                output_name += "_copy"
                output_path = save_root / "export_mesh_render" / (output_name + ".png")
            rendered_image = PIL.Image.fromarray(image)
            rendered_image.save(output_path)
            self.output_text.append(f"-> Export mesh render to:\n {output_path}")
            return image
        return None

    @utils.require_attributes([('scene.mask_container.reference', "Please load a mask first!")])
    def export_mask(self):
        output_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose a folder to save BOP images.",
                                                                str(SAVE_ROOT))
        if not output_dir:
            return
        save_root = pathlib.Path(output_dir)
        os.makedirs(save_root / "export_mask", exist_ok=True)
        mask_model = self.scene.mask_container.masks[self.scene.mask_container.reference]
        output_name = "export_" + self.scene.mask_container.reference
        output_path = save_root / "export_mask" / (output_name + ".png")
        while output_path.exists():
            output_name += "_copy"
            output_path = save_root / "export_mask" / (output_name + ".png")
        # Update and store the transformed mask actor if there is any transformation
        self.scene.mask_container.update_mask(self.scene.mask_container.reference)
        image = self.scene.mask_container.render_mask(camera=self.plotter.camera.copy(), cx=self.scene.cx,
                                                      cy=self.scene.cy)
        rendered_image = PIL.Image.fromarray(image)
        rendered_image.save(output_path)
        mask_model.path = output_path
        self.output_text.append(f"-> Export Mask render to:\n {output_path}")

    @utils.require_attributes([('scene.image_container.reference', "Please load an image first!")])
    def export_image(self):
        output_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose a folder to save BOP images.",
                                                                str(SAVE_ROOT))
        if not output_dir:
            return
        save_root = pathlib.Path(output_dir)
        os.makedirs(save_root / "export_image", exist_ok=True)
        image_rendered = self.scene.image_container.render_image(camera=self.plotter.camera.copy())
        rendered_image = PIL.Image.fromarray(image_rendered)
        output_name = "export_" + self.scene.image_container.reference
        output_path = save_root / "export_image" / (output_name + '.png')
        while output_path.exists():
            output_name += "_copy"
            output_path = save_root / "export_image" / (output_name + ".png")
        rendered_image.save(output_path)
        self.output_text.append(f"-> Export image render to:\n {output_path}")

    def export_camera_info(self):
        output_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose a folder to save BOP images.",
                                                                str(SAVE_ROOT))
        if not output_dir:
            return
        save_root = pathlib.Path(output_dir)
        os.makedirs(save_root / "export_camera_info", exist_ok=True)
        output_path = save_root / "export_camera_info" / "camera_info.pkl"
        camera_intrinsics = np.array([[self.scene.fx, 0, self.scene.cx], [0, self.scene.fy, self.scene.cy], [0, 0, 1]],
                                     dtype=np.float32)
        camera_info = {'camera_intrinsics': camera_intrinsics, 'canvas_height': self.scene.canvas_height,
                       'canvas_width': self.scene.canvas_width}
        with open(output_path, "wb") as f: pickle.dump(camera_info, f)
        self.output_text.append(f"-> Export camera info to:\n {output_path}")

    """
    Copyright (C) 2025 Hanagumori-B <yjp.y@foxmail.com>
    To export a dataset in BOP format.
    """

    @utils.require_attributes([('scene.image_container.reference', "Please load an image first!"),
                               ('scene.mesh_container.meshes', "Please load a mesh first!")])
    def generate_bop_images(self, save_dirs: dict):
        depth_dir = save_dirs['depth_dir']
        mask_dir = save_dirs['mask_dir']
        mask_visib_dir = save_dirs['mask_visib_dir']
        rgb_dir = save_dirs['rgb_dir']

        image_id = int("".join([c for c in list(self.scene.image_container.images.keys())[0] if c.isnumeric()]))

        # 生成并保存深度图 (Depth Image)----------------------------------------------------------------------------------
        offscreen_plotter = pv.Plotter(off_screen=True, window_size=[self.scene.canvas_width,
                                                                     self.scene.canvas_height])  # 渲染器的窗口大小必须和相机画布大小一致
        offscreen_plotter.camera = self.plotter.camera.copy()  # 复制主窗口的相机状态，确保虚拟相机和用户看到的完全一样
        offscreen_plotter.render_window.SetMultiSamples(0)
        offscreen_plotter.render_window.SetLineSmoothing(False)
        offscreen_plotter.render_window.SetPolygonSmoothing(False)
        models_with_id = []
        for index, model in enumerate(self.scene.mesh_container.meshes):
            name = model.name
            name = "".join([c for c in name if c.isnumeric()])
            models_with_id.append({'name': name, 'model': model, 'id': index + 1})

        offscreen_plotter.clear()

        # 添加背景平面
        # 加载RGB图像作为纹理
        rgb_image_model = list(self.scene.image_container.images.values())[0]
        rgb_image_texture = pv.Texture(rgb_image_model.source_obj)
        # --- 核心修改：使用UI控件的值来创建背景 ---
        # if self.background_group.checkbox.isChecked():
        # 1. 读取UI值
        rx = self.bg_rx_control.spin_box.value()
        ry = self.bg_ry_control.spin_box.value()
        rz = self.bg_rz_control.spin_box.value()
        distance = self.bg_distance_control.spin_box.value()
        width = self.bg_width_control.spin_box.value()
        height = self.bg_height_control.spin_box.value()

        # 2. 创建变换矩阵和平面
        transform_matrix = utils.compose_transform(np.array([rx, ry, rz]), np.array([0, 0, distance]))

        background_plane_geom = pv.Plane(
            i_size=width, j_size=height, i_resolution=1, j_resolution=1)

        # 3. 将平面添加到离屏渲染器并应用变换
        bg_actor = offscreen_plotter.add_mesh(background_plane_geom, color='white', ambient=1.0)
        bg_actor.user_matrix = transform_matrix
        # --- 背景修改结束 ---

        # 创建一个足够大的平面
        # 平面的大小和位置需要根据您的相机内参来精确计算，
        # 一个简化的方法是创建一个比视野稍大的平面。
        # focal_length = (self.scene.fx + self.scene.fy) / 2
        # background_distance = 50000
        # plane_height = (self.scene.canvas_height / focal_length) * background_distance
        # plane_width = (self.scene.canvas_width / focal_length) * background_distance
        #
        # background_plane = pv.Plane(
        #     center=(0, 0, background_distance),
        #     direction=(0, 0, -1),  # 面朝相机
        #     i_size=plane_width,
        #     j_size=plane_height,
        #     i_resolution=1,
        #     j_resolution=1
        # )
        #
        # # 将平面添加到渲染器，并应用纹理
        # # 我们不需要光照，因为我们只关心它的深度
        # offscreen_plotter.add_mesh(background_plane, texture=rgb_image_texture, ambient=1.0, diffuse=0.0, specular=0.0)

        for item in models_with_id:
            obj_id = item['id']
            mesh_pv = item['model'].pv_obj
            # mesh_pv['obj_id'] = np.full(mesh_pv.n_points, obj_id, dtype=np.uint8)
            actor = offscreen_plotter.add_mesh(
                mesh_pv,
                # scalars='obj_id',
                cmap="gray",
                clim=[1, 255],
                lighting=False,
                name=item['name'],
                show_scalar_bar=False
            )
            actor.user_matrix = item['model'].actor.user_matrix

        offscreen_plotter.screenshot(None)
        combined_visib_mask_rgb = offscreen_plotter.screenshot(return_img=True)
        combined_visib_mask_gray = combined_visib_mask_rgb[:, :, 0]
        # # 获取深度图（Z-buffer），PyVista返回的是[0, 1]范围内的值
        # depth_raw = offscreen_plotter.get_image_depth()
        #
        # # 将深度值转换为毫米单位的16位整型图像
        # # 注意：这里的转换依赖于相机的近剪裁面和远剪裁面
        # near, far = offscreen_plotter.camera.clipping_range
        # far = max(far, background_distance + 1000)
        # offscreen_plotter.camera.clipping_range = (near, far)
        #
        # offscreen_plotter.screenshot(None)
        # depth_raw = offscreen_plotter.get_image_depth()
        # if far - near < 1e-6:
        #     depth_mm = np.zeros_like(depth_raw, dtype=np.uint16)
        # else:
        #     # 将深度值从[0, 1]区间 线性化并转换为毫米
        #     depth_mm = near + depth_raw * (far - near)  # 对于线性深度图，转换更简单
        #     depth_mm[depth_raw >= 0.999] = 0  # 将背景区域设为0

        # 初始化一个空的深度图
        height, width = self.scene.canvas_height, self.scene.canvas_width
        depth_mm = np.zeros((height, width), dtype=np.uint16)

        # 获取相机位置
        camera_pos = np.array(offscreen_plotter.camera.position)

        # 找到所有非背景像素的2D坐标
        points_y, points_x = np.where(combined_visib_mask_gray > 0)

        # 使用VTK的拾取器来获取每个像素的3D世界坐标
        picker = vtk.vtkWorldPointPicker()
        for y, x in zip(points_y, points_x):
            # VTK的Y坐标和Numpy数组的Y坐标是相反的
            picker.Pick(x, height - 1 - y, 0, offscreen_plotter.renderer)
            world_pos = np.array(picker.GetPickPosition())

            # 计算该点到相机中心的欧氏距离
            distance = np.linalg.norm(world_pos - camera_pos)

            # 将距离（假设单位为mm）存入深度图
            depth_mm[y, x] = np.round(distance).astype(np.uint16)

        depth_image = PIL.Image.fromarray(depth_mm.astype(np.uint16))

        depth_image.save(depth_dir / f"{image_id:06d}.png")
        # self.output_text.append("-> 深度图 (depth) 已生成。")

        offscreen_plotter.close()

        # 生成可见部分掩码图 (Mask Visib)----------------------------------------------------------------------------------
        offscreen_plotter = pv.Plotter(off_screen=True,
                                       window_size=[self.scene.canvas_width, self.scene.canvas_height], )
        offscreen_plotter.camera = self.plotter.camera.copy()

        offscreen_plotter.render_window.SetMultiSamples(0)
        offscreen_plotter.render_window.SetLineSmoothing(False)
        offscreen_plotter.render_window.SetPolygonSmoothing(False)

        offscreen_plotter.clear()
        offscreen_plotter.set_background('black')

        for item in models_with_id:
            obj_id = item['id']
            actor = offscreen_plotter.add_mesh(
                item['model'].pv_obj, color=f"#{obj_id:02x}{obj_id:02x}{obj_id:02x}",
                ambient=1.0, diffuse=0.0, specular=0.0, name=item['name'], show_scalar_bar=False)
            actor.user_matrix = item['model'].actor.user_matrix

        offscreen_plotter.screenshot(None)

        combined_visib_mask_rgb = offscreen_plotter.screenshot(return_img=True)
        combined_visib_mask_gray = combined_visib_mask_rgb[:, :, 0]

        # 然后，为每个物体从组合图中提取出自己的二值掩码并保存
        for item in models_with_id:
            obj_id = item['id']
            # ann_id = int(item['name'])
            ann_id = obj_id - 1
            # 创建一个全黑的画布
            binary_mask = np.zeros_like(combined_visib_mask_gray, dtype=np.uint8)
            # 只有在组合图中像素值等于当前物体ID的地方，才设为白色(255)
            binary_mask[combined_visib_mask_gray == obj_id] = 255

            mask_image = PIL.Image.fromarray(binary_mask)
            mask_image.save(mask_visib_dir / f"{image_id:06d}_{ann_id:06d}.png")

        # self.output_text.append(f"-> {len(models_with_id)} 张可见掩码图 (mask_visib) 已生成。")

        offscreen_plotter.close()

        #  生成完整掩码图 (Mask)------------------------------------------------------------------------------------------
        for item in models_with_id:
            offscreen_plotter = pv.Plotter(off_screen=True,
                                           window_size=[self.scene.canvas_width, self.scene.canvas_height])
            offscreen_plotter.camera = self.plotter.camera.copy()
            offscreen_plotter.render_window.SetMultiSamples(0)
            offscreen_plotter.render_window.SetLineSmoothing(False)
            offscreen_plotter.render_window.SetPolygonSmoothing(False)

            obj_id = item['id']
            # ann_id = int(item['name'])
            ann_id = obj_id - 1
            offscreen_plotter.clear()
            offscreen_plotter.set_background('black')

            # 只渲染当前这一个模型，颜色为纯白，无光照
            actor = offscreen_plotter.add_mesh(
                item['model'].pv_obj, color='white',
                ambient=1.0, diffuse=0.0, specular=0.0,
                name=item['name'], show_scalar_bar=False)
            actor.user_matrix = item['model'].actor.user_matrix

            # 渲染结果本身就是一张黑白二值图
            offscreen_plotter.screenshot(None)
            single_mask_rgb = offscreen_plotter.screenshot(return_img=True)
            single_mask_gray = single_mask_rgb[:, :, 0]  # R=G=B=255 or 0

            mask_image = PIL.Image.fromarray(single_mask_gray)
            mask_image.save(mask_dir / f"{image_id:06d}_{ann_id:06d}.png")
            offscreen_plotter.close()

        rgb_image = PIL.Image.open(list(self.scene.image_container.images.values())[0].path)
        rgb_image.save(rgb_dir / f"{image_id:06d}.png")
        # self.output_text.append(f"-> {len(models_with_id)} 张完整掩码图 (mask) 已生成。")

        # self.output_text.append("-> 所有图像生成完毕！")

    def generate_scene_camera_json_line(self):
        scene_camera_json_line = {}
        if self.current_workspace_index >= 0:
            scene_camera_json_line[str(self.current_workspace_index)] = {
                "cam_K": [self.scene.fx, 0.0, self.scene.cx,
                          0.0, self.scene.fy, self.scene.cy,
                          0.0, 0.0, 1.0],
                "depth_scale": 1.0
            }
        return scene_camera_json_line

    def generate_scene_gt_json_line(self):
        scene_gt_json_line = {}
        if self.current_workspace_index >= 0:
            meshes_gt = []
            for mesh in self.scene.mesh_container.meshes:
                mesh_matrix = utils.get_actor_user_matrix(mesh)
                meshes_gt.append({
                    "cam_R_m2c": mesh_matrix[:3, :3].flatten().tolist(),
                    "cam_t_m2c": mesh_matrix[:3, 3].flatten().tolist(),
                    "obj_id": int(''.join([c for c in str(mesh.base_name) if c.isdigit()]))
                })
            scene_gt_json_line[str(self.current_workspace_index)] = meshes_gt
        return scene_gt_json_line

    def generate_scene_gt_info_json_line(self, save_dirs):
        depth_dir = save_dirs['depth_dir']
        mask_dir = save_dirs['mask_dir']
        mask_visib_dir = save_dirs['mask_visib_dir']

        scene_gt_info_json_line = {}
        if self.current_workspace_index >= 0:
            meshes_gt_info = []
            depth_image = cv.imread(str(depth_dir / f"{self.current_workspace_index:06d}.png"), cv.IMREAD_UNCHANGED)
            for i, mesh in enumerate(self.scene.mesh_container.meshes):
                # 转换为布尔数组
                mask_full = cv.imread(str(mask_dir / f'{self.current_workspace_index:06d}_{i:06d}.png'),
                                      cv.IMREAD_GRAYSCALE) > 0
                mask_visib = cv.imread(str(mask_visib_dir / f'{self.current_workspace_index:06d}_{i:06d}.png'),
                                       cv.IMREAD_GRAYSCALE) > 0

                px_count_full = int(np.count_nonzero(mask_full))
                bbox_full = utils.get_bbox_from_mask(mask_full)
                px_count_visib = int(np.count_nonzero(mask_visib))
                bbox_visib = utils.get_bbox_from_mask(mask_visib)

                valid_depth_mask = (depth_image > 0)
                valid_pixels = np.logical_and(mask_full, valid_depth_mask)
                px_count_valid = int(np.count_nonzero(valid_pixels))

                height, width = self.scene.canvas_height, self.scene.canvas_width
                x, y, w, h = bbox_full
                if x <= 0 or y <= 0 or (x + w) >= width or (y + h) >= height:
                    zoom_factor = 2
                    temp_plotter = pv.Plotter(off_screen=True, window_size=[width * zoom_factor, height * zoom_factor])
                    temp_plotter.camera = self.plotter.camera.copy()

                    # temp_plotter.camera.zoom(1 / zoom_factor)  # 缩小，即扩大视野

                    original_angle_deg = temp_plotter.camera.view_angle
                    original_angle_rad = math.radians(original_angle_deg)
                    new_angle_rad = 2 * math.atan(2 * math.tan(original_angle_rad / 2))
                    new_angle_deg = math.degrees(new_angle_rad)
                    temp_plotter.camera.view_angle = new_angle_deg

                    temp_plotter.clear()
                    temp_plotter.set_background('black')
                    actor = temp_plotter.add_mesh(mesh.pv_obj, color='white', ambient=1.0,
                                                  diffuse=0.0, specular=0.0, name=mesh.name, show_scalar_bar=False)
                    actor.user_matrix = mesh.actor.user_matrix
                    temp_plotter.screenshot(None)
                    full_silhouette_rgb = temp_plotter.screenshot(return_img=True)
                    full_silhouette_gray = full_silhouette_rgb[:, :, 0]
                    full_silhouette_mask = full_silhouette_gray > 0

                    # mask_image = PIL.Image.fromarray(full_silhouette_gray)
                    # mask_image.save(mask_visib_dir / f"{i:06d}large.png")

                    px_count_all = int(np.count_nonzero(full_silhouette_mask))
                    bbox_obj = utils.get_bbox_from_mask(full_silhouette_mask)
                    bbox_obj[0] -= int(width / zoom_factor)
                    bbox_obj[1] -= int(height / zoom_factor)
                    temp_plotter.close()

                    if bbox_visib == [-1, -1, -1, -1]:
                        bbox_obj = [-1, -1, -1, -1]

                else:
                    px_count_all = px_count_full
                    bbox_obj = bbox_full

                visib_fract = px_count_visib / px_count_all if px_count_all > 0 else 0.0

                meshes_gt_info.append({
                    "bbox_obj": bbox_obj,
                    "bbox_visib": bbox_visib,
                    "px_count_all": px_count_all,
                    "px_count_valid": px_count_valid,
                    "px_count_visib": px_count_visib,
                    "visib_fract": visib_fract,
                })
            scene_gt_info_json_line[str(self.current_workspace_index)] = meshes_gt_info

        return scene_gt_info_json_line

    @utils.require_attributes([('scene.image_container.reference', "Please load an image first!"),
                               ('scene.mesh_container.meshes', "Please load a mesh first!")])
    def export_bop_dataset(self):
        if not self.workspaces:
            return

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(None, "Choose a folder to save BOP images.")
        if not output_dir:
            return
        output_path = pathlib.Path(output_dir)

        i = 0
        try:
            for d in sorted([d for d in output_path.iterdir() if d.is_dir() and d.name.isdigit()]):
                if int(d.name) == i:
                    i += 1
            output_path = output_path / f"{i:06d}"
        except FileNotFoundError:
            utils.display_warning(f"{output_path} does not exist!")
            return

        self.output_text.append(f"\n-> save BOP images to: {output_path}")

        save_dirs = {
            "depth_dir": output_path / "depth",
            "mask_dir": output_path / "mask",
            "mask_visib_dir": output_path / "mask_visib",
            "rgb_dir": output_path / "rgb"
        }

        for d in save_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        cur_idx = self.current_workspace_index
        scene_camera = {}
        scene_gt = {}
        scene_gt_info = {}
        dialog = ExportBopDialog()
        dialog.show()
        try:
            for i in range(len(self.workspaces)):
                dialog.update_progress(i + 1, len(self.workspaces))
                self.switch_workspace(i)
                time.sleep(0.5)
                self.generate_bop_images(save_dirs)
                scene_camera.update(self.generate_scene_camera_json_line())
                scene_gt.update(self.generate_scene_gt_json_line())
                scene_gt_info.update(self.generate_scene_gt_info_json_line(save_dirs))
            with open(output_path / 'scene_camera.json', 'w') as f:
                json.dump(scene_camera, f, cls=BopJsonEncoder)
            with open(output_path / 'scene_gt.json', 'w') as f:
                json.dump(scene_gt, f, cls=BopJsonEncoder)
            with open(output_path / 'scene_gt_info.json', 'w') as f:
                json.dump(scene_gt_info, f, cls=BopJsonEncoder)
            self.output_text.append("\n-----Finished to save all BOP images.-----")
        except Exception as e:
            self.output_text.append('Failed to export BOP images: {}'.format(e))
            print(e)
        finally:
            dialog.close = dialog.accept
            dialog.close()
        self.switch_workspace(cur_idx)
    """
    Part of exporting dataset in BOP format ends.
    """
