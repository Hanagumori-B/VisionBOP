"""
@author: Yike (Nicole) Zhang
@license: (C) Copyright.
@contact: yike.zhang@vanderbilt.edu
@software: Vision6D
@file: mesh_container.py
@time: 2023-07-03 20:27
@desc: create container for mesh related actions in application
"""
import pathlib
import trimesh
import matplotlib
import numpy as np
import pyvista as pv
from typing import Dict

from PyQt5 import QtWidgets

from ..tools import utils
from ..components import MeshModel
from .singleton import Singleton


class MeshContainer(metaclass=Singleton):
    def __init__(self, plotter):
        self.plotter = plotter
        self.reference = None
        # self.meshes: Dict[str, MeshModel] = {}
        self.meshes: list[MeshModel] = []
        self._actor_to_index_map: Dict[pv.Actor, int] = {}
        self.colors = ["wheat", "cyan", "magenta", "yellow", "lime", "dodgerblue", "white", "black"]

    def add_mesh_actor(self, mesh_source, transformation_matrix=np.eye(4)):
        source_mesh = None

        if isinstance(mesh_source, pathlib.Path) or isinstance(mesh_source, str):
            mesh_path = str(mesh_source)
            if pathlib.Path(mesh_path).suffix == '.mesh':
                mesh_source = utils.load_trimesh(mesh_source)
            else:
                mesh_source = pv.read(mesh_path)

        if isinstance(mesh_source, trimesh.Trimesh):
            source_mesh = mesh_source
            source_mesh.vertices = source_mesh.vertices.reshape(-1, 3)
            source_mesh.faces = source_mesh.faces.reshape(-1, 3)
            pv_mesh = pv.wrap(mesh_source)

        if isinstance(mesh_source, pv.PolyData):
            source_mesh = trimesh.Trimesh(mesh_source.points, mesh_source.faces.reshape((-1, 4))[:, 1:], process=False)
            source_mesh.vertices = source_mesh.vertices.reshape(-1, 3)
            source_mesh.faces = source_mesh.faces.reshape(-1, 3)
            pv_mesh = pv.wrap(source_mesh)

        if source_mesh is not None:
            base_name = pathlib.Path(mesh_path).stem
            count = sum(1 for m in self.meshes if m.name.startswith(base_name))
            name = f"{base_name} ({count})" if count > 0 else base_name
            # while name in self.meshes.keys():
            #     name += "_copy"

            # Create a new MeshModel instance
            mesh_model = MeshModel()
            mesh_model.name = name
            mesh_model.base_name = base_name
            mesh_model.path = mesh_path
            mesh_model.source_obj = source_mesh
            mesh_model.pv_obj = pv_mesh

            current_index = len(self.meshes)
            self.meshes.append(mesh_model)

            # Set spacing for the mesh
            mesh_model.pv_obj.points *= mesh_model.spacing
            mesh_model.color = self.colors[len(self.meshes) % len(self.colors)]

            # Add the new mesh_model to the meshes dictionary
            # self.meshes[mesh_model.name] = mesh_model

            # Add the mesh to the plotter
            mesh = self.plotter.add_mesh(
                mesh_model.pv_obj,
                color=mesh_model.color,
                opacity=mesh_model.opacity,
                pickable=True,
                name=mesh_model.name,
                show_scalar_bar=False
            )
            mesh.user_matrix = (
                transformation_matrix if self.reference is None
                else utils.get_actor_user_matrix(self.meshes[self.reference]).copy()
            )
            mesh_model.actor = mesh
            self._actor_to_index_map[mesh] = current_index

            # Save the initial pose
            mesh_model.undo_poses.append(utils.get_actor_user_matrix(mesh_model).copy())

            return mesh_model, current_index
        return None, -1

    def get_mesh_by_index(self, index: int):
        if isinstance(index, int) and 0 <= index < len(self.meshes):
            return self.meshes[index]
        return None

    def get_index_by_actor(self, actor):
        return self._actor_to_index_map.get(actor, -1)

    def remove_mesh(self, index):
        if not (0 <= index < len(self.meshes)):
            return False

        mesh_model = self.meshes.pop(index)
        self.plotter.remove_actor(mesh_model.actor)
        del self._actor_to_index_map[mesh_model.actor]

        # 重新构建映射
        new_map = {}
        for i, m in enumerate(self.meshes):
            new_map[m.actor] = i
            # 也许还需要更新actor的内部名称
        self._actor_to_index_map = new_map
        return True

    def set_color(self, index, color):
        scalars = None
        mesh_model = self.get_mesh_by_index(index)
        name = mesh_model.name
        if color == "nocs":
            scalars = utils.color_mesh_nocs(mesh_model.pv_obj.points)
        elif color == "texture":
            texture_path, _ = QtWidgets.QFileDialog().getOpenFileName(None, "Open file", "", "Files (*.npy)")
            if texture_path:
                mesh_model.texture_path = texture_path
                scalars = np.load(texture_path) / 255  # make sure the color range is from 0 to 1

        if scalars is not None:
            mesh = self.plotter.add_mesh(mesh_model.pv_obj, scalars=scalars, rgb=True, opacity=mesh_model.opacity,
                                         pickable=True, name=name, show_scalar_bar=False)
        else:
            mesh = self.plotter.add_mesh(mesh_model.pv_obj, opacity=mesh_model.opacity, pickable=True,
                                         name=name, show_scalar_bar=False)
            mesh.GetMapper().SetScalarVisibility(0)
            # if color in self.colors:
            #     mesh.GetProperty().SetColor(matplotlib.colors.to_rgb(color))
            # else:
            #     mesh_model.color = "wheat"
            #     mesh.GetProperty().SetColor(matplotlib.colors.to_rgb(mesh_model.color))
            mesh.GetProperty().SetColor(matplotlib.colors.to_rgb(color))

        mesh.user_matrix = mesh_model.actor.user_matrix
        mesh_model.actor = mesh  # ^ very import to change the actor too!
        if scalars is None and color == 'texture': color = mesh_model.color
        return color

    def set_mesh_opacity(self, index, mesh_opacity: float):
        mesh_model = self.get_mesh_by_index(index)
        mesh_model.previous_opacity = mesh_model.opacity
        mesh_model.opacity = mesh_opacity
        mesh_model.actor.user_matrix = pv.array_from_vtkmatrix(mesh_model.actor.GetMatrix())
        mesh_model.actor.GetProperty().opacity = mesh_opacity

    def get_poses_from_undo(self):
        # mesh_model = self.meshes[self.reference]
        # transformation_matrix = mesh_model.undo_poses.pop()
        # matrix = mesh_model.actor.user_matrix
        # while mesh_model.undo_poses and np.isclose(transformation_matrix, matrix).all():
        #     transformation_matrix = mesh_model.undo_poses.pop()
        #     mesh_model.actor.user_matrix = transformation_matrix
        mesh_model = self.get_mesh_by_index(self.reference)
        if len(mesh_model.undo_poses) > 1:
            # Pop the current pose and add it to the redo stack
            current_pose = mesh_model.undo_poses.pop()
            mesh_model.redo_poses.append(current_pose)

            # Get the previous pose from the undo stack
            previous_pose = mesh_model.undo_poses[-1]

            # Apply the previous pose
            mesh_model.actor.user_matrix = previous_pose
            return True
        return False

    def get_pose_from_redo(self):
        mesh_model = self.get_mesh_by_index(self.reference)
        if mesh_model.redo_poses:
            # Pop the pose from the redo stack
            pose_to_redo = mesh_model.redo_poses.pop()

            # Add it back to the undo stack and apply it
            mesh_model.undo_poses.append(pose_to_redo)
            mesh_model.actor.user_matrix = pose_to_redo
            return True
        return False

    def render_mesh(self, index, camera, width, height):
        render = utils.create_render(width, height)
        render.clear()
        mesh_model = self.get_mesh_by_index(index)
        name = mesh_model.name
        vertices, faces = mesh_model.source_obj.vertices, mesh_model.source_obj.faces
        pv_mesh = pv.wrap(trimesh.Trimesh(vertices, faces, process=False))
        colors = utils.get_mesh_actor_scalars(mesh_model.actor)
        if colors is not None:
            mesh = render.add_mesh(pv_mesh, scalars=colors, rgb=True, style='surface', opacity=1, name=name)
        else:
            mesh = render.add_mesh(pv_mesh, color=mesh_model.color, style='surface', opacity=1, name=name)
        matrix = utils.get_actor_user_matrix(mesh_model)
        mesh.user_matrix = matrix
        # set the light source to add the textures
        light = pv.Light(light_type='headlight')
        render.add_light(light)
        render.camera = camera
        render.disable()
        render.show(auto_close=False)
        image = render.last_image
        return image
