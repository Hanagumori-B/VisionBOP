import math
import numpy as np
import pyvista as pv
from functools import partial
from ..tools import utils
from . import ImageContainer
from . import MaskContainer
from . import MeshContainer


class Scene:
    def __init__(self, plotter, output_text):
        self.plotter = plotter
        self.camera = pv.Camera()
        self.output_text = output_text

        # set up the camera props
        self.image_container = ImageContainer(plotter=self.plotter)
        self.mask_container = MaskContainer(plotter=self.plotter)
        self.mesh_container = MeshContainer(plotter=self.plotter)

        # set camera related attributes for the linemod dataset
        self.fx = 572.4114
        self.fy = 573.57043
        self.cx = 325.2611
        self.cy = 242.04899
        self.canvas_height = 480
        self.canvas_width = 640
        self.cam_viewup = (0, -1, 0)

    def zoom_in(self):
        self.plotter.camera.zoom(1.1)

    def zoom_out(self):
        self.plotter.camera.zoom(1 / 1.1)

    def set_camera_intrinsics(self, fx, fy, cx, cy, height):
        # Set camera intrinsic attribute
        self.camera_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        view_angle = (180 / math.pi) * (2.0 * math.atan2(height / 2.0,
                                                         fy))  # or view_angle = np.degrees(2.0 * math.atan2(height/2.0, f)) # focal_length = (height / 2.0) / math.tan(math.radians(self.plotter.camera.view_angle / 2))
        self.camera.SetViewAngle(view_angle)  # view angle should be in degrees

    def set_camera_extrinsics(self, cam_viewup):
        self.camera.SetPosition((0, 0, -1e-8))  # Set the camera position at the origin of the world coordinate system
        self.camera.SetFocalPoint((0, 0, 0))  # Get the camera window center
        self.camera.SetViewUp(cam_viewup)

    def tap_toggle_opacity(self):
        if self.mesh_container.reference is not None and self.image_container.reference is not None:
            mesh_model = self.mesh_container.meshes[self.mesh_container.reference]
            image_model = self.image_container.images[self.image_container.reference]
            if mesh_model.opacity == 1.0:
                mesh_model.opacity = 0.0
                image_model.opacity = 1.0
            elif mesh_model.opacity == 0.9:
                mesh_model.opacity = 1.0
                image_model.opacity = 0.0
            else:
                mesh_model.opacity = 0.9
                image_model.opacity = 0.9
            image_model.actor.GetProperty().opacity = image_model.opacity
            image_model.opacity_spinbox.setValue(image_model.opacity)
            mesh_model.opacity_spinbox.setValue(mesh_model.opacity)
        elif self.mesh_container.reference is not None:
            mesh_model = self.mesh_container.meshes[self.mesh_container.reference]
            if mesh_model.opacity == 1.0:
                mesh_model.opacity = 0.0
            elif mesh_model.opacity == 0.9:
                mesh_model.opacity = 1.0
            else:
                mesh_model.opacity = 0.9
            mesh_model.opacity_spinbox.setValue(mesh_model.opacity)

    def ctrl_tap_opacity(self):
        if self.mesh_container.reference is not None:
            for mesh_model in self.mesh_container.meshes:
                if mesh_model.opacity != 0:
                    mesh_model.opacity_spinbox.setValue(0)
                else:
                    mesh_model.opacity_spinbox.setValue(mesh_model.previous_opacity)
        else:
            if self.image_container.reference is not None:
                image_model = self.image_container.images[self.image_container.reference]
                if image_model.opacity != 0:
                    image_model.opacity_spinbox.setValue(0)
                else:
                    image_model.opacity_spinbox.setValue(image_model.previous_opacity)
            if self.mask_container.reference is not None:
                mask_model = self.mask_container.masks[self.mask_container.reference]
                if mask_model.opacity != 0:
                    mask_model.opacity_spinbox.setValue(0)
                else:
                    mask_model.opacity_spinbox.setValue(mask_model.previous_opacity)

    def handle_mesh_click(self, index, output_text):
        self.mesh_container.reference = index
        mesh_model = self.mesh_container.meshes[self.mesh_container.reference]
        matrix = utils.get_actor_user_matrix(mesh_model)
        text = "[[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}],\n[{:.4f}, {:.4f}, {:.4f}, {:.4f}]]\n".format(
            matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3],
            matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3],
            matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3],
            matrix[3, 0], matrix[3, 1], matrix[3, 2], matrix[3, 3])
        if output_text:
            self.output_text.append(f"--> Mesh {index} pose is:")
            self.output_text.append(text)
        mesh_model.undo_poses.append(mesh_model.actor.user_matrix)
        mesh_model.undo_poses = mesh_model.undo_poses[-20:]

    def handle_image_click(self, name):
        # Add a new image as current reference
        self.image_container.reference = name
        image_model = self.image_container.images[self.image_container.reference]
        self.image_container.add_image_actor(image_model, self.fy, self.cx, self.cy)

        # Disconnect existing connections to prevent accumulation
        try:
            image_model.opacity_spinbox.valueChanged.disconnect()
        except TypeError:
            pass  # No existing connection

        # Set up the value in the opacity box
        image_model.opacity_spinbox.setValue(image_model.opacity)
        image_model.opacity_spinbox.valueChanged.connect(partial(self.image_container.set_image_opacity, name))

        # Update opacity for all images
        for image_name, img_model in self.image_container.images.items():
            if image_name != name and img_model is not None:
                img_model.opacity = 0.0
                img_model.opacity_spinbox.setValue(0.0)
                # Remove the other actors exist to free memory (very important)
                if hasattr(img_model, 'actor') and img_model.actor in self.plotter.renderer.actors.values():
                    self.plotter.remove_actor(img_model.actor)
                    del img_model.actor  # Remove reference to the actor
            else:
                img_model.opacity = 0.9
                img_model.opacity_spinbox.setValue(0.9)

    def mesh_color_value_change(self, index: int, color):
        if self.mesh_container.get_mesh_by_index(index) is not None:
            try:
                color = self.mesh_container.set_color(index, color)
                mesh_model = self.mesh_container.get_mesh_by_index(index)
                mesh_model.color = color
                if color != "nocs" and color != "select":
                    mesh_model.color_button.setStyleSheet(f"background-color: {mesh_model.color}")
                elif color == 'nocs':
                    gradient_str = """
                                    background-color: qlineargradient(
                                        spread:pad, x1:0, y1:0, x2:1, y2:1,
                                        stop:0 red, stop:0.17 orange, stop:0.33 yellow,
                                        stop:0.5 green, stop:0.67 blue, stop:0.83 indigo, stop:1 violet);
                                    """
                    mesh_model.color_button.setStyleSheet(gradient_str)
            except ValueError:
                utils.display_warning(f"Cannot set color ({color}) to {index}")

    def mask_color_value_change(self, name, color):
        if name in self.mask_container.masks:
            try:
                self.mask_container.set_mask_color(name, color)
                mask_model = self.mask_container.masks[name]
                mask_model.color = color
            except ValueError:
                utils.display_warning(f"Cannot set color ({color}) to mask")
                mask_model.color_button.setStyleSheet(f"background-color: {mask_model.color}")
