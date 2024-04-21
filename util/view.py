from projection.utils import get_extrinsic_matrix, get_intrinsic_matrix
from matplotlib import pyplot as plt
from projection.utils import get_cameras, get_extrinsic_matrix, get_intrinsic_matrix
from projection.projections import camera_to_image, world_to_camera, filter_pixel_points, get_3D_indices
import os
import torch
import numpy as np
import random


class Viewset:
    def __init__(self, camera_json, dataset_path) -> None:
        self.cameras = get_cameras(camera_json)
        self._dataset_path = dataset_path
        # assume intrinsics and dimensions are consistent across all views
        self._intrinsic = get_intrinsic_matrix(self.cameras[0])
        self.WIDTH = self.cameras[0]['width']
        self.HEIGHT = self.cameras[0]['height']
        self._count = len(self.cameras)

    def get(self, i):
        # Get Extrinsic & intrinsic Matrices for selected camera
        extrinsic = get_extrinsic_matrix(self.cameras[i])

        image = plt.imread(
            os.path.join(self._dataset_path, f"{self.cameras[i]['img_name']}.jpg"))

        scale = 1
        if self.WIDTH != image.shape[1]:
            scale = self.WIDTH / image.shape[1]

        return View(extrinsic, self._intrinsic, image, scale, self.WIDTH, self.HEIGHT, i)

    def sample(self, rate):
        indices = list(range(self._count))
        if rate < 1:
            k = round(self._count * rate)
            indices = random.sample(indices, k)

        return [self.get(i) for i in indices]


class View:
    def __init__(self, extrinsic, intrinsic, image, scale, width, height, index) -> None:
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self.image = image
        self.scale = scale
        self.width = width
        self.height = height
        self.index = index

    def scale_from_image(self, coords):
        return torch.round(coords * self.scale)

    def scale_to_image(self, coords):
        return torch.round(coords / self.scale)

    def project(self, gaussians, filter_outside_boundary=True):
        cam_coords = world_to_camera(gaussians, self.extrinsic)
        img_coords, depths, indices = camera_to_image(
            cam_coords, self.intrinsic)

        # filter out points outside of view boundary
        if filter_outside_boundary:
            img_coords, depths, indices = filter_pixel_points(
                img_coords, depths, indices, self.width, self.height)

        return Projection2D(img_coords, depths, indices)


class Projection2D:
    def __init__(self, coords, depths, indices) -> None:
        self.coords = coords
        self.depths = depths
        self.indices = indices

    def to_sam_input(self, scale=1):
        points = np.round(self.coords.numpy().T / scale)
        labels = np.array([1] * points.shape[0])
        return points, labels
