"""BFM09 face synthesis and differentiable Eye&Nose rasterization for AT3D-P."""

from typing import Optional, Tuple

import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
import torch.nn.functional as F

from afrlib.utils.log import get_logger


logger = get_logger(__name__)


class BFMFaceModel(nn.Module):
    """The released 80/64/80-dimensional BFM parameterization."""

    def __init__(self, model_path: str, camera_distance: float = 10.0) -> None:
        super().__init__()
        model = loadmat(model_path)

        mean_shape = model["meanshape"].astype(np.float32).reshape(-1, 3)
        mean_shape -= mean_shape.mean(axis=0, keepdims=True)
        self.register_buffer("mean_shape", torch.from_numpy(mean_shape.reshape(-1)))
        self.register_buffer("identity_basis", torch.from_numpy(model["idBase"].astype(np.float32)))
        self.register_buffer("expression_basis", torch.from_numpy(model["exBase"].astype(np.float32)))
        self.register_buffer("mean_texture", torch.from_numpy(model["meantex"].astype(np.float32).reshape(-1)))
        self.register_buffer("texture_basis", torch.from_numpy(model["texBase"].astype(np.float32)))
        self.register_buffer("point_faces", torch.from_numpy(model["point_buf"].astype(np.int64) - 1))
        self.register_buffer("full_triangles", torch.from_numpy(model["tri"].astype(np.int64) - 1))
        self.register_buffer(
            "eye_nose_triangles", torch.from_numpy(model["tri_eye_nose"].astype(np.int64) - 1)
        )
        self.register_buffer(
            "initial_lighting",
            torch.tensor([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).view(1, 1, 9),
        )
        self.camera_distance = float(camera_distance)

    @property
    def vertex_count(self) -> int:
        return self.mean_shape.numel() // 3

    def forward(
        self,
        identity: torch.Tensor,
        expression: torch.Tensor,
        texture: torch.Tensor,
        angle: torch.Tensor,
        translation: torch.Tensor,
        lighting: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        face_shape = self.compute_shape(identity, expression)
        rotation = self.compute_rotation(angle)
        transformed = face_shape @ rotation + translation.unsqueeze(1)
        camera_vertices = torch.cat(
            [transformed[..., :2], self.camera_distance - transformed[..., 2:3]], dim=-1
        )

        face_texture = self.compute_texture(texture)
        normals = self.compute_normals(face_shape) @ rotation
        colors = self.compute_colors(face_texture, normals, lighting)
        return camera_vertices, colors

    def compute_shape(self, identity: torch.Tensor, expression: torch.Tensor) -> torch.Tensor:
        batch = identity.shape[0]
        identity_part = torch.einsum("ij,bj->bi", self.identity_basis, identity)
        expression_part = torch.einsum("ij,bj->bi", self.expression_basis, expression)
        return (identity_part + expression_part + self.mean_shape).reshape(batch, -1, 3)

    def compute_texture(self, texture: torch.Tensor) -> torch.Tensor:
        batch = texture.shape[0]
        values = torch.einsum("ij,bj->bi", self.texture_basis, texture) + self.mean_texture
        return values.reshape(batch, -1, 3) / 255.0

    def compute_normals(self, shape: torch.Tensor) -> torch.Tensor:
        first = shape[:, self.full_triangles[:, 0]]
        second = shape[:, self.full_triangles[:, 1]]
        third = shape[:, self.full_triangles[:, 2]]
        face_normals = F.normalize(torch.cross(first - second, second - third, dim=-1), dim=-1)
        # point_buf uses index 0 as padding in MATLAB; after subtracting one it
        # indexes this final all-zero normal through Python's -1 semantics.
        padding = torch.zeros(shape.shape[0], 1, 3, device=shape.device, dtype=shape.dtype)
        face_normals = torch.cat([face_normals, padding], dim=1)
        return F.normalize(face_normals[:, self.point_faces].sum(dim=2), dim=-1)

    def compute_colors(
        self, texture: torch.Tensor, normals: torch.Tensor, lighting: torch.Tensor
    ) -> torch.Tensor:
        pi = np.pi
        a0, a1, a2 = pi, 2 * pi / np.sqrt(3.0), 2 * pi / np.sqrt(8.0)
        c0 = 1 / np.sqrt(4 * pi)
        c1 = np.sqrt(3.0) / np.sqrt(4 * pi)
        c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * pi)
        x, y, z = normals[..., :1], normals[..., 1:2], normals[..., 2:3]
        harmonics = torch.cat(
            [
                a0 * c0 * torch.ones_like(x),
                -a1 * c1 * y,
                a1 * c1 * z,
                -a1 * c1 * x,
                a2 * c2 * x * y,
                -a2 * c2 * y * z,
                0.5 * a2 * c2 / np.sqrt(3.0) * (3 * z.square() - 1),
                -a2 * c2 * x * z,
                0.5 * a2 * c2 * (x.square() - y.square()),
            ],
            dim=-1,
        )

        gamma = lighting.reshape(-1, 3, 9) + self.initial_lighting
        gamma = gamma.permute(0, 2, 1)
        shaded = torch.cat(
            [
                harmonics @ gamma[..., :1],
                harmonics @ gamma[..., 1:2],
                harmonics @ gamma[..., 2:3],
            ],
            dim=-1,
        )
        return texture * shaded

    @staticmethod
    def compute_rotation(angles: torch.Tensor) -> torch.Tensor:
        batch = angles.shape[0]
        one = torch.ones(batch, 1, device=angles.device, dtype=angles.dtype)
        zero = torch.zeros_like(one)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:3]
        rotation_x = torch.cat(
            [one, zero, zero, zero, x.cos(), -x.sin(), zero, x.sin(), x.cos()], dim=1
        ).reshape(batch, 3, 3)
        rotation_y = torch.cat(
            [y.cos(), zero, y.sin(), zero, one, zero, -y.sin(), zero, y.cos()], dim=1
        ).reshape(batch, 3, 3)
        rotation_z = torch.cat(
            [z.cos(), -z.sin(), zero, z.sin(), z.cos(), zero, zero, zero, one], dim=1
        ).reshape(batch, 3, 3)
        return (rotation_z @ rotation_y @ rotation_x).transpose(1, 2)


class MeshRenderer(nn.Module):
    """Minimal nvdiffrast CUDA renderer copied from Deep3DFaceRecon."""

    def __init__(
        self,
        rasterize_fov: float,
        z_near: float = 5.0,
        z_far: float = 15.0,
        rasterize_size: int = 224,
    ) -> None:
        super().__init__()
        # Delay the CUDA-only dependency until AT3D is actually instantiated.
        import nvdiffrast.torch as dr

        self._dr = dr
        half_width = np.tan(np.deg2rad(rasterize_fov * 0.5)) * z_near
        projection = np.asarray(
            [
                [z_near / half_width, 0, 0, 0],
                [0, z_near / -half_width, 0, 0],
                [0, 0, -(z_far + z_near) / (z_far - z_near), -(2 * z_far * z_near) / (z_far - z_near)],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )
        projection = projection @ np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
        self.register_buffer("projection", torch.from_numpy(projection))
        self.rasterize_size = int(rasterize_size)
        self._context: Optional[object] = None

    def forward(
        self, vertices: torch.Tensor, triangles: torch.Tensor, colors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if vertices.device.type != "cuda":
            raise RuntimeError("AT3D's nvdiffrast renderer requires a CUDA device")
        ones = torch.ones_like(vertices[..., :1])
        homogeneous = torch.cat([vertices, ones], dim=-1)
        homogeneous = torch.cat(
            [homogeneous[..., :1], -homogeneous[..., 1:2], homogeneous[..., 2:]], dim=-1
        )
        clip_vertices = homogeneous @ self.projection.T

        if self._context is None:
            self._context = self._dr.RasterizeCudaContext(device=vertices.device)
            logger.info("Created AT3D rasterizer on %s", vertices.device)

        triangles = triangles.to(dtype=torch.int32).contiguous()
        raster, _ = self._dr.rasterize(
            self._context,
            clip_vertices.contiguous(),
            triangles,
            resolution=(self.rasterize_size, self.rasterize_size),
        )
        mask = (raster[..., 3] > 0).to(vertices.dtype).unsqueeze(1)
        image, _ = self._dr.interpolate(colors.contiguous(), raster, triangles)
        return mask, image.permute(0, 3, 1, 2) * mask


class AT3DRenderer(nn.Module):
    """Synthesize a face and render its Eye&Nose triangle subset."""

    def __init__(
        self,
        model_path: str,
        focal: float = 1015.0,
        center: float = 112.0,
        camera_distance: float = 10.0,
        z_near: float = 5.0,
        z_far: float = 15.0,
    ) -> None:
        super().__init__()
        self.face_model = BFMFaceModel(model_path, camera_distance=camera_distance)
        field_of_view = 2 * np.arctan(center / focal) * 180 / np.pi
        self.rasterizer = MeshRenderer(
            rasterize_fov=float(field_of_view),
            z_near=z_near,
            z_far=z_far,
            rasterize_size=int(2 * center),
        )

    def forward(
        self,
        identity: torch.Tensor,
        expression: torch.Tensor,
        texture: torch.Tensor,
        angle: torch.Tensor,
        translation: torch.Tensor,
        lighting: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vertices, colors = self.face_model(
            identity, expression, texture, angle, translation, lighting
        )
        mask, patch = self.rasterizer(
            vertices, self.face_model.eye_nose_triangles, colors
        )
        return mask, patch, vertices, colors
