"""
Core transformation framework with comprehensive geometric transformations.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import cv2
import math


@dataclass
class TransformParams:
    """Parameters for all supported transformations."""
    
    # Affine transformations
    rotation: float = 0.0          # degrees, -180 to 180
    scale_x: float = 1.0           # 0.5 to 2.0
    scale_y: float = 1.0           # 0.5 to 2.0
    translate_x: float = 0.0       # pixels, -image_width to image_width
    translate_y: float = 0.0       # pixels, -image_height to image_height
    shear_x: float = 0.0           # degrees, -45 to 45
    shear_y: float = 0.0           # degrees, -45 to 45
    
    # Perspective transformation
    perspective_x: float = 0.0     # -0.002 to 0.002
    perspective_y: float = 0.0     # -0.002 to 0.002
    
    # Non-linear distortions
    barrel_k1: float = 0.0         # -0.5 to 0.5 (barrel/pincushion)
    barrel_k2: float = 0.0         # -0.5 to 0.5 (barrel/pincushion)
    wave_amp_x: float = 0.0        # 0 to 10 (wave amplitude X)
    wave_freq_x: float = 0.0       # 0 to 0.1 (wave frequency X)
    wave_amp_y: float = 0.0        # 0 to 10 (wave amplitude Y)
    wave_freq_y: float = 0.0       # 0 to 0.1 (wave frequency Y)
    
    def is_identity(self) -> bool:
        """Check if parameters represent identity transformation."""
        return (
            abs(self.rotation) < 1e-6 and
            abs(self.scale_x - 1.0) < 1e-6 and
            abs(self.scale_y - 1.0) < 1e-6 and
            abs(self.translate_x) < 1e-6 and
            abs(self.translate_y) < 1e-6 and
            abs(self.shear_x) < 1e-6 and
            abs(self.shear_y) < 1e-6 and
            abs(self.perspective_x) < 1e-6 and
            abs(self.perspective_y) < 1e-6 and
            abs(self.barrel_k1) < 1e-6 and
            abs(self.barrel_k2) < 1e-6 and
            abs(self.wave_amp_x) < 1e-6 and
            abs(self.wave_amp_y) < 1e-6
        )
    
    def copy(self) -> 'TransformParams':
        """Create a deep copy of the parameters."""
        return TransformParams(
            rotation=self.rotation,
            scale_x=self.scale_x,
            scale_y=self.scale_y,
            translate_x=self.translate_x,
            translate_y=self.translate_y,
            shear_x=self.shear_x,
            shear_y=self.shear_y,
            perspective_x=self.perspective_x,
            perspective_y=self.perspective_y,
            barrel_k1=self.barrel_k1,
            barrel_k2=self.barrel_k2,
            wave_amp_x=self.wave_amp_x,
            wave_freq_x=self.wave_freq_x,
            wave_amp_y=self.wave_amp_y,
            wave_freq_y=self.wave_freq_y
        )


class TransformationEngine:
    """Core transformation application engine."""
    
    def __init__(self):
        self.interpolation_mode = cv2.INTER_LINEAR
        self.border_mode = cv2.BORDER_REFLECT_101
    
    def apply_transform(self, 
                       image: np.ndarray, 
                       params: TransformParams,
                       output_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Apply all transformations in correct order.
        
        Args:
            image: Input image (H, W, C) or (H, W)
            params: Transformation parameters
            output_size: Output size (width, height). If None, uses input size.
            
        Returns:
            Transformed image
        """
        if params.is_identity():
            return image.copy()
        
        if output_size is None:
            output_size = (image.shape[1], image.shape[0])
        
        result = image.copy()
        
        # Apply affine transformations first
        if self._has_affine(params):
            M = self._build_affine_matrix(params, image.shape)
            result = cv2.warpAffine(
                result, M, output_size,
                flags=self.interpolation_mode,
                borderMode=self.border_mode
            )
        
        # Apply perspective transformation
        if self._has_perspective(params):
            P = self._build_perspective_matrix(params, result.shape, output_size)
            result = cv2.warpPerspective(
                result, P, output_size,
                flags=self.interpolation_mode,
                borderMode=self.border_mode
            )
        
        # Apply non-linear distortions
        if self._has_barrel(params):
            result = self._apply_barrel_distortion(result, params)
        
        if self._has_wave(params):
            result = self._apply_wave_distortion(result, params)
        
        return result
    
    def _has_affine(self, params: TransformParams) -> bool:
        """Check if any affine transformations are needed."""
        return not (
            abs(params.rotation) < 1e-6 and
            abs(params.scale_x - 1.0) < 1e-6 and
            abs(params.scale_y - 1.0) < 1e-6 and
            abs(params.translate_x) < 1e-6 and
            abs(params.translate_y) < 1e-6 and
            abs(params.shear_x) < 1e-6 and
            abs(params.shear_y) < 1e-6
        )
    
    def _has_perspective(self, params: TransformParams) -> bool:
        """Check if perspective transformation is needed."""
        return abs(params.perspective_x) > 1e-6 or abs(params.perspective_y) > 1e-6
    
    def _has_barrel(self, params: TransformParams) -> bool:
        """Check if barrel distortion is needed."""
        return abs(params.barrel_k1) > 1e-6 or abs(params.barrel_k2) > 1e-6
    
    def _has_wave(self, params: TransformParams) -> bool:
        """Check if wave distortion is needed."""
        return (abs(params.wave_amp_x) > 1e-6 or 
                abs(params.wave_amp_y) > 1e-6)
    
    def _build_affine_matrix(self, params: TransformParams, shape: Tuple[int, ...]) -> np.ndarray:
        """Build 2x3 affine transformation matrix."""
        h, w = shape[:2]
        center_x, center_y = w / 2, h / 2
        
        # Start with identity
        M = np.eye(3, dtype=np.float32)
        
        # Translation to center
        M_center = np.array([
            [1, 0, -center_x],
            [0, 1, -center_y],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Rotation
        if abs(params.rotation) > 1e-6:
            cos_r = math.cos(math.radians(params.rotation))
            sin_r = math.sin(math.radians(params.rotation))
            M_rot = np.array([
                [cos_r, -sin_r, 0],
                [sin_r, cos_r, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            M = M_rot @ M
        
        # Scale
        if abs(params.scale_x - 1.0) > 1e-6 or abs(params.scale_y - 1.0) > 1e-6:
            M_scale = np.array([
                [params.scale_x, 0, 0],
                [0, params.scale_y, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            M = M_scale @ M
        
        # Shear
        if abs(params.shear_x) > 1e-6 or abs(params.shear_y) > 1e-6:
            shear_x_rad = math.radians(params.shear_x)
            shear_y_rad = math.radians(params.shear_y)
            M_shear = np.array([
                [1, math.tan(shear_x_rad), 0],
                [math.tan(shear_y_rad), 1, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            M = M_shear @ M
        
        # Translation back from center
        M_uncenter = np.array([
            [1, 0, center_x],
            [0, 1, center_y],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Additional translation
        M_translate = np.array([
            [1, 0, params.translate_x],
            [0, 1, params.translate_y],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Combine all transformations
        M = M_translate @ M_uncenter @ M @ M_center
        
        # Return 2x3 matrix for cv2.warpAffine
        return M[:2, :]
    
    def _build_perspective_matrix(self, 
                                 params: TransformParams, 
                                 shape: Tuple[int, ...],
                                 output_size: Tuple[int, int]) -> np.ndarray:
        """Build 3x3 perspective transformation matrix."""
        h, w = shape[:2]
        
        # Define source corners (input image)
        src_corners = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)
        
        # Apply perspective distortion to corners
        dst_corners = src_corners.copy()
        
        # Apply perspective transformation
        center_x, center_y = w / 2, h / 2
        
        for i, (x, y) in enumerate(dst_corners):
            # Distance from center
            dx = x - center_x
            dy = y - center_y
            
            # Apply perspective distortion
            dst_corners[i][0] = x + dx * params.perspective_x * abs(dy)
            dst_corners[i][1] = y + dy * params.perspective_y * abs(dx)
        
        # Get perspective transform matrix
        return cv2.getPerspectiveTransform(src_corners, dst_corners)
    
    def _apply_barrel_distortion(self, image: np.ndarray, params: TransformParams) -> np.ndarray:
        """Apply barrel/pincushion distortion."""
        h, w = image.shape[:2]
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Normalize coordinates to [-1, 1]
        x_norm = (x - w / 2) / (w / 2)
        y_norm = (y - h / 2) / (h / 2)
        
        # Calculate radial distance
        r = np.sqrt(x_norm**2 + y_norm**2)
        
        # Apply barrel distortion
        r_distorted = r * (1 + params.barrel_k1 * r**2 + params.barrel_k2 * r**4)
        
        # Avoid division by zero
        mask = r > 1e-6
        x_distorted = np.where(mask, x_norm * r_distorted / r, x_norm)
        y_distorted = np.where(mask, y_norm * r_distorted / r, y_norm)
        
        # Convert back to pixel coordinates
        x_distorted = (x_distorted * w / 2) + w / 2
        y_distorted = (y_distorted * h / 2) + h / 2
        
        # Remap image
        return cv2.remap(
            image, 
            x_distorted.astype(np.float32), 
            y_distorted.astype(np.float32),
            self.interpolation_mode,
            borderMode=self.border_mode
        )
    
    def _apply_wave_distortion(self, image: np.ndarray, params: TransformParams) -> np.ndarray:
        """Apply wave distortion."""
        h, w = image.shape[:2]
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply wave distortion
        x_distorted = x.astype(np.float32)
        y_distorted = y.astype(np.float32)
        
        if abs(params.wave_amp_x) > 1e-6:
            x_distorted += params.wave_amp_x * np.sin(2 * np.pi * params.wave_freq_x * y)
        
        if abs(params.wave_amp_y) > 1e-6:
            y_distorted += params.wave_amp_y * np.sin(2 * np.pi * params.wave_freq_y * x)
        
        # Remap image
        return cv2.remap(
            image,
            x_distorted,
            y_distorted,
            self.interpolation_mode,
            borderMode=self.border_mode
        )
    
    def estimate_bounds(self, 
                       image_shape: Tuple[int, int], 
                       params: TransformParams) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Estimate output bounds after transformation.
        
        Args:
            image_shape: (height, width) of input image
            params: Transformation parameters
            
        Returns:
            ((min_x, min_y), (max_x, max_y)) bounding box
        """
        h, w = image_shape
        
        # Define corner points
        corners = np.array([
            [0, 0, 1],
            [w, 0, 1],
            [w, h, 1],
            [0, h, 1]
        ], dtype=np.float32).T
        
        if self._has_affine(params):
            # Apply affine transformation
            M = np.vstack([self._build_affine_matrix(params, image_shape), [0, 0, 1]])
            corners = M @ corners
        
        # Extract x, y coordinates
        x_coords = corners[0] / corners[2]
        y_coords = corners[1] / corners[2]
        
        min_x, max_x = int(np.floor(x_coords.min())), int(np.ceil(x_coords.max()))
        min_y, max_y = int(np.floor(y_coords.min())), int(np.ceil(y_coords.max()))
        
        return (min_x, min_y), (max_x, max_y)