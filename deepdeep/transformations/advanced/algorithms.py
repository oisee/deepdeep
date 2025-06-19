"""
Advanced transformation algorithms for enhanced image processing.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import cv2
import math


class AdvancedTransforms:
    """Advanced transformation algorithms beyond basic affine/perspective."""
    
    def __init__(self):
        pass
    
    def apply_spherical_warp(self, image: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """
        Apply spherical warping effect (fish-eye or barrel effect).
        
        Args:
            image: Input image (H, W, C)
            strength: Warping strength (0.0 = none, 1.0 = strong)
            
        Returns:
            Spherically warped image
        """
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        max_radius = min(center_x, center_y)
        
        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        x = x - center_x
        y = y - center_y
        
        # Calculate radius and angle
        radius = np.sqrt(x**2 + y**2)
        angle = np.arctan2(y, x)
        
        # Apply spherical transformation
        normalized_radius = radius / max_radius
        warped_radius = normalized_radius * (1 + strength * normalized_radius**2)
        warped_radius = np.clip(warped_radius * max_radius, 0, max_radius * 1.5)
        
        # Convert back to coordinates
        new_x = (warped_radius * np.cos(angle) + center_x).astype(np.float32)
        new_y = (warped_radius * np.sin(angle) + center_y).astype(np.float32)
        
        # Remap image
        result = cv2.remap(image, new_x, new_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return result
    
    def apply_twist_warp(self, image: np.ndarray, angle: float = 0.2) -> np.ndarray:
        """
        Apply twist/spiral warping effect.
        
        Args:
            image: Input image (H, W, C)
            angle: Maximum twist angle in radians
            
        Returns:
            Twisted image
        """
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        max_radius = min(center_x, center_y)
        
        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        x = x - center_x
        y = y - center_y
        
        # Calculate radius and current angle
        radius = np.sqrt(x**2 + y**2)
        current_angle = np.arctan2(y, x)
        
        # Apply twist based on distance from center
        normalized_radius = np.clip(radius / max_radius, 0, 1)
        twist_amount = angle * (1 - normalized_radius**2)  # More twist at edges
        new_angle = current_angle + twist_amount
        
        # Convert back to coordinates
        new_x = (radius * np.cos(new_angle) + center_x).astype(np.float32)
        new_y = (radius * np.sin(new_angle) + center_y).astype(np.float32)
        
        # Remap image
        result = cv2.remap(image, new_x, new_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return result
    
    def apply_ripple_effect(self, image: np.ndarray, amplitude: float = 10.0, frequency: float = 0.1) -> np.ndarray:
        """
        Apply ripple/wave effect.
        
        Args:
            image: Input image (H, W, C)
            amplitude: Wave amplitude in pixels
            frequency: Wave frequency
            
        Returns:
            Rippled image
        """
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        x = x - center_x
        y = y - center_y
        
        # Calculate distance from center
        radius = np.sqrt(x**2 + y**2)
        
        # Apply ripple displacement
        ripple = np.sin(radius * frequency) * amplitude
        
        # Calculate displacement direction (radial)
        with np.errstate(divide='ignore', invalid='ignore'):
            dx = ripple * x / (radius + 1e-8)
            dy = ripple * y / (radius + 1e-8)
        
        # Create new coordinates
        new_x = (x + center_x - dx).astype(np.float32)
        new_y = (y + center_y - dy).astype(np.float32)
        
        # Remap image
        result = cv2.remap(image, new_x, new_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return result
    
    def apply_mesh_warp(self, image: np.ndarray, control_points: np.ndarray, displaced_points: np.ndarray) -> np.ndarray:
        """
        Apply mesh-based warping using control points.
        
        Args:
            image: Input image (H, W, C)
            control_points: Original control point positions (N, 2)
            displaced_points: New control point positions (N, 2)
            
        Returns:
            Mesh-warped image
        """
        h, w = image.shape[:2]
        
        # Use thin plate spline transformation if enough points
        if len(control_points) >= 3:
            try:
                # Create transformation using OpenCV
                tps = cv2.createThinPlateSplineShapeTransformer()
                
                # Reshape points for OpenCV format
                source_shape = control_points.reshape(1, -1, 2).astype(np.float32)
                target_shape = displaced_points.reshape(1, -1, 2).astype(np.float32)
                
                tps.estimateTransformation(target_shape, source_shape, [])
                
                # Create coordinate grid
                coords = np.mgrid[0:h, 0:w].transpose(1, 2, 0).astype(np.float32)
                coords = coords.reshape(-1, 2)
                
                # Apply transformation
                transformed_coords = tps.applyTransformation(coords.reshape(1, -1, 2))[1]
                transformed_coords = transformed_coords.reshape(h, w, 2)
                
                # Remap image
                map_x = transformed_coords[:, :, 1].astype(np.float32)
                map_y = transformed_coords[:, :, 0].astype(np.float32)
                
                result = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                return result
                
            except Exception:
                # Fallback to simple interpolation
                pass
        
        # Fallback: return original image if TPS fails
        return image.copy()
    
    def apply_artistic_oil_painting(self, image: np.ndarray, size: int = 7, dynratio: int = 1) -> np.ndarray:
        """
        Apply oil painting artistic effect.
        
        Args:
            image: Input image (H, W, C)
            size: Size of the neighborhood area
            dynratio: Image is divided into dynratio before processing
            
        Returns:
            Oil painting style image
        """
        try:
            # Use OpenCV's oil painting filter if available
            result = cv2.xphoto.oilPainting(image, size, dynratio)
            return result
        except AttributeError:
            # Fallback: simple smoothing
            kernel = np.ones((size, size), np.float32) / (size * size)
            smoothed = cv2.filter2D(image, -1, kernel)
            return smoothed
    
    def apply_edge_preserving_filter(self, image: np.ndarray, flags: int = 1, sigma_s: float = 50, sigma_r: float = 0.4) -> np.ndarray:
        """
        Apply edge-preserving smoothing filter.
        
        Args:
            image: Input image (H, W, C)
            flags: Edge preserving filter type (1 or 2)
            sigma_s: Size of the neighborhood
            sigma_r: How dissimilar colors are averaged
            
        Returns:
            Edge-preserving filtered image
        """
        try:
            result = cv2.edgePreservingFilter(image, flags=flags, sigma_s=sigma_s, sigma_r=sigma_r)
            return result
        except AttributeError:
            # Fallback: bilateral filter
            result = cv2.bilateralFilter(image, 9, 75, 75)
            return result
    
    def apply_detail_enhancement(self, image: np.ndarray, sigma_s: float = 10, sigma_r: float = 0.15) -> np.ndarray:
        """
        Enhance image details while preserving structure.
        
        Args:
            image: Input image (H, W, C)
            sigma_s: Size of the neighborhood  
            sigma_r: How dissimilar colors are averaged
            
        Returns:
            Detail-enhanced image
        """
        try:
            result = cv2.detailEnhance(image, sigma_s=sigma_s, sigma_r=sigma_r)
            return result
        except AttributeError:
            # Fallback: unsharp masking
            gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
            result = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
            return np.clip(result, 0, 255).astype(np.uint8)
    
    def apply_stylization(self, image: np.ndarray, sigma_s: float = 150, sigma_r: float = 0.25) -> np.ndarray:
        """
        Apply artistic stylization effect.
        
        Args:
            image: Input image (H, W, C)
            sigma_s: Size of the neighborhood
            sigma_r: How dissimilar colors are averaged
            
        Returns:
            Stylized image
        """
        try:
            result = cv2.stylization(image, sigma_s=sigma_s, sigma_r=sigma_r)
            return result
        except AttributeError:
            # Fallback: simple quantization
            # Reduce color depth for artistic effect
            factor = 64
            result = (image // factor) * factor
            return result.astype(np.uint8)
    
    def apply_adaptive_threshold_artistic(self, image: np.ndarray, style: str = 'sketch') -> np.ndarray:
        """
        Apply adaptive thresholding for artistic effects.
        
        Args:
            image: Input image (H, W, C)
            style: Style type ('sketch', 'ink', 'pencil')
            
        Returns:
            Artistic thresholded image
        """
        # Convert to grayscale first
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        if style == 'sketch':
            # Sketch-like effect
            inv_gray = 255 - gray
            blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
            sketch = cv2.divide(gray, 255 - blur, scale=256)
            result = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
            
        elif style == 'ink':
            # Ink drawing effect
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            result = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
        elif style == 'pencil':
            # Pencil drawing effect
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Laplacian(blur, cv2.CV_8U, ksize=5)
            edges = 255 - edges
            result = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
        else:
            result = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        return result
    
    def apply_color_quantization(self, image: np.ndarray, k: int = 8) -> np.ndarray:
        """
        Apply k-means color quantization.
        
        Args:
            image: Input image (H, W, C)
            k: Number of color clusters
            
        Returns:
            Quantized image
        """
        h, w, c = image.shape
        data = image.reshape((-1, c)).astype(np.float32)
        
        # Apply k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8 and reshape
        centers = np.uint8(centers)
        quantized_data = centers[labels.flatten()]
        quantized = quantized_data.reshape(h, w, c)
        
        return quantized


class ZXSpectrumEffects:
    """ZX Spectrum-specific visual effects and enhancements."""
    
    def __init__(self):
        self.zx_palette = np.array([
            [0, 0, 0], [0, 0, 215], [215, 0, 0], [215, 0, 215],
            [0, 215, 0], [0, 215, 215], [215, 215, 0], [215, 215, 215],
            [0, 0, 0], [0, 0, 255], [255, 0, 0], [255, 0, 255],
            [0, 255, 0], [0, 255, 255], [255, 255, 0], [255, 255, 255]
        ], dtype=np.uint8)
    
    def simulate_phosphor_glow(self, image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
        """Simulate CRT phosphor glow effect."""
        # Create glow by blurring bright areas
        bright_mask = np.mean(image, axis=2) > 128
        
        # Blur the image
        glowed = cv2.GaussianBlur(image, (15, 15), 0)
        
        # Blend original with glow
        result = image.astype(np.float32)
        glow_contribution = glowed.astype(np.float32) * intensity
        
        # Add glow primarily to bright areas
        for i in range(3):
            result[:, :, i] = np.where(bright_mask, 
                                     np.clip(result[:, :, i] + glow_contribution[:, :, i], 0, 255),
                                     result[:, :, i])
        
        return result.astype(np.uint8)
    
    def apply_scanlines(self, image: np.ndarray, intensity: float = 0.8) -> np.ndarray:
        """Apply CRT scanline effect."""
        h, w, c = image.shape
        result = image.copy().astype(np.float32)
        
        # Create scanline pattern (darken every other line)
        for y in range(1, h, 2):
            result[y] *= intensity
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def apply_color_bleeding(self, image: np.ndarray, amount: float = 0.2) -> np.ndarray:
        """Simulate color bleeding between adjacent pixels."""
        kernel = np.array([[0, amount/4, 0],
                          [amount/4, 1-amount, amount/4],
                          [0, amount/4, 0]])
        
        result = np.zeros_like(image, dtype=np.float32)
        
        for i in range(3):
            result[:, :, i] = cv2.filter2D(image[:, :, i].astype(np.float32), -1, kernel)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def apply_attribute_clash_simulation(self, image: np.ndarray, block_size: int = 8) -> np.ndarray:
        """Simulate ZX Spectrum attribute clash by limiting colors per block."""
        h, w, c = image.shape
        result = image.copy()
        
        # Process in 8x8 blocks
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                block_h = min(block_size, h - y)
                block_w = min(block_size, w - x)
                block = image[y:y+block_h, x:x+block_w]
                
                # Find the 2 most common colors in the block
                pixels = block.reshape(-1, 3)
                unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
                
                if len(unique_colors) > 2:
                    # Keep only the 2 most frequent colors
                    top_2_indices = np.argsort(counts)[-2:]
                    dominant_colors = unique_colors[top_2_indices]
                    
                    # Map all pixels to closest of the 2 dominant colors
                    for py in range(block_h):
                        for px in range(block_w):
                            pixel = block[py, px]
                            distances = [np.linalg.norm(pixel - color) for color in dominant_colors]
                            closest_idx = np.argmin(distances)
                            result[y+py, x+px] = dominant_colors[closest_idx]
        
        return result