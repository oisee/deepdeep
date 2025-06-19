"""
Advanced dithering algorithms for better color mixing and transitions.
"""

import numpy as np
from typing import List, Tuple, Optional
from enum import Enum


class DitheringMethod(Enum):
    """Available dithering methods."""
    FLOYD_STEINBERG = "floyd_steinberg"
    ORDERED = "ordered"
    RANDOM = "random"
    BLUE_NOISE = "blue_noise"
    SIERRA = "sierra"
    ATKINSON = "atkinson"


class ZXDithering:
    """Dithering algorithms optimized for ZX Spectrum palette."""
    
    def __init__(self):
        # ZX Spectrum 16-color palette
        self.zx_palette = np.array([
            # Standard colors (0-7)
            [0, 0, 0],         # 0: Black
            [0, 0, 215],       # 1: Blue
            [215, 0, 0],       # 2: Red
            [215, 0, 215],     # 3: Magenta
            [0, 215, 0],       # 4: Green
            [0, 215, 215],     # 5: Cyan
            [215, 215, 0],     # 6: Yellow
            [215, 215, 215],   # 7: White
            
            # Bright colors (8-15)
            [0, 0, 0],         # 8: Bright Black (same as Black)
            [0, 0, 255],       # 9: Bright Blue
            [255, 0, 0],       # 10: Bright Red
            [255, 0, 255],     # 11: Bright Magenta
            [0, 255, 0],       # 12: Bright Green
            [0, 255, 255],     # 13: Bright Cyan
            [255, 255, 0],     # 14: Bright Yellow
            [255, 255, 255]    # 15: Bright White
        ], dtype=np.uint8)
        
        # Precompute ordered dither matrices
        self.bayer_2x2 = np.array([
            [0, 2],
            [3, 1]
        ]) / 4.0
        
        self.bayer_4x4 = np.array([
            [0,  8,  2, 10],
            [12, 4, 14,  6],
            [3, 11,  1,  9],
            [15, 7, 13,  5]
        ]) / 16.0
        
        self.bayer_8x8 = np.array([
            [ 0, 32,  8, 40,  2, 34, 10, 42],
            [48, 16, 56, 24, 50, 18, 58, 26],
            [12, 44,  4, 36, 14, 46,  6, 38],
            [60, 28, 52, 20, 62, 30, 54, 22],
            [ 3, 35, 11, 43,  1, 33,  9, 41],
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47,  7, 39, 13, 45,  5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21]
        ]) / 64.0
    
    def apply_dithering(self, image: np.ndarray, method: DitheringMethod = DitheringMethod.FLOYD_STEINBERG) -> np.ndarray:
        """
        Apply dithering to image for better ZX Spectrum color approximation.
        
        Args:
            image: Input RGB image (H, W, 3)
            method: Dithering method to use
            
        Returns:
            Dithered image using ZX Spectrum palette
        """
        if method == DitheringMethod.FLOYD_STEINBERG:
            return self._floyd_steinberg_dither(image)
        elif method == DitheringMethod.ORDERED:
            return self._ordered_dither(image)
        elif method == DitheringMethod.RANDOM:
            return self._random_dither(image)
        elif method == DitheringMethod.BLUE_NOISE:
            return self._blue_noise_dither(image)
        elif method == DitheringMethod.SIERRA:
            return self._sierra_dither(image)
        elif method == DitheringMethod.ATKINSON:
            return self._atkinson_dither(image)
        else:
            raise ValueError(f"Unsupported dithering method: {method}")
    
    def _find_closest_colors(self, pixel: np.ndarray, num_colors: int = 2) -> Tuple[List[int], List[float]]:
        """Find closest palette colors and their distances."""
        distances = np.linalg.norm(self.zx_palette - pixel, axis=1)
        closest_indices = np.argsort(distances)[:num_colors]
        closest_distances = distances[closest_indices]
        return closest_indices.tolist(), closest_distances.tolist()
    
    def _floyd_steinberg_dither(self, image: np.ndarray) -> np.ndarray:
        """Floyd-Steinberg error diffusion dithering."""
        h, w, c = image.shape
        result = image.astype(np.float32).copy()
        
        for y in range(h):
            for x in range(w):
                old_pixel = result[y, x].copy()
                
                # Find closest color
                distances = np.linalg.norm(self.zx_palette - old_pixel, axis=1)
                closest_idx = np.argmin(distances)
                new_pixel = self.zx_palette[closest_idx].astype(np.float32)
                
                result[y, x] = new_pixel
                error = old_pixel - new_pixel
                
                # Distribute error to neighboring pixels
                if x + 1 < w:
                    result[y, x + 1] += error * 7/16
                if y + 1 < h:
                    if x - 1 >= 0:
                        result[y + 1, x - 1] += error * 3/16
                    result[y + 1, x] += error * 5/16
                    if x + 1 < w:
                        result[y + 1, x + 1] += error * 1/16
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _sierra_dither(self, image: np.ndarray) -> np.ndarray:
        """Sierra error diffusion dithering (more aggressive than Floyd-Steinberg)."""
        h, w, c = image.shape
        result = image.astype(np.float32).copy()
        
        for y in range(h):
            for x in range(w):
                old_pixel = result[y, x].copy()
                
                # Find closest color
                distances = np.linalg.norm(self.zx_palette - old_pixel, axis=1)
                closest_idx = np.argmin(distances)
                new_pixel = self.zx_palette[closest_idx].astype(np.float32)
                
                result[y, x] = new_pixel
                error = old_pixel - new_pixel
                
                # Sierra filter (distributes error over larger area)
                if x + 1 < w:
                    result[y, x + 1] += error * 5/32
                if x + 2 < w:
                    result[y, x + 2] += error * 3/32
                if y + 1 < h:
                    if x - 2 >= 0:
                        result[y + 1, x - 2] += error * 2/32
                    if x - 1 >= 0:
                        result[y + 1, x - 1] += error * 4/32
                    result[y + 1, x] += error * 5/32
                    if x + 1 < w:
                        result[y + 1, x + 1] += error * 4/32
                    if x + 2 < w:
                        result[y + 1, x + 2] += error * 2/32
                if y + 2 < h:
                    if x - 1 >= 0:
                        result[y + 2, x - 1] += error * 2/32
                    result[y + 2, x] += error * 3/32
                    if x + 1 < w:
                        result[y + 2, x + 1] += error * 2/32
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _atkinson_dither(self, image: np.ndarray) -> np.ndarray:
        """Atkinson dithering (used by Apple LaserWriter)."""
        h, w, c = image.shape
        result = image.astype(np.float32).copy()
        
        for y in range(h):
            for x in range(w):
                old_pixel = result[y, x].copy()
                
                # Find closest color
                distances = np.linalg.norm(self.zx_palette - old_pixel, axis=1)
                closest_idx = np.argmin(distances)
                new_pixel = self.zx_palette[closest_idx].astype(np.float32)
                
                result[y, x] = new_pixel
                error = old_pixel - new_pixel
                
                # Atkinson filter (only distributes 3/4 of error)
                if x + 1 < w:
                    result[y, x + 1] += error * 1/8
                if x + 2 < w:
                    result[y, x + 2] += error * 1/8
                if y + 1 < h:
                    if x - 1 >= 0:
                        result[y + 1, x - 1] += error * 1/8
                    result[y + 1, x] += error * 1/8
                    if x + 1 < w:
                        result[y + 1, x + 1] += error * 1/8
                if y + 2 < h:
                    result[y + 2, x] += error * 1/8
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _ordered_dither(self, image: np.ndarray, matrix_size: int = 4) -> np.ndarray:
        """Ordered dithering using Bayer matrix."""
        h, w, c = image.shape
        
        if matrix_size == 2:
            dither_matrix = self.bayer_2x2
        elif matrix_size == 4:
            dither_matrix = self.bayer_4x4
        elif matrix_size == 8:
            dither_matrix = self.bayer_8x8
        else:
            dither_matrix = self.bayer_4x4
        
        matrix_h, matrix_w = dither_matrix.shape
        result = np.zeros_like(image)
        
        for y in range(h):
            for x in range(w):
                pixel = image[y, x].astype(np.float32)
                threshold = dither_matrix[y % matrix_h, x % matrix_w] * 255
                
                # Find two closest colors
                closest_indices, distances = self._find_closest_colors(pixel, 2)
                
                if len(closest_indices) > 1:
                    # Calculate blend factor based on distance
                    total_distance = distances[0] + distances[1]
                    if total_distance > 0:
                        blend_factor = distances[0] / total_distance
                    else:
                        blend_factor = 0.5
                    
                    # Use threshold to choose between colors
                    if np.mean(pixel) + threshold * blend_factor > 127.5:
                        chosen_color = self.zx_palette[closest_indices[1]]
                    else:
                        chosen_color = self.zx_palette[closest_indices[0]]
                else:
                    chosen_color = self.zx_palette[closest_indices[0]]
                
                result[y, x] = chosen_color
        
        return result.astype(np.uint8)
    
    def _random_dither(self, image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
        """Random dithering with controlled noise."""
        h, w, c = image.shape
        result = np.zeros_like(image)
        
        # Generate random noise
        noise = (np.random.random((h, w)) - 0.5) * intensity * 255
        
        for y in range(h):
            for x in range(w):
                pixel = image[y, x].astype(np.float32) + noise[y, x]
                pixel = np.clip(pixel, 0, 255)
                
                # Find closest color
                distances = np.linalg.norm(self.zx_palette - pixel, axis=1)
                closest_idx = np.argmin(distances)
                result[y, x] = self.zx_palette[closest_idx]
        
        return result.astype(np.uint8)
    
    def _blue_noise_dither(self, image: np.ndarray) -> np.ndarray:
        """Blue noise dithering (more natural-looking random patterns)."""
        # For simplicity, using a pre-generated blue noise pattern
        # In a full implementation, this would use actual blue noise
        h, w, c = image.shape
        
        # Generate pseudo-blue noise (low-frequency suppressed random)
        noise = np.random.random((h, w))
        # Apply simple high-pass filter to suppress low frequencies
        from scipy import ndimage
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) / 8
        try:
            noise = ndimage.convolve(noise, kernel, mode='constant')
        except:
            # Fallback if scipy not available
            pass
        
        noise = (noise - 0.5) * 0.3 * 255
        
        result = np.zeros_like(image)
        for y in range(h):
            for x in range(w):
                pixel = image[y, x].astype(np.float32) + noise[y, x]
                pixel = np.clip(pixel, 0, 255)
                
                # Find closest color
                distances = np.linalg.norm(self.zx_palette - pixel, axis=1)
                closest_idx = np.argmin(distances)
                result[y, x] = self.zx_palette[closest_idx]
        
        return result.astype(np.uint8)
    
    def adaptive_dither(self, image: np.ndarray, edge_threshold: float = 30.0) -> np.ndarray:
        """
        Adaptive dithering that uses different methods based on image content.
        
        - Smooth areas: Ordered dithering
        - Detailed areas: Floyd-Steinberg
        - High contrast edges: Atkinson
        """
        h, w, c = image.shape
        
        # Calculate edge strength using Sobel
        gray = np.mean(image, axis=2)
        try:
            from scipy import ndimage
            sobel_x = ndimage.sobel(gray, axis=1)
            sobel_y = ndimage.sobel(gray, axis=0)
            edge_strength = np.sqrt(sobel_x**2 + sobel_y**2)
        except:
            # Fallback simple edge detection
            edge_strength = np.abs(np.gradient(gray, axis=0)[0]) + np.abs(np.gradient(gray, axis=1)[0])
        
        # Apply different dithering methods based on content
        result = np.zeros_like(image)
        
        # Smooth areas (low edge strength) - use ordered dithering
        smooth_mask = edge_strength < edge_threshold * 0.5
        smooth_areas = self._ordered_dither(image)
        result[smooth_mask] = smooth_areas[smooth_mask]
        
        # High edge areas - use Atkinson dithering
        edge_mask = edge_strength > edge_threshold
        edge_areas = self._atkinson_dither(image)
        result[edge_mask] = edge_areas[edge_mask]
        
        # Medium detail areas - use Floyd-Steinberg
        detail_mask = ~smooth_mask & ~edge_mask
        detail_areas = self._floyd_steinberg_dither(image)
        result[detail_mask] = detail_areas[detail_mask]
        
        return result.astype(np.uint8)


def create_color_transition_dither(start_color: Tuple[int, int, int], 
                                  end_color: Tuple[int, int, int], 
                                  width: int, height: int,
                                  method: DitheringMethod = DitheringMethod.FLOYD_STEINBERG) -> np.ndarray:
    """
    Create a dithered gradient between two colors.
    
    Args:
        start_color: RGB tuple for start color
        end_color: RGB tuple for end color  
        width: Width of gradient
        height: Height of gradient
        method: Dithering method to use
        
    Returns:
        Dithered gradient image
    """
    # Create smooth gradient
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    
    for x in range(width):
        ratio = x / (width - 1) if width > 1 else 0
        color = [
            int(start_color[i] * (1 - ratio) + end_color[i] * ratio)
            for i in range(3)
        ]
        gradient[:, x] = color
    
    # Apply dithering
    ditherer = ZXDithering()
    return ditherer.apply_dithering(gradient, method)