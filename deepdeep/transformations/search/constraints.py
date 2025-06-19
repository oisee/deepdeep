"""
Constraint evaluation for ZX Spectrum modes.
"""

from typing import Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class ConstraintViolation:
    """Represents a constraint violation with score and details."""
    score: float  # 0.0 = perfect, higher = worse
    details: Dict[str, Any]


class ConstraintEvaluator:
    """Evaluates how well images fit ZX Spectrum constraints."""
    
    def __init__(self, mode: str = 'standard'):
        self.mode = mode
        self.zx_palette = self._get_zx_palette()
        
    def _get_zx_palette(self) -> np.ndarray:
        """Get ZX Spectrum color palette (RGB values)."""
        return np.array([
            [0, 0, 0],        # Black
            [0, 0, 215],      # Blue  
            [215, 0, 0],      # Red
            [215, 0, 215],    # Magenta
            [0, 215, 0],      # Green
            [0, 215, 215],    # Cyan
            [215, 215, 0],    # Yellow
            [215, 215, 215],  # White
            # Bright colors
            [0, 0, 0],        # Bright Black (same as black)
            [0, 0, 255],      # Bright Blue
            [255, 0, 0],      # Bright Red
            [255, 0, 255],    # Bright Magenta
            [0, 255, 0],      # Bright Green
            [0, 255, 255],    # Bright Cyan
            [255, 255, 0],    # Bright Yellow
            [255, 255, 255],  # Bright White
        ], dtype=np.uint8)
    
    def evaluate(self, image: np.ndarray) -> ConstraintViolation:
        """
        Evaluate constraint violations for an image.
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            ConstraintViolation with score and details
        """
        if self.mode == 'standard':
            return self._evaluate_standard(image)
        elif self.mode == 'gigascreen':
            return self._evaluate_gigascreen(image)
        elif self.mode == 'mc8x4':
            return self._evaluate_mc8x4(image)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _evaluate_standard(self, image: np.ndarray) -> ConstraintViolation:
        """Evaluate standard ZX Spectrum constraints (2 colors per 8x8 block)."""
        h, w = image.shape[:2]
        
        # Pad image to be divisible by 8
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        
        h, w = image.shape[:2]
        violations = []
        total_blocks = 0
        
        # Check each 8x8 block
        for y in range(0, h, 8):
            for x in range(0, w, 8):
                block = image[y:y+8, x:x+8]
                block_violation = self._evaluate_block_colors(block, max_colors=2)
                violations.append(block_violation)
                total_blocks += 1
        
        # Calculate overall score
        avg_violation = np.mean(violations)
        
        details = {
            'avg_block_violation': avg_violation,
            'total_blocks': total_blocks,
            'worst_violation': np.max(violations),
            'violation_histogram': np.histogram(violations, bins=10)[0].tolist()
        }
        
        return ConstraintViolation(score=avg_violation, details=details)
    
    def _evaluate_gigascreen(self, image: np.ndarray) -> ConstraintViolation:
        """Evaluate GigaScreen constraints (4 colors via flicker)."""
        # For GigaScreen, we need to consider that we can achieve 4 colors
        # by flickering between two 2-color images at 50Hz
        standard_result = self._evaluate_standard(image)
        
        # GigaScreen is more forgiving - reduce penalty
        gigascreen_score = standard_result.score * 0.6
        
        details = standard_result.details.copy()
        details['gigascreen_bonus'] = 0.4
        
        return ConstraintViolation(score=gigascreen_score, details=details)
    
    def _evaluate_mc8x4(self, image: np.ndarray) -> ConstraintViolation:
        """Evaluate Multicolor 8x4 constraints."""
        h, w = image.shape[:2]
        
        # Pad image to be divisible by 4 vertically, 8 horizontally
        pad_h = (4 - h % 4) % 4
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        
        h, w = image.shape[:2]
        violations = []
        total_blocks = 0
        
        # Check each 8x4 block
        for y in range(0, h, 4):
            for x in range(0, w, 8):
                block = image[y:y+4, x:x+8]
                # MC8x4 allows more colors per block
                block_violation = self._evaluate_block_colors(block, max_colors=4)
                violations.append(block_violation)
                total_blocks += 1
        
        avg_violation = np.mean(violations)
        
        details = {
            'avg_block_violation': avg_violation,
            'total_blocks': total_blocks,
            'worst_violation': np.max(violations),
            'block_size': '8x4'
        }
        
        return ConstraintViolation(score=avg_violation, details=details)
    
    def _evaluate_block_colors(self, block: np.ndarray, max_colors: int) -> float:
        """
        Evaluate color constraint violation for a single block.
        
        Args:
            block: RGB block (H, W, 3)
            max_colors: Maximum allowed colors
            
        Returns:
            Violation score (0.0 = perfect, higher = worse)
        """
        # Quantize to ZX Spectrum palette
        quantized = self._quantize_to_palette(block)
        
        # Count unique colors
        unique_colors = np.unique(quantized.reshape(-1, 3), axis=0)
        num_colors = len(unique_colors)
        
        if num_colors <= max_colors:
            return 0.0  # No violation
        
        # Penalty increases exponentially with extra colors
        excess_colors = num_colors - max_colors
        violation = excess_colors / max_colors
        
        return min(violation, 2.0)  # Cap at 2.0
    
    def _quantize_to_palette(self, image: np.ndarray) -> np.ndarray:
        """Quantize image to ZX Spectrum palette."""
        h, w = image.shape[:2]
        image_flat = image.reshape(-1, 3).astype(np.float32)
        
        # Find closest palette color for each pixel
        palette_flat = self.zx_palette.astype(np.float32)
        
        # Calculate distances to all palette colors
        distances = np.linalg.norm(
            image_flat[:, np.newaxis, :] - palette_flat[np.newaxis, :, :],
            axis=2
        )
        
        # Find closest color indices
        closest_indices = np.argmin(distances, axis=1)
        
        # Map to closest colors
        quantized_flat = palette_flat[closest_indices]
        
        return quantized_flat.reshape(h, w, 3).astype(np.uint8)
    
    def estimate_quality_score(self, image: np.ndarray) -> float:
        """
        Estimate overall quality score for an image.
        
        Returns value between 0.0 (worst) and 1.0 (best).
        """
        violation = self.evaluate(image)
        
        # Convert violation score to quality score
        quality = 1.0 / (1.0 + violation.score)
        
        return quality