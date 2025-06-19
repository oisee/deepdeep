"""
Recomposition engine for reassembling transformed objects optimally.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import cv2

from ...segmentation.detectors import ObjectWorkspace
from ...transformations.search.explorer import TransformationResult
from ...transformations.search.constraints import ConstraintEvaluator


@dataclass
class CompositionResult:
    """Result of recomposition process."""
    canvas: np.ndarray
    object_positions: Dict[int, Tuple[int, int]]  # object_id -> (x, y)
    layer_order: List[int]
    quality_score: float
    constraint_score: float
    details: Dict[str, Any]


class RecompositionEngine:
    """Reassemble transformed objects optimally."""
    
    def __init__(self, mode: str = 'standard', canvas_size: Tuple[int, int] = (256, 192)):
        self.mode = mode
        self.canvas_size = canvas_size  # (width, height)
        self.evaluator = ConstraintEvaluator(mode)
        
    def manual_compose(self, 
                      workspaces: List[ObjectWorkspace],
                      selected_variants: Dict[int, int],
                      positions: Optional[Dict[int, Tuple[int, int]]] = None,
                      layer_order: Optional[List[int]] = None) -> CompositionResult:
        """
        User-guided composition with manual positioning.
        
        Args:
            workspaces: List of object workspaces
            selected_variants: object_id -> variant_index mapping
            positions: object_id -> (x, y) position mapping
            layer_order: Rendering order (background to foreground)
            
        Returns:
            CompositionResult
        """
        # Create blank canvas
        canvas = np.zeros((*self.canvas_size[::-1], 3), dtype=np.uint8)  # (h, w, 3)
        
        # Determine layer order
        if layer_order is None:
            layer_order = self._default_layer_order(workspaces)
        
        # Determine positions
        if positions is None:
            positions = self._default_positions(workspaces)
        
        used_positions = {}
        
        # Render objects in layer order
        for obj_id in layer_order:
            if obj_id >= len(workspaces):
                continue
                
            ws = workspaces[obj_id]
            variant_idx = selected_variants.get(obj_id, 0)
            
            # Get transformation result
            if (ws.optimization_results and 
                variant_idx < len(ws.optimization_results)):
                variant = ws.optimization_results[variant_idx]
                transformed_image = variant.transformed_image
            else:
                # Use original if no variants available
                if ws.extracted.shape[2] == 4:
                    transformed_image = ws.extracted[:, :, :3]
                else:
                    transformed_image = ws.extracted
            
            # Get position
            if obj_id in positions:
                x, y = positions[obj_id]
            else:
                x, y = ws.original_bbox[:2]
            
            # Apply object to canvas
            canvas = self._apply_object_to_canvas(
                canvas, transformed_image, ws.mask, (x, y)
            )
            
            used_positions[obj_id] = (x, y)
        
        # Evaluate result
        quality_score = self.evaluator.estimate_quality_score(canvas)
        constraint_violation = self.evaluator.evaluate(canvas)
        
        return CompositionResult(
            canvas=canvas,
            object_positions=used_positions,
            layer_order=layer_order,
            quality_score=quality_score,
            constraint_score=constraint_violation.score,
            details=constraint_violation.details
        )
    
    def auto_compose(self, 
                    workspaces: List[ObjectWorkspace],
                    selected_variants: Dict[int, int]) -> CompositionResult:
        """
        Automatic optimal composition.
        
        Args:
            workspaces: List of object workspaces
            selected_variants: object_id -> variant_index mapping
            
        Returns:
            CompositionResult
        """
        # Phase 1: Resolve overlaps
        print("Phase 1: Resolving overlaps...")
        positions = self._resolve_overlaps(workspaces, selected_variants)
        
        # Phase 2: Optimize for color constraints
        print("Phase 2: Optimizing for constraints...")
        if self.mode == 'standard':
            positions = self._minimize_attribute_conflicts(
                workspaces, selected_variants, positions
            )
        elif self.mode == 'gigascreen':
            positions = self._optimize_screen_distribution(
                workspaces, selected_variants, positions
            )
        
        # Phase 3: Final assembly
        print("Phase 3: Final assembly...")
        layer_order = self._optimize_layer_order(workspaces, selected_variants, positions)
        
        return self.manual_compose(
            workspaces, selected_variants, positions, layer_order
        )
    
    def _default_layer_order(self, workspaces: List[ObjectWorkspace]) -> List[int]:
        """Determine default rendering order (background to foreground)."""
        # Sort by class priority and size
        class_priority = {
            'background': 0,
            'sprite': 1,
            'text': 2,
            'face': 3
        }
        
        objects_with_priority = []
        for i, ws in enumerate(workspaces):
            priority = class_priority.get(ws.class_name, 1)
            size = ws.original_bbox[2] * ws.original_bbox[3]
            objects_with_priority.append((i, priority, -size))  # Negative size for descending
        
        # Sort by priority, then by size (larger first within same priority)
        objects_with_priority.sort(key=lambda x: (x[1], x[2]))
        
        return [obj[0] for obj in objects_with_priority]
    
    def _default_positions(self, workspaces: List[ObjectWorkspace]) -> Dict[int, Tuple[int, int]]:
        """Get default positions (original bounding box positions)."""
        positions = {}
        for i, ws in enumerate(workspaces):
            x, y = ws.original_bbox[:2]
            # Ensure position is within canvas
            x = max(0, min(self.canvas_size[0] - 32, x))
            y = max(0, min(self.canvas_size[1] - 32, y))
            positions[i] = (x, y)
        return positions
    
    def _resolve_overlaps(self, 
                         workspaces: List[ObjectWorkspace],
                         selected_variants: Dict[int, int]) -> Dict[int, Tuple[int, int]]:
        """Resolve overlapping objects by adjusting positions."""
        positions = self._default_positions(workspaces)
        
        # Simple overlap resolution: move overlapping objects
        for i, ws1 in enumerate(workspaces):
            for j, ws2 in enumerate(workspaces[i+1:], i+1):
                if self._check_overlap(positions[i], positions[j], ws1, ws2):
                    # Move the smaller object
                    if ws1.original_bbox[2] * ws1.original_bbox[3] < ws2.original_bbox[2] * ws2.original_bbox[3]:
                        positions[i] = self._find_non_overlapping_position(
                            positions[i], workspaces, positions, i
                        )
                    else:
                        positions[j] = self._find_non_overlapping_position(
                            positions[j], workspaces, positions, j
                        )
        
        return positions
    
    def _check_overlap(self, 
                      pos1: Tuple[int, int], 
                      pos2: Tuple[int, int],
                      ws1: ObjectWorkspace, 
                      ws2: ObjectWorkspace) -> bool:
        """Check if two objects overlap at given positions."""
        x1, y1 = pos1
        x2, y2 = pos2
        w1, h1 = ws1.original_bbox[2:4]
        w2, h2 = ws2.original_bbox[2:4]
        
        # Check rectangle overlap
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)
    
    def _find_non_overlapping_position(self, 
                                     current_pos: Tuple[int, int],
                                     workspaces: List[ObjectWorkspace],
                                     positions: Dict[int, Tuple[int, int]],
                                     exclude_id: int) -> Tuple[int, int]:
        """Find a position that doesn't overlap with other objects."""
        x, y = current_pos
        ws = workspaces[exclude_id]
        w, h = ws.original_bbox[2:4]
        
        # Try positions in a spiral pattern
        for radius in range(1, 50, 5):
            for angle in range(0, 360, 30):
                new_x = int(x + radius * np.cos(np.radians(angle)))
                new_y = int(y + radius * np.sin(np.radians(angle)))
                
                # Check canvas bounds
                if (new_x < 0 or new_y < 0 or 
                    new_x + w > self.canvas_size[0] or 
                    new_y + h > self.canvas_size[1]):
                    continue
                
                # Check overlap with other objects
                overlaps = False
                for other_id, other_pos in positions.items():
                    if other_id != exclude_id:
                        if self._check_overlap(
                            (new_x, new_y), other_pos, 
                            ws, workspaces[other_id]
                        ):
                            overlaps = True
                            break
                
                if not overlaps:
                    return (new_x, new_y)
        
        # If no good position found, return original
        return current_pos
    
    def _minimize_attribute_conflicts(self, 
                                    workspaces: List[ObjectWorkspace],
                                    selected_variants: Dict[int, int],
                                    positions: Dict[int, Tuple[int, int]]) -> Dict[int, Tuple[int, int]]:
        """Adjust positions to minimize color clashes at boundaries."""
        # For standard mode, try to minimize colors in each 8x8 block
        
        # Create temporary canvas to analyze conflicts
        canvas = np.zeros((*self.canvas_size[::-1], 3), dtype=np.uint8)
        
        # Apply all objects to see conflicts
        for obj_id, (x, y) in positions.items():
            if obj_id >= len(workspaces):
                continue
                
            ws = workspaces[obj_id]
            variant_idx = selected_variants.get(obj_id, 0)
            
            if (ws.optimization_results and 
                variant_idx < len(ws.optimization_results)):
                transformed_image = ws.optimization_results[variant_idx].transformed_image
            else:
                if ws.extracted.shape[2] == 4:
                    transformed_image = ws.extracted[:, :, :3]
                else:
                    transformed_image = ws.extracted
            
            canvas = self._apply_object_to_canvas(
                canvas, transformed_image, ws.mask, (x, y)
            )
        
        # Analyze constraint violations
        violation = self.evaluator.evaluate(canvas)
        
        # If violations are high, try small position adjustments
        if violation.score > 0.5:
            optimized_positions = positions.copy()
            
            # Try small adjustments for each object
            for obj_id in positions:
                best_pos = positions[obj_id]
                best_score = violation.score
                
                # Try positions in a small radius
                for dx in [-4, 0, 4]:
                    for dy in [-4, 0, 4]:
                        if dx == 0 and dy == 0:
                            continue
                            
                        new_x = positions[obj_id][0] + dx
                        new_y = positions[obj_id][1] + dy
                        
                        # Check bounds
                        if (new_x < 0 or new_y < 0 or 
                            new_x + workspaces[obj_id].original_bbox[2] > self.canvas_size[0] or
                            new_y + workspaces[obj_id].original_bbox[3] > self.canvas_size[1]):
                            continue
                        
                        # Test this position
                        test_positions = optimized_positions.copy()
                        test_positions[obj_id] = (new_x, new_y)
                        
                        # Create test canvas
                        test_canvas = np.zeros((*self.canvas_size[::-1], 3), dtype=np.uint8)
                        
                        for test_obj_id, (test_x, test_y) in test_positions.items():
                            if test_obj_id >= len(workspaces):
                                continue
                                
                            test_ws = workspaces[test_obj_id]
                            test_variant_idx = selected_variants.get(test_obj_id, 0)
                            
                            if (test_ws.optimization_results and 
                                test_variant_idx < len(test_ws.optimization_results)):
                                test_image = test_ws.optimization_results[test_variant_idx].transformed_image
                            else:
                                if test_ws.extracted.shape[2] == 4:
                                    test_image = test_ws.extracted[:, :, :3]
                                else:
                                    test_image = test_ws.extracted
                            
                            test_canvas = self._apply_object_to_canvas(
                                test_canvas, test_image, test_ws.mask, (test_x, test_y)
                            )
                        
                        # Evaluate
                        test_violation = self.evaluator.evaluate(test_canvas)
                        
                        if test_violation.score < best_score:
                            best_score = test_violation.score
                            best_pos = (new_x, new_y)
                
                optimized_positions[obj_id] = best_pos
            
            return optimized_positions
        
        return positions
    
    def _optimize_screen_distribution(self, 
                                    workspaces: List[ObjectWorkspace],
                                    selected_variants: Dict[int, int],
                                    positions: Dict[int, Tuple[int, int]]) -> Dict[int, Tuple[int, int]]:
        """Optimize positions for GigaScreen flicker distribution."""
        # For GigaScreen, we want to distribute objects to minimize flicker artifacts
        # This is a simplified version - in practice would be more sophisticated
        return positions
    
    def _optimize_layer_order(self, 
                             workspaces: List[ObjectWorkspace],
                             selected_variants: Dict[int, int],
                             positions: Dict[int, Tuple[int, int]]) -> List[int]:
        """Optimize layer order for best visual result."""
        # Start with default order
        default_order = self._default_layer_order(workspaces)
        
        # For now, return default - could implement more sophisticated optimization
        return default_order
    
    def _apply_object_to_canvas(self, 
                              canvas: np.ndarray,
                              object_image: np.ndarray,
                              mask: np.ndarray,
                              position: Tuple[int, int]) -> np.ndarray:
        """Apply transformed object to canvas at given position."""
        x, y = position
        canvas_h, canvas_w = canvas.shape[:2]
        obj_h, obj_w = object_image.shape[:2]
        
        # Calculate intersection with canvas
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(canvas_w, x + obj_w)
        y2 = min(canvas_h, y + obj_h)
        
        if x2 <= x1 or y2 <= y1:
            return canvas  # No intersection
        
        # Calculate object region
        obj_x1 = x1 - x
        obj_y1 = y1 - y
        obj_x2 = obj_x1 + (x2 - x1)
        obj_y2 = obj_y1 + (y2 - y1)
        
        # Apply object with alpha blending if available
        result = canvas.copy()
        
        if object_image.shape[2] == 4:  # RGBA
            # Alpha blending
            alpha = object_image[obj_y1:obj_y2, obj_x1:obj_x2, 3:4] / 255.0
            
            for c in range(3):
                result[y1:y2, x1:x2, c] = (
                    alpha[:, :, 0] * object_image[obj_y1:obj_y2, obj_x1:obj_x2, c] +
                    (1 - alpha[:, :, 0]) * result[y1:y2, x1:x2, c]
                ).astype(np.uint8)
        else:
            # Direct copy
            result[y1:y2, x1:x2] = object_image[obj_y1:obj_y2, obj_x1:obj_x2]
        
        return result
    
    def preview_composition(self, 
                          workspaces: List[ObjectWorkspace],
                          selected_variants: Dict[int, int],
                          positions: Dict[int, Tuple[int, int]]) -> np.ndarray:
        """Create a quick preview of the composition."""
        result = self.manual_compose(workspaces, selected_variants, positions)
        return result.canvas