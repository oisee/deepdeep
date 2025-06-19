"""
Object detection and optimization for independent transformation.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import cv2

from ..transformations.search.explorer import TransformationExplorer, TransformationResult
from .utils import extract_object, extract_background, create_bounding_box_mask, refine_mask_edges


@dataclass
class ObjectWorkspace:
    """Workspace for a single detected object."""
    id: int
    class_name: str
    confidence: float
    original_bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    mask: np.ndarray
    extracted: np.ndarray
    background: np.ndarray
    optimization_results: Optional[List[TransformationResult]] = None


class SimpleObjectDetector:
    """
    Simple object detector using traditional CV methods.
    This serves as a fallback when ML models aren't available.
    """
    
    def __init__(self):
        self.min_object_size = 100  # Minimum pixels for an object
        
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects using simple methods.
        
        Args:
            image: Input image (H, W, 3)
            
        Returns:
            List of detected objects with bbox and confidence
        """
        objects = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Find edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            # Filter by size
            area = cv2.contourArea(contour)
            if area < self.min_object_size:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Simple heuristics for object classification
            aspect_ratio = w / h
            
            if aspect_ratio > 3.0:
                class_name = 'text'  # Wide objects likely text
            elif 0.7 <= aspect_ratio <= 1.3 and area > 1000:
                class_name = 'face'  # Square-ish large objects might be faces
            elif area > 2000:
                class_name = 'sprite'  # Large objects
            else:
                class_name = 'background'
            
            objects.append({
                'class': class_name,
                'confidence': min(0.8, area / 5000),  # Simple confidence based on size
                'bbox': (x, y, w, h)
            })
        
        return objects


class ObjectBasedOptimizer:
    """Optimize each object independently with appropriate constraints."""
    
    def __init__(self, mode: str = 'standard'):
        self.mode = mode
        self.detector = SimpleObjectDetector()
        self.explorer = TransformationExplorer(mode)
        
    def segment_and_optimize(self, image: np.ndarray) -> List[ObjectWorkspace]:
        """
        Segment image into objects and optimize each independently.
        
        Args:
            image: Input image (H, W, 3)
            
        Returns:
            List of ObjectWorkspace with optimization results
        """
        print("Detecting objects...")
        objects = self.detector.detect(image)
        print(f"Found {len(objects)} objects")
        
        workspaces = []
        
        # Create workspace for each object
        for i, obj in enumerate(objects):
            print(f"Processing object {i}: {obj['class']}")
            
            # Create mask from bounding box
            mask = create_bounding_box_mask(image.shape[:2], obj['bbox'])
            
            # Refine mask using image gradients
            mask = refine_mask_edges(mask, image)
            
            # Extract object and background
            extracted = extract_object(image, mask)
            background = extract_background(image, mask)
            
            workspace = ObjectWorkspace(
                id=i,
                class_name=obj['class'],
                confidence=obj['confidence'],
                original_bbox=obj['bbox'],
                mask=mask,
                extracted=extracted,
                background=background
            )
            
            workspaces.append(workspace)
        
        # Optimize each object independently
        for ws in workspaces:
            print(f"Optimizing {ws.class_name} (ID: {ws.id})")
            ws.optimization_results = self._optimize_object(ws)
        
        return workspaces
    
    def _optimize_object(self, workspace: ObjectWorkspace) -> List[TransformationResult]:
        """Optimize transformations for a single object."""
        # Get class-specific constraints
        constraints = self.explorer.get_object_constraints(workspace.class_name)
        
        # Create search config based on constraints
        search_config = self._create_search_config(constraints)
        
        # Convert extracted object to RGB for processing
        if workspace.extracted.shape[2] == 4:
            # Convert RGBA to RGB with white background
            rgb_extracted = np.zeros((*workspace.extracted.shape[:2], 3), dtype=np.uint8)
            alpha = workspace.extracted[:, :, 3] / 255.0
            
            for c in range(3):
                rgb_extracted[:, :, c] = (
                    workspace.extracted[:, :, c] * alpha + 
                    255 * (1 - alpha)
                ).astype(np.uint8)
        else:
            rgb_extracted = workspace.extracted[:, :, :3]
        
        # Skip very small objects
        if rgb_extracted.shape[0] < 10 or rgb_extracted.shape[1] < 10:
            return []
        
        # Explore transformations
        try:
            results = self.explorer.explore_transformations(rgb_extracted, search_config)
            return results[:10]  # Return top 10 results
        except Exception as e:
            print(f"Optimization failed for object {workspace.id}: {e}")
            return []
    
    def _create_search_config(self, constraints: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Create search configuration from constraints."""
        # Extract ranges for coarse search
        rot_range = constraints.get('rotation', (-15, 15))
        scale_range = constraints.get('scale_x', (0.8, 1.2))
        trans_range = constraints.get('translate_x', (-20, 20))
        
        return {
            'max_results': 20,
            'enable_fine_search': True,
            'enable_nonlinear': constraints.get('barrel_k1', (0, 0))[1] > 0,
            'coarse': {
                'rotation_step': max(1, (rot_range[1] - rot_range[0]) / 6),
                'scale_step': max(0.05, (scale_range[1] - scale_range[0]) / 6),
                'translate_step': max(2, (trans_range[1] - trans_range[0]) / 6),
                'rotation_range': rot_range,
                'scale_range': scale_range,
                'translate_range': trans_range,
                'max_combinations': 200  # Limit for speed
            },
            'fine': {
                'rotation_step': 0.5,
                'scale_step': 0.01,
                'translate_step': 1,
                'search_radius': 2,
            }
        }
    
    def load_yolo_detector(self):
        """Load YOLO detector if available."""
        try:
            from ultralytics import YOLO
            
            class YOLODetector:
                def __init__(self):
                    self.model = YOLO('yolov8n.pt')  # Nano model for speed
                    
                def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
                    results = self.model(image, verbose=False)
                    objects = []
                    
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for i in range(len(boxes)):
                                # Get box coordinates
                                box = boxes.xyxy[i].cpu().numpy()
                                x1, y1, x2, y2 = box
                                
                                # Get class name and confidence
                                class_id = int(boxes.cls[i])
                                confidence = float(boxes.conf[i])
                                class_name = self.model.names[class_id]
                                
                                # Map YOLO classes to our classes
                                mapped_class = self._map_yolo_class(class_name)
                                
                                objects.append({
                                    'class': mapped_class,
                                    'confidence': confidence,
                                    'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1))
                                })
                    
                    return objects
                
                def _map_yolo_class(self, yolo_class: str) -> str:
                    """Map YOLO class names to our object types."""
                    face_classes = ['person']
                    text_classes = ['book', 'laptop', 'tv', 'cell phone']
                    sprite_classes = ['car', 'bus', 'truck', 'airplane', 'boat', 'bicycle', 'motorcycle']
                    
                    if yolo_class in face_classes:
                        return 'face'
                    elif yolo_class in text_classes:
                        return 'text'
                    elif yolo_class in sprite_classes:
                        return 'sprite'
                    else:
                        return 'background'
            
            self.detector = YOLODetector()
            print("YOLO detector loaded successfully")
            
        except ImportError:
            print("YOLO not available, using simple detector")
    
    def load_sam_segmenter(self):
        """Load Segment Anything Model if available."""
        try:
            # This would load SAM if available
            # For now, we'll use the simple edge-based refinement
            print("SAM not implemented yet, using edge refinement")
        except ImportError:
            print("SAM not available")
    
    def get_optimization_summary(self, workspaces: List[ObjectWorkspace]) -> Dict[str, Any]:
        """Get summary of optimization results."""
        summary = {
            'total_objects': len(workspaces),
            'objects_by_class': {},
            'optimization_stats': {
                'successful': 0,
                'failed': 0,
                'avg_score_improvement': 0
            }
        }
        
        # Count objects by class
        for ws in workspaces:
            if ws.class_name not in summary['objects_by_class']:
                summary['objects_by_class'][ws.class_name] = 0
            summary['objects_by_class'][ws.class_name] += 1
            
            # Count optimization success
            if ws.optimization_results and len(ws.optimization_results) > 0:
                summary['optimization_stats']['successful'] += 1
            else:
                summary['optimization_stats']['failed'] += 1
        
        return summary