"""
Utility functions for object extraction and manipulation.
"""

import numpy as np
import cv2
from typing import Tuple


def extract_object(image: np.ndarray, mask: np.ndarray, padding: int = 10) -> np.ndarray:
    """
    Extract object from image using mask with padding.
    
    Args:
        image: Source image (H, W, 3)
        mask: Binary mask (H, W) where 1 = object
        padding: Padding around object bounding box
        
    Returns:
        Extracted object with transparent background
    """
    # Find bounding box of mask
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return np.zeros((1, 1, 4), dtype=np.uint8)  # Empty object
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    # Add padding
    h, w = image.shape[:2]
    y_min = max(0, y_min - padding)
    y_max = min(h, y_max + padding + 1)
    x_min = max(0, x_min - padding)
    x_max = min(w, x_max + padding + 1)
    
    # Extract region
    object_region = image[y_min:y_max, x_min:x_max]
    mask_region = mask[y_min:y_max, x_min:x_max]
    
    # Create RGBA image with transparency
    if object_region.shape[2] == 3:
        rgba_object = np.zeros((*object_region.shape[:2], 4), dtype=np.uint8)
        rgba_object[:, :, :3] = object_region
        rgba_object[:, :, 3] = mask_region * 255
    else:
        rgba_object = object_region.copy()
        rgba_object[:, :, 3] = mask_region * 255
    
    return rgba_object


def extract_background(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extract background by removing object using mask.
    
    Args:
        image: Source image (H, W, 3)
        mask: Binary mask (H, W) where 1 = object to remove
        
    Returns:
        Background image with object region inpainted
    """
    # Invert mask (background = 1, object = 0)
    bg_mask = (1 - mask).astype(np.uint8) * 255
    
    # Use inpainting to fill object region
    # Convert mask to the format expected by cv2.inpaint
    inpaint_mask = mask.astype(np.uint8) * 255
    
    # Apply inpainting
    background = cv2.inpaint(image, inpaint_mask, 3, cv2.INPAINT_TELEA)
    
    return background


def resize_with_aspect_ratio(image: np.ndarray, 
                           max_width: int, 
                           max_height: int) -> Tuple[np.ndarray, float]:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_width: Maximum width
        max_height: Maximum height
        
    Returns:
        (resized_image, scale_factor)
    """
    h, w = image.shape[:2]
    
    # Calculate scale factor
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h, 1.0)  # Don't upscale
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale
    else:
        return image.copy(), 1.0


def create_bounding_box_mask(image_shape: Tuple[int, int], 
                           bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Create binary mask from bounding box.
    
    Args:
        image_shape: (height, width)
        bbox: (x, y, width, height)
        
    Returns:
        Binary mask
    """
    h, w = image_shape
    x, y, bw, bh = bbox
    
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Ensure coordinates are within image bounds
    x1 = max(0, min(w, x))
    y1 = max(0, min(h, y))
    x2 = max(0, min(w, x + bw))
    y2 = max(0, min(h, y + bh))
    
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1
    
    return mask


def refine_mask_edges(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    """
    Refine mask edges using image gradients.
    
    Args:
        mask: Initial binary mask
        image: Source image for gradient information
        
    Returns:
        Refined mask
    """
    # Convert to grayscale for edge detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Find edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges slightly
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Use GrabCut for refinement if mask is not too small
    if np.sum(mask) > 100:  # Only if object is large enough
        try:
            # Convert mask to GrabCut format
            gc_mask = np.zeros(mask.shape, dtype=np.uint8)
            gc_mask[mask == 1] = cv2.GC_PR_FGD  # Probable foreground
            gc_mask[mask == 0] = cv2.GC_PR_BGD  # Probable background
            
            # Initialize background and foreground models
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Apply GrabCut
            cv2.grabCut(image, gc_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
            
            # Extract foreground
            refined_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
            
            return refined_mask
            
        except Exception:
            # Fall back to original mask if GrabCut fails
            pass
    
    return mask


def morphological_cleanup(mask: np.ndarray, 
                         remove_small_objects: int = 50,
                         fill_holes: bool = True) -> np.ndarray:
    """
    Clean up mask using morphological operations.
    
    Args:
        mask: Binary mask
        remove_small_objects: Remove objects smaller than this (pixels)
        fill_holes: Whether to fill holes in objects
        
    Returns:
        Cleaned mask
    """
    cleaned = mask.copy()
    
    # Remove small objects
    if remove_small_objects > 0:
        # Find connected components
        num_labels, labels = cv2.connectedComponents(cleaned)
        
        # Remove small components
        for label in range(1, num_labels):
            component_size = np.sum(labels == label)
            if component_size < remove_small_objects:
                cleaned[labels == label] = 0
    
    # Fill holes
    if fill_holes:
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Fill contours
        cv2.fillPoly(cleaned, contours, 1)
    
    return cleaned