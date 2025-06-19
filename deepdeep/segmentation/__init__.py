"""
Object detection and segmentation for independent optimization.
"""

from .detectors import ObjectBasedOptimizer, ObjectWorkspace
from .utils import extract_object, extract_background

__all__ = ["ObjectBasedOptimizer", "ObjectWorkspace", "extract_object", "extract_background"]