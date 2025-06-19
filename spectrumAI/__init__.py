"""
SpectrumAI: Next-Generation ZX Spectrum Image Converter

A revolutionary image-to-ZX Spectrum converter using transformation space exploration,
perceptual modeling, and differentiable optimization.
"""

__version__ = "0.1.0"
__author__ = "SpectrumAI Team"
__license__ = "MIT"

from .transformations.geometric.affine import TransformParams, TransformationEngine
from .transformations.search.explorer import TransformationExplorer
from .segmentation.detectors import ObjectBasedOptimizer

__all__ = [
    "TransformParams",
    "TransformationEngine", 
    "TransformationExplorer",
    "ObjectBasedOptimizer"
]