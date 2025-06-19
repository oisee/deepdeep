"""Tests for transformation functionality."""

import pytest
import numpy as np
from spectrumAI.transformations.geometric.affine import TransformParams, TransformationEngine
from spectrumAI.transformations.search.explorer import TransformationExplorer
from spectrumAI.transformations.search.constraints import ConstraintEvaluator


class TestTransformParams:
    """Test TransformParams functionality."""
    
    def test_identity_transform(self):
        """Test identity transformation detection."""
        params = TransformParams()
        assert params.is_identity()
        
        params.rotation = 0.1
        assert not params.is_identity()
    
    def test_copy(self):
        """Test parameter copying."""
        params1 = TransformParams(rotation=10, scale_x=1.5)
        params2 = params1.copy()
        
        assert params1.rotation == params2.rotation
        assert params1.scale_x == params2.scale_x
        
        # Ensure deep copy
        params2.rotation = 20
        assert params1.rotation == 10


class TestTransformationEngine:
    """Test TransformationEngine functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = TransformationEngine()
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_identity_transform(self):
        """Test that identity transform returns unchanged image."""
        params = TransformParams()
        result = self.engine.apply_transform(self.test_image, params)
        
        np.testing.assert_array_equal(result, self.test_image)
    
    def test_rotation(self):
        """Test rotation transformation."""
        params = TransformParams(rotation=90)
        result = self.engine.apply_transform(self.test_image, params)
        
        assert result.shape == self.test_image.shape
        assert not np.array_equal(result, self.test_image)
    
    def test_scaling(self):
        """Test scaling transformation."""
        params = TransformParams(scale_x=1.5, scale_y=1.5)
        result = self.engine.apply_transform(self.test_image, params)
        
        assert result.shape == self.test_image.shape
    
    def test_translation(self):
        """Test translation transformation."""
        params = TransformParams(translate_x=10, translate_y=5)
        result = self.engine.apply_transform(self.test_image, params)
        
        assert result.shape == self.test_image.shape
        assert not np.array_equal(result, self.test_image)
    
    def test_combined_transforms(self):
        """Test combination of multiple transformations."""
        params = TransformParams(
            rotation=15,
            scale_x=1.2,
            scale_y=1.2,
            translate_x=5,
            translate_y=-3
        )
        result = self.engine.apply_transform(self.test_image, params)
        
        assert result.shape == self.test_image.shape
        assert not np.array_equal(result, self.test_image)


class TestConstraintEvaluator:
    """Test ConstraintEvaluator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = ConstraintEvaluator('standard')
        # Create test image with known color properties
        self.test_image = np.zeros((64, 64, 3), dtype=np.uint8)
        self.test_image[:32, :32] = [255, 0, 0]  # Red square
        self.test_image[32:, 32:] = [0, 255, 0]  # Green square
    
    def test_perfect_image(self):
        """Test evaluation of image that perfectly fits constraints."""
        # Create 2-color 8x8 blocks
        perfect_image = np.zeros((16, 16, 3), dtype=np.uint8)
        perfect_image[:8, :8] = [0, 0, 0]      # Black
        perfect_image[8:, 8:] = [255, 255, 255]  # White
        
        violation = self.evaluator.evaluate(perfect_image)
        assert violation.score == 0.0
    
    def test_violating_image(self):
        """Test evaluation of image that violates constraints."""
        # Create image with multiple colors that don't match ZX palette
        violating_image = np.zeros((16, 16, 3), dtype=np.uint8)
        # Create an 8x8 block with 4 different non-ZX colors
        violating_image[:8, :8] = [100, 50, 150]   # Purple-ish (not in ZX palette)
        violating_image[:4, :4] = [200, 100, 50]   # Orange-ish (not in ZX palette)
        violating_image[4:8, :4] = [50, 150, 100]  # Green-ish (not in ZX palette)
        violating_image[:4, 4:8] = [150, 50, 200]  # Magenta-ish (not in ZX palette)
        
        violation = self.evaluator.evaluate(violating_image)
        assert violation.score > 0.0
    
    def test_quality_score(self):
        """Test quality score calculation."""
        score = self.evaluator.estimate_quality_score(self.test_image)
        assert 0.0 <= score <= 1.0
    
    def test_different_modes(self):
        """Test different evaluation modes."""
        evaluators = {
            'standard': ConstraintEvaluator('standard'),
            'gigascreen': ConstraintEvaluator('gigascreen'),
            'mc8x4': ConstraintEvaluator('mc8x4')
        }
        
        for mode, evaluator in evaluators.items():
            violation = evaluator.evaluate(self.test_image)
            assert violation.score >= 0.0
            assert 'avg_block_violation' in violation.details


class TestTransformationExplorer:
    """Test TransformationExplorer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.explorer = TransformationExplorer('standard')
        # Create small test image for faster testing
        self.test_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    
    def test_exploration_returns_results(self):
        """Test that exploration returns results."""
        # Use limited search config for speed
        search_config = {
            'max_results': 5,
            'enable_fine_search': False,
            'enable_nonlinear': False,
            'coarse': {
                'rotation_step': 10,
                'scale_step': 0.2,
                'translate_step': 5,
                'rotation_range': (-10, 10),
                'scale_range': (0.9, 1.1),
                'translate_range': (-5, 5),
                'max_combinations': 20
            }
        }
        
        results = self.explorer.explore_transformations(self.test_image, search_config)
        
        assert len(results) > 0
        assert all(hasattr(r, 'score') for r in results)
        assert all(hasattr(r, 'params') for r in results)
        assert all(hasattr(r, 'transformed_image') for r in results)
    
    def test_results_sorted_by_score(self):
        """Test that results are sorted by score (lower is better)."""
        search_config = {
            'max_results': 3,
            'enable_fine_search': False,
            'enable_nonlinear': False,
            'coarse': {
                'max_combinations': 10
            }
        }
        
        results = self.explorer.explore_transformations(self.test_image, search_config)
        
        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores)
    
    def test_object_constraints(self):
        """Test object-specific constraints."""
        face_constraints = self.explorer.get_object_constraints('face')
        text_constraints = self.explorer.get_object_constraints('text')
        
        assert 'rotation' in face_constraints
        assert 'rotation' in text_constraints
        
        # Face should have more restrictive rotation than background
        face_rot_range = face_constraints['rotation'][1] - face_constraints['rotation'][0]
        text_rot_range = text_constraints['rotation'][1] - text_constraints['rotation'][0]
        
        assert face_rot_range <= 10  # Face rotation should be small
        assert text_rot_range <= 5   # Text rotation should be very small


if __name__ == '__main__':
    pytest.main([__file__])