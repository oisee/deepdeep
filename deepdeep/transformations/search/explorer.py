"""
Transformation space exploration with multiple search strategies.
"""

from typing import List, Dict, Any, Tuple, Optional, Iterator
from dataclasses import dataclass
import numpy as np
import itertools
from ..geometric.affine import TransformParams, TransformationEngine
from .constraints import ConstraintEvaluator


@dataclass
class TransformationResult:
    """Result of a transformation exploration."""
    params: TransformParams
    transformed_image: np.ndarray
    score: float
    constraint_violation: float
    quality_score: float
    details: Dict[str, Any]
    
    def __lt__(self, other):
        """For sorting - lower score is better."""
        return self.score < other.score


class TransformationExplorer:
    """Explores transformation space intelligently."""
    
    def __init__(self, mode: str = 'standard'):
        self.mode = mode
        self.engine = TransformationEngine()
        self.evaluator = ConstraintEvaluator(mode)
        
    def explore_transformations(self, 
                              image: np.ndarray,
                              search_config: Optional[Dict[str, Any]] = None) -> List[TransformationResult]:
        """
        Generate and evaluate transformation variants.
        
        Args:
            image: Input image (H, W, 3)
            search_config: Search configuration parameters
            
        Returns:
            List of TransformationResult sorted by score (best first)
        """
        if search_config is None:
            search_config = self._get_default_search_config()
        
        results = []
        
        # Phase 1: Coarse grid search
        print("Phase 1: Coarse search...")
        coarse_results = self._coarse_search(image, search_config.get('coarse', {}))
        results.extend(coarse_results)
        
        # Phase 2: Fine-tune around best results
        if search_config.get('enable_fine_search', True):
            print("Phase 2: Fine search...")
            fine_results = self._fine_search(
                image, 
                coarse_results[:search_config.get('fine_candidates', 10)],
                search_config.get('fine', {})
            )
            results.extend(fine_results)
        
        # Phase 3: Try non-linear on best affine results
        if search_config.get('enable_nonlinear', True):
            print("Phase 3: Non-linear search...")
            nonlinear_results = self._explore_nonlinear(
                image,
                sorted(results)[:search_config.get('nonlinear_candidates', 5)]
            )
            results.extend(nonlinear_results)
        
        # Sort and return best results
        results.sort()
        return results[:search_config.get('max_results', 50)]
    
    def _get_default_search_config(self) -> Dict[str, Any]:
        """Get default search configuration."""
        return {
            'max_results': 50,
            'enable_fine_search': True,
            'enable_nonlinear': True,
            'fine_candidates': 10,
            'nonlinear_candidates': 5,
            'coarse': {
                'rotation_step': 5,      # degrees
                'scale_step': 0.1,       # scale increment
                'translate_step': 8,     # pixels
                'rotation_range': (-15, 15),
                'scale_range': (0.8, 1.2),
                'translate_range': (-20, 20),
            },
            'fine': {
                'rotation_step': 0.5,
                'scale_step': 0.01,
                'translate_step': 1,
                'search_radius': 3,  # steps around best coarse result
            }
        }
    
    def _coarse_search(self, image: np.ndarray, config: Dict[str, Any]) -> List[TransformationResult]:
        """Coarse grid search across transformation space."""
        results = []
        
        # Generate parameter ranges
        rotations = self._generate_range(
            config.get('rotation_range', (-15, 15)),
            config.get('rotation_step', 5)
        )
        scales_x = self._generate_range(
            config.get('scale_range', (0.8, 1.2)),
            config.get('scale_step', 0.1)
        )
        scales_y = scales_x  # Keep aspect ratio for coarse search
        
        translates_x = self._generate_range(
            config.get('translate_range', (-20, 20)),
            config.get('translate_step', 8)
        )
        translates_y = self._generate_range(
            config.get('translate_range', (-20, 20)),
            config.get('translate_step', 8)
        )
        
        # Generate all combinations (limited to avoid explosion)
        max_combinations = config.get('max_combinations', 1000)
        
        param_combinations = list(itertools.product(
            rotations, scales_x, scales_y, translates_x, translates_y
        ))
        
        # Limit combinations if too many
        if len(param_combinations) > max_combinations:
            # Sample uniformly
            indices = np.linspace(0, len(param_combinations) - 1, max_combinations, dtype=int)
            param_combinations = [param_combinations[i] for i in indices]
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        # Evaluate each combination
        for i, (rot, sx, sy, tx, ty) in enumerate(param_combinations):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(param_combinations)}")
                
            params = TransformParams(
                rotation=rot,
                scale_x=sx,
                scale_y=sy,
                translate_x=tx,
                translate_y=ty
            )
            
            result = self._evaluate_transformation(image, params)
            if result is not None:
                results.append(result)
        
        # Sort and return best results
        results.sort()
        return results[:50]  # Top 50 coarse results
    
    def _fine_search(self, 
                    image: np.ndarray,
                    coarse_results: List[TransformationResult],
                    config: Dict[str, Any]) -> List[TransformationResult]:
        """Fine search around best coarse results."""
        results = []
        
        radius = config.get('search_radius', 3)
        
        for coarse_result in coarse_results:
            base_params = coarse_result.params
            
            # Generate fine parameter variations around base
            fine_params = self._generate_fine_variations(base_params, config, radius)
            
            for params in fine_params:
                result = self._evaluate_transformation(image, params)
                if result is not None:
                    results.append(result)
        
        return results
    
    def _explore_nonlinear(self, 
                          image: np.ndarray,
                          base_results: List[TransformationResult]) -> List[TransformationResult]:
        """Explore non-linear transformations on best affine results."""
        results = []
        
        # Try barrel distortion
        barrel_values = [-0.2, -0.1, 0.1, 0.2]
        
        # Try wave distortion  
        wave_configs = [
            {'wave_amp_x': 2, 'wave_freq_x': 0.02},
            {'wave_amp_y': 2, 'wave_freq_y': 0.02},
            {'wave_amp_x': 1, 'wave_freq_x': 0.05, 'wave_amp_y': 1, 'wave_freq_y': 0.05},
        ]
        
        for base_result in base_results:
            base_params = base_result.params
            
            # Try barrel distortion
            for barrel_k1 in barrel_values:
                params = base_params.copy()
                params.barrel_k1 = barrel_k1
                
                result = self._evaluate_transformation(image, params)
                if result is not None:
                    results.append(result)
            
            # Try wave distortion
            for wave_config in wave_configs:
                params = base_params.copy()
                for key, value in wave_config.items():
                    setattr(params, key, value)
                
                result = self._evaluate_transformation(image, params)
                if result is not None:
                    results.append(result)
        
        return results
    
    def _generate_range(self, range_tuple: Tuple[float, float], step: float) -> List[float]:
        """Generate values in range with given step."""
        start, end = range_tuple
        return list(np.arange(start, end + step, step))
    
    def _generate_fine_variations(self, 
                                 base_params: TransformParams, 
                                 config: Dict[str, Any],
                                 radius: int) -> List[TransformParams]:
        """Generate fine parameter variations around base parameters."""
        variations = []
        
        rot_step = config.get('rotation_step', 0.5)
        scale_step = config.get('scale_step', 0.01)
        trans_step = config.get('translate_step', 1)
        
        # Generate variations in each dimension
        for rot_delta in range(-radius, radius + 1):
            for scale_delta in range(-radius, radius + 1):
                for trans_x_delta in range(-radius, radius + 1):
                    for trans_y_delta in range(-radius, radius + 1):
                        params = base_params.copy()
                        params.rotation += rot_delta * rot_step
                        params.scale_x += scale_delta * scale_step
                        params.scale_y += scale_delta * scale_step  # Keep aspect ratio
                        params.translate_x += trans_x_delta * trans_step
                        params.translate_y += trans_y_delta * trans_step
                        
                        variations.append(params)
        
        return variations
    
    def _evaluate_transformation(self, 
                               image: np.ndarray, 
                               params: TransformParams) -> Optional[TransformationResult]:
        """Evaluate a single transformation."""
        try:
            # Apply transformation
            transformed = self.engine.apply_transform(image, params)
            
            # Evaluate constraints
            violation = self.evaluator.evaluate(transformed)
            quality = self.evaluator.estimate_quality_score(transformed)
            
            # Calculate combined score (lower is better)
            # Weight constraint violation heavily, but also consider quality
            score = violation.score * 2.0 + (1.0 - quality) * 0.5
            
            return TransformationResult(
                params=params,
                transformed_image=transformed,
                score=score,
                constraint_violation=violation.score,
                quality_score=quality,
                details=violation.details
            )
            
        except Exception as e:
            # Skip transformations that fail
            print(f"Transformation failed: {e}")
            return None
    
    def quick_evaluate(self, image: np.ndarray, params: TransformParams) -> float:
        """Quick evaluation for optimization algorithms."""
        result = self._evaluate_transformation(image, params)
        return result.score if result is not None else float('inf')
    
    def get_object_constraints(self, obj_class: str) -> Dict[str, Tuple[float, float]]:
        """Get transformation constraints for different object types."""
        constraints = {
            'face': {
                'rotation': (-5, 5),        # Minimal rotation for faces
                'scale_x': (0.9, 1.1),      # Minimal scaling
                'scale_y': (0.9, 1.1),
                'translate_x': (-10, 10),   # Limited translation
                'translate_y': (-10, 10),
                'shear_x': (0, 0),          # No shear
                'shear_y': (0, 0),
                'perspective_x': (0, 0),    # No perspective
                'perspective_y': (0, 0),
            },
            'text': {
                'rotation': (-2, 2),        # Almost no rotation
                'scale_x': (0.95, 1.05),    # Very minimal scaling
                'scale_y': (0.95, 1.05),
                'translate_x': (-5, 5),
                'translate_y': (-5, 5),
                'shear_x': (0, 0),          # No shear for readability
                'shear_y': (0, 0),
                'perspective_x': (0, 0),
                'perspective_y': (0, 0),
            },
            'sprite': {
                'rotation': (-15, 15),      # Some rotation OK
                'scale_x': (0.8, 1.2),      # More scaling freedom
                'scale_y': (0.8, 1.2),
                'translate_x': (-20, 20),
                'translate_y': (-20, 20),
                'shear_x': (-5, 5),
                'shear_y': (-5, 5),
                'perspective_x': (-0.001, 0.001),  # Slight perspective OK
                'perspective_y': (-0.001, 0.001),
            },
            'background': {
                'rotation': (-30, 30),      # More freedom
                'scale_x': (0.5, 1.5),      # Can scale more
                'scale_y': (0.5, 1.5),
                'translate_x': (-50, 50),
                'translate_y': (-50, 50),
                'shear_x': (-15, 15),
                'shear_y': (-15, 15),
                'perspective_x': (-0.002, 0.002),
                'perspective_y': (-0.002, 0.002),
                'barrel_k1': (-0.2, 0.2),   # Can distort backgrounds
                'barrel_k2': (-0.1, 0.1),
            }
        }
        
        return constraints.get(obj_class, constraints['background'])