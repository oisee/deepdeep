"""
Transformation space exploration with multiple search strategies.
"""

from typing import List, Dict, Any, Tuple, Optional, Iterator
from dataclasses import dataclass
import numpy as np
import itertools
import time
from ..geometric.affine import TransformParams, TransformationEngine
from .constraints import ConstraintEvaluator
from ...utils.interactive import InteractiveHandler, InteractiveChoice

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    
    # Simple progress bar fallback
    def tqdm(iterable, desc="", total=None, disable=False):
        if disable or total is None:
            return iterable
        
        def progress_iter():
            start_time = time.time()
            for i, item in enumerate(iterable):
                if i % max(1, total // 20) == 0:  # Update every 5%
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    percent = (i + 1) / total * 100
                    print(f"\r{desc}: {percent:.1f}% ({i+1}/{total}) [{rate:.1f}it/s]", end="", flush=True)
                yield item
            print()  # New line when complete
        
        return progress_iter()


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
                              search_config: Optional[Dict[str, Any]] = None,
                              fine_grain_level: str = 'medium') -> List[TransformationResult]:
        """
        Generate and evaluate transformation variants.
        
        Args:
            image: Input image (H, W, 3)
            search_config: Search configuration parameters
            fine_grain_level: 'fast', 'medium', 'fine', 'ultra_fine'
            
        Returns:
            List of TransformationResult sorted by score (best first)
        """
        if search_config is None:
            search_config = self._get_default_search_config(fine_grain_level)
        
        results = []
        
        # Phase 1: Coarse grid search
        print("Phase 1: Coarse search...")
        try:
            coarse_results = self._coarse_search(image, search_config.get('coarse', {}))
            results.extend(coarse_results)
            
            # Save intermediate coarse results
            if search_config.get('save_intermediate', False):
                self._save_intermediate_results(coarse_results, "coarse", search_config.get('output_prefix', 'intermediate'))
                
        except KeyboardInterrupt:
            print(f"\n⚠️  Coarse search interrupted! Saving {len(results)} results found so far...")
            if results:
                self._save_intermediate_results(results, "coarse_interrupted", search_config.get('output_prefix', 'interrupted'))
            return sorted(results)[:search_config.get('max_results', 50)]
        
        # Phase 2: Fine-tune around best results
        if search_config.get('enable_fine_search', True):
            print("Phase 2: Fine search...")
            try:
                fine_results = self._fine_search(
                    image, 
                    coarse_results[:search_config.get('fine_candidates', 10)],
                    search_config.get('fine', {}),
                    search_config  # Pass config for interruption handling
                )
                results.extend(fine_results)
                
                # Save intermediate fine results
                if search_config.get('save_intermediate', False):
                    self._save_intermediate_results(fine_results, "fine", search_config.get('output_prefix', 'intermediate'))
                    
            except KeyboardInterrupt:
                print(f"\n⚠️  Fine search interrupted! Saving {len(results)} results found so far...")
                if results:
                    self._save_intermediate_results(results, "fine_interrupted", search_config.get('output_prefix', 'interrupted'))
                return sorted(results)[:search_config.get('max_results', 50)]
        
        # Phase 3: Try non-linear on best affine results
        if search_config.get('enable_nonlinear', True):
            print("Phase 3: Non-linear search...")
            try:
                nonlinear_results = self._explore_nonlinear(
                    image,
                    sorted(results)[:search_config.get('nonlinear_candidates', 5)]
                )
                results.extend(nonlinear_results)
                
                # Save intermediate nonlinear results
                if search_config.get('save_intermediate', False):
                    self._save_intermediate_results(nonlinear_results, "nonlinear", search_config.get('output_prefix', 'intermediate'))
                    
            except KeyboardInterrupt:
                print(f"\n⚠️  Non-linear search interrupted! Saving {len(results)} results found so far...")
                if results:
                    self._save_intermediate_results(results, "nonlinear_interrupted", search_config.get('output_prefix', 'interrupted'))
                return sorted(results)[:search_config.get('max_results', 50)]
        
        # Sort and return best results
        results.sort()
        return results[:search_config.get('max_results', 50)]
    
    def _get_default_search_config(self, fine_grain_level: str = 'medium') -> Dict[str, Any]:
        """
        Get default search configuration with configurable fine-grain level.
        
        Args:
            fine_grain_level: 'fast', 'medium', 'fine', 'ultra_fine'
        """
        # Define fine-grain level configurations
        fine_configs = {
            'fast': {
                'enable_fine_search': False,
                'enable_nonlinear': False,
                'max_results': 10,
                'coarse': {
                    'rotation_step': 10,
                    'scale_step': 0.2,
                    'translate_step': 16,
                    'max_combinations': 20
                }
            },
            'medium': {
                'enable_fine_search': True,
                'enable_nonlinear': False,
                'max_results': 25,
                'fine_candidates': 3,  # Reduced from 5
                'coarse': {
                    'rotation_step': 5,
                    'scale_step': 0.1,
                    'translate_step': 8,
                    'max_combinations': 50
                },
                'fine': {
                    'rotation_step': 3,      # Increased from 2
                    'scale_step': 0.08,      # Increased from 0.05
                    'translate_step': 6,     # Increased from 4
                    'search_radius': 1,      # Reduced from 2 - this is key!
                }
            },
            'fine': {
                'enable_fine_search': True,
                'enable_nonlinear': True,
                'max_results': 50,
                'fine_candidates': 10,
                'nonlinear_candidates': 5,
                'coarse': {
                    'rotation_step': 5,
                    'scale_step': 0.1,
                    'translate_step': 8,
                    'max_combinations': 100
                },
                'fine': {
                    'rotation_step': 0.5,
                    'scale_step': 0.01,
                    'translate_step': 1,
                    'search_radius': 3,
                }
            },
            'ultra_fine': {
                'enable_fine_search': True,
                'enable_nonlinear': True,
                'max_results': 100,
                'fine_candidates': 15,
                'nonlinear_candidates': 8,
                'coarse': {
                    'rotation_step': 2,
                    'scale_step': 0.05,
                    'translate_step': 4,
                    'max_combinations': 200
                },
                'fine': {
                    'rotation_step': 0.25,
                    'scale_step': 0.005,
                    'translate_step': 0.5,
                    'search_radius': 4,
                }
            }
        }
        
        base_config = {
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
                'max_combinations': 1000,
            },
            'fine': {
                'rotation_step': 0.5,
                'scale_step': 0.01,
                'translate_step': 1,
                'search_radius': 3,  # steps around best coarse result
            }
        }
        
        # Override with fine-grain specific settings
        if fine_grain_level in fine_configs:
            level_config = fine_configs[fine_grain_level]
            # Deep merge configurations
            for key, value in level_config.items():
                if isinstance(value, dict) and key in base_config:
                    base_config[key].update(value)
                else:
                    base_config[key] = value
        
        return base_config
    
    def _coarse_search(self, image: np.ndarray, config: Dict[str, Any]) -> List[TransformationResult]:
        """Coarse grid search across transformation space."""
        results = []
        
        # Generate parameter ranges
        rotations = self._generate_range(
            config.get('rotation_range', (-15, 15)),
            config.get('rotation_step', 5)
        )
        scales = self._generate_range(
            config.get('scale_range', (0.8, 1.2)),
            config.get('scale_step', 0.1)
        )
        
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
            rotations, scales, translates_x, translates_y
        ))
        
        # Limit combinations if too many
        if len(param_combinations) > max_combinations:
            # Sample uniformly
            indices = np.linspace(0, len(param_combinations) - 1, max_combinations, dtype=int)
            param_combinations = [param_combinations[i] for i in indices]
        
        # Evaluate each combination with progress bar and interactive handling
        param_combinations_progress = tqdm(
            param_combinations, 
            desc=f"Coarse search ({len(param_combinations)} combinations)",
            disable=len(param_combinations) < 10
        )
        
        # Get interactive handler if configured
        interactive_handler = config.get('interactive_handler')
        
        for rot, scale, tx, ty in param_combinations_progress:
            # Check for interactive trigger
            if interactive_handler:
                choice = interactive_handler.check_for_interactive_trigger()
                if choice == InteractiveChoice.EXIT:
                    print(f"\n🚪 User requested exit from coarse search. Saving {len(results)} results...")
                    break
                elif choice == InteractiveChoice.SKIP_PHASE:
                    print(f"\n⏭️  User requested skip coarse phase. Saving {len(results)} results...")
                    break
                elif choice == InteractiveChoice.SAVE_INTERMEDIATE:
                    if results:
                        self._save_intermediate_results(sorted(results)[:10], "coarse_manual", config.get('output_prefix', 'manual'))
            
            params = TransformParams(
                rotation=rot,
                scale=scale,
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
                    config: Dict[str, Any],
                    search_config: Optional[Dict[str, Any]] = None) -> List[TransformationResult]:
        """Fine search around best coarse results."""
        results = []
        
        radius = config.get('search_radius', 3)
        
        # Collect all fine variations first to show total progress
        all_fine_params = []
        for coarse_result in coarse_results:
            base_params = coarse_result.params
            fine_params = self._generate_fine_variations(base_params, config, radius)
            all_fine_params.extend(fine_params)
        
        # Process with progress bar and interruption handling
        fine_params_progress = tqdm(
            all_fine_params, 
            desc=f"Fine search ({len(all_fine_params)} variations)",
            disable=len(all_fine_params) < 10
        )
        
        # Get interactive handler if configured
        interactive_handler = search_config.get('interactive_handler') if search_config else None
        
        processed_count = 0
        for params in fine_params_progress:
            try:
                # Check for interactive trigger
                if interactive_handler:
                    choice = interactive_handler.check_for_interactive_trigger()
                    if choice == InteractiveChoice.EXIT:
                        print(f"\n🚪 User requested exit from fine search after {processed_count}/{len(all_fine_params)} variations!")
                        break
                    elif choice == InteractiveChoice.SKIP_PHASE:
                        print(f"\n⏭️  User requested skip fine phase after {processed_count}/{len(all_fine_params)} variations!")
                        break
                    elif choice == InteractiveChoice.SAVE_INTERMEDIATE:
                        if results:
                            self._save_intermediate_results(sorted(results)[:10], "fine_manual", search_config.get('output_prefix', 'manual'))
                
                result = self._evaluate_transformation(image, params)
                if result is not None:
                    results.append(result)
                processed_count += 1
                
                # Save intermediate results every 50 iterations if configured
                if (search_config and search_config.get('save_intermediate', False) and 
                    processed_count % 50 == 0 and results):
                    self._save_intermediate_results(
                        sorted(results)[:10], 
                        f"fine_checkpoint_{processed_count}", 
                        search_config.get('output_prefix', 'checkpoint')
                    )
                    
            except KeyboardInterrupt:
                print(f"\n⚠️  Ctrl+C: Fine search interrupted after {processed_count}/{len(all_fine_params)} variations!")
                break
        
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
                        params.scale += scale_delta * scale_step
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
                'scale': (0.8, 1.2),        # Max 20% scaling for faces/persons
                'translate_x': (-10, 10),   # Limited translation
                'translate_y': (-10, 10),
                'shear_x': (0, 0),          # No shear
                'shear_y': (0, 0),
                'perspective_x': (0, 0),    # No perspective
                'perspective_y': (0, 0),
            },
            'person': {
                'rotation': (-8, 8),        # Slightly more rotation than faces
                'scale': (0.8, 1.2),        # Max 20% scaling for faces/persons
                'translate_x': (-15, 15),   
                'translate_y': (-15, 15),
                'shear_x': (-2, 2),         # Minimal shear
                'shear_y': (-2, 2),
                'perspective_x': (0, 0),    # No perspective
                'perspective_y': (0, 0),
            },
            'text': {
                'rotation': (-2, 2),        # Almost no rotation
                'scale': (0.95, 1.05),      # Very minimal scaling for readability
                'translate_x': (-5, 5),
                'translate_y': (-5, 5),
                'shear_x': (0, 0),          # No shear for readability
                'shear_y': (0, 0),
                'perspective_x': (0, 0),
                'perspective_y': (0, 0),
            },
            'sprite': {
                'rotation': (-15, 15),      # Some rotation OK
                'scale': (0.8, 1.2),        # Moderate scaling
                'translate_x': (-20, 20),
                'translate_y': (-20, 20),
                'shear_x': (-5, 5),
                'shear_y': (-5, 5),
                'perspective_x': (-0.001, 0.001),  # Slight perspective OK
                'perspective_y': (-0.001, 0.001),
            },
            'abstract': {
                'rotation': (-30, 30),      # More artistic freedom
                'scale': (0.5, 1.5),        # Up to 50% scaling for abstract objects
                'translate_x': (-40, 40),
                'translate_y': (-40, 40),
                'shear_x': (-10, 10),
                'shear_y': (-10, 10),
                'perspective_x': (-0.002, 0.002),
                'perspective_y': (-0.002, 0.002),
                'barrel_k1': (-0.15, 0.15),
                'barrel_k2': (-0.1, 0.1),
            },
            'background': {
                'rotation': (-30, 30),      # Full artistic freedom
                'scale': (0.5, 1.5),        # Up to 50% scaling for backgrounds
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
    
    def _save_intermediate_results(self, 
                                 results: List[TransformationResult], 
                                 phase: str, 
                                 prefix: str = "intermediate") -> None:
        """Save intermediate results to disk for analysis."""
        if not results:
            return
            
        # Sort by best score
        sorted_results = sorted(results)
        top_results = sorted_results[:min(5, len(sorted_results))]  # Save top 5
        
        try:
            from PIL import Image
            import os
            
            # Create output directory if needed
            output_dir = f"{prefix}_{phase}"
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"\n💾 Saving {len(top_results)} best {phase} results to {output_dir}/")
            
            # Save each result with metadata
            for i, result in enumerate(top_results):
                # Save transformed image
                img_path = f"{output_dir}/result_{i+1}_score_{result.score:.3f}.png"
                Image.fromarray(result.transformed_image.astype('uint8')).save(img_path)
                
                # Save metadata
                meta_path = f"{output_dir}/result_{i+1}_metadata.txt"
                with open(meta_path, 'w') as f:
                    f.write(f"Score: {result.score:.6f}\n")
                    f.write(f"Quality Score: {result.quality_score:.6f}\n")
                    f.write(f"Constraint Violation: {result.constraint_violation:.6f}\n")
                    f.write(f"Transformation Parameters:\n")
                    f.write(f"  Rotation: {result.params.rotation:.2f}°\n")
                    f.write(f"  Scale: {result.params.scale:.3f}\n")
                    f.write(f"  Translate X: {result.params.translate_x:.1f}px\n")
                    f.write(f"  Translate Y: {result.params.translate_y:.1f}px\n")
                    if abs(result.params.shear_x) > 1e-6:
                        f.write(f"  Shear X: {result.params.shear_x:.2f}°\n")
                    if abs(result.params.shear_y) > 1e-6:
                        f.write(f"  Shear Y: {result.params.shear_y:.2f}°\n")
                    if abs(result.params.barrel_k1) > 1e-6:
                        f.write(f"  Barrel K1: {result.params.barrel_k1:.4f}\n")
                    if abs(result.params.wave_amp_x) > 1e-6:
                        f.write(f"  Wave Amp X: {result.params.wave_amp_x:.2f}\n")
                    f.write(f"\nDetails: {result.details}\n")
            
            # Create summary
            summary_path = f"{output_dir}/summary.txt"
            with open(summary_path, 'w') as f:
                f.write(f"Intermediate Results Summary - {phase.title()} Phase\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"Total results processed: {len(results)}\n")
                f.write(f"Results saved: {len(top_results)}\n")
                f.write(f"Best score: {sorted_results[0].score:.6f}\n")
                f.write(f"Worst score in top 5: {sorted_results[min(4, len(sorted_results)-1)].score:.6f}\n\n")
                
                f.write("Top Results:\n")
                for i, result in enumerate(top_results):
                    f.write(f"{i+1:2d}. Score: {result.score:.6f}, "
                           f"Quality: {result.quality_score:.3f}, "
                           f"Rotation: {result.params.rotation:+6.1f}°, "
                           f"Scale: {result.params.scale:.2f}x\n")
            
            print(f"✅ Saved results to {output_dir}/ (best score: {sorted_results[0].score:.3f})")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not save intermediate results: {e}")
            pass  # Don't fail the search if saving fails