"""
Command-line interface for DeepDeep.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

from .segmentation.detectors import ObjectBasedOptimizer
from .transformations.composition.recomposer import RecompositionEngine
from .transformations.search.constraints import ConstraintEvaluator


def load_image(image_path: str) -> np.ndarray:
    """Load image from file."""
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        return np.array(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        sys.exit(1)


def adjust_contrast(image: np.ndarray, contrast_factor: float = 1.0, brightness_offset: float = 0.0, auto_curves: bool = False) -> np.ndarray:
    """
    Adjust contrast and brightness to improve dithering effectiveness.
    
    Args:
        image: Input image (H, W, C)
        contrast_factor: Contrast multiplier (1.0=no change, >1.0=more contrast, <1.0=less contrast)
        brightness_offset: Brightness offset (-100 to +100)
        auto_curves: Apply histogram stretching to maximize dynamic range
        
    Returns:
        Contrast-adjusted image
    """
    # Convert to float for precise calculations
    adjusted = image.astype(np.float32)
    
    # Apply auto curves (histogram stretching) first if requested
    if auto_curves:
        # Find current min/max per channel to preserve color balance
        for c in range(adjusted.shape[2]):
            channel = adjusted[:, :, c]
            current_min = np.min(channel)
            current_max = np.max(channel)
            
            # Avoid division by zero
            if current_max > current_min:
                # Stretch to full 0-255 range
                adjusted[:, :, c] = (channel - current_min) / (current_max - current_min) * 255.0
    
    # Apply manual adjustments if specified
    if abs(contrast_factor - 1.0) > 1e-6 or abs(brightness_offset) > 1e-6:
        # Apply contrast: new_value = (old_value - 128) * contrast + 128
        # This centers the contrast adjustment around middle gray
        adjusted = (adjusted - 128.0) * contrast_factor + 128.0
        
        # Apply brightness offset
        adjusted = adjusted + brightness_offset
    
    # Clip to valid range
    adjusted = np.clip(adjusted, 0, 255)
    
    return adjusted.astype(np.uint8)


def analyze_image_contrast(image: np.ndarray) -> dict:
    """
    Analyze image contrast characteristics to suggest optimal settings.
    
    Args:
        image: Input image (H, W, C)
        
    Returns:
        Dictionary with contrast analysis and recommendations
    """
    # Convert to grayscale for analysis
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image.copy()
    
    # Calculate contrast metrics
    std_dev = np.std(gray)
    range_span = np.max(gray) - np.min(gray)
    mean_brightness = np.mean(gray)
    
    # Calculate histogram distribution
    hist, bins = np.histogram(gray, bins=64, range=(0, 256))
    
    # Find where most pixels are concentrated
    peak_bin = np.argmax(hist)
    peak_value = bins[peak_bin]
    
    # Determine contrast level
    if std_dev < 30:
        contrast_level = "very_low"
        recommended_contrast = 1.5
        dithering_effectiveness = "poor"
    elif std_dev < 50:
        contrast_level = "low"
        recommended_contrast = 1.3
        dithering_effectiveness = "fair"
    elif std_dev < 80:
        contrast_level = "moderate"
        recommended_contrast = 1.1
        dithering_effectiveness = "good"
    else:
        contrast_level = "high"
        recommended_contrast = 1.0
        dithering_effectiveness = "excellent"
    
    # Brightness adjustment recommendation
    if mean_brightness < 100:
        recommended_brightness = 20
        brightness_level = "dark"
    elif mean_brightness > 180:
        recommended_brightness = -20
        brightness_level = "bright"
    else:
        recommended_brightness = 0
        brightness_level = "balanced"
    
    return {
        'std_dev': float(std_dev),
        'range_span': float(range_span),
        'mean_brightness': float(mean_brightness),
        'peak_value': float(peak_value),
        'contrast_level': contrast_level,
        'brightness_level': brightness_level,
        'dithering_effectiveness': dithering_effectiveness,
        'recommended_contrast': recommended_contrast,
        'recommended_brightness': recommended_brightness,
        'needs_adjustment': std_dev < 60 or mean_brightness < 80 or mean_brightness > 200
    }


def crop_to_spectrum_screen(image: np.ndarray, crop_mode: str = 'c') -> np.ndarray:
    """
    Crop image to ZX Spectrum standard screen size (256x192).
    
    Args:
        image: Input image (H, W, C)
        crop_mode: Crop position - 'c' (center), 't' (top), 'b' (bottom), 'l' (left), 'r' (right),
                   'tl' (top-left), 'tr' (top-right), 'bl' (bottom-left), 'br' (bottom-right)
    
    Returns:
        Cropped image (192, 256, C)
    """
    h, w = image.shape[:2]
    target_w, target_h = 256, 192
    
    print(f"📐 Cropping from {(w, h)} to ZX Spectrum screen {(target_w, target_h)} ({crop_mode})")
    
    # Calculate crop coordinates based on mode
    if crop_mode == 'c':  # center
        start_x = max(0, (w - target_w) // 2)
        start_y = max(0, (h - target_h) // 2)
    elif crop_mode == 't':  # top
        start_x = max(0, (w - target_w) // 2)
        start_y = 0
    elif crop_mode == 'b':  # bottom
        start_x = max(0, (w - target_w) // 2)
        start_y = max(0, h - target_h)
    elif crop_mode == 'l':  # left
        start_x = 0
        start_y = max(0, (h - target_h) // 2)
    elif crop_mode == 'r':  # right
        start_x = max(0, w - target_w)
        start_y = max(0, (h - target_h) // 2)
    elif crop_mode == 'tl':  # top-left
        start_x = 0
        start_y = 0
    elif crop_mode == 'tr':  # top-right
        start_x = max(0, w - target_w)
        start_y = 0
    elif crop_mode == 'bl':  # bottom-left
        start_x = 0
        start_y = max(0, h - target_h)
    elif crop_mode == 'br':  # bottom-right
        start_x = max(0, w - target_w)
        start_y = max(0, h - target_h)
    else:
        raise ValueError(f"Invalid crop_mode: {crop_mode}. Use 'c', 't', 'b', 'l', 'r', 'tl', 'tr', 'bl', 'br'")
    
    # Ensure we don't exceed image boundaries
    end_x = min(start_x + target_w, w)
    end_y = min(start_y + target_h, h)
    actual_w = end_x - start_x
    actual_h = end_y - start_y
    
    # Crop the image
    cropped = image[start_y:end_y, start_x:end_x]
    
    # If image is smaller than target, pad with black
    if actual_w < target_w or actual_h < target_h:
        padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        
        # Center the cropped content in the padded area
        pad_start_y = (target_h - actual_h) // 2
        pad_start_x = (target_w - actual_w) // 2
        padded[pad_start_y:pad_start_y + actual_h, pad_start_x:pad_start_x + actual_w] = cropped
        
        print(f"🔲 Padded to target size: {(actual_w, actual_h)} → {(target_w, target_h)}")
        return padded
    
    print(f"✂️  Cropped: {(start_x, start_y)} to {(end_x, end_y)}")
    return cropped


def fit_image_to_spectrum(image: np.ndarray, fit_mode: str = 'h', target_size: int = 256, resize_factor: float = 1.0) -> np.ndarray:
    """
    Fit image to ZX Spectrum proportions and optionally resize.
    
    Args:
        image: Input image (H, W, C)
        fit_mode: 'h' for horizontal fit, 'v' for vertical fit
        target_size: Target dimension size (width for 'h', height for 'v')
                    Common ZX Spectrum sizes:
                    - 256 (standard screen width)
                    - 320 (high-res mode)
                    - 192 (standard screen height)
        resize_factor: Additional resize factor after fitting
        
    Returns:
        Processed image
    """
    h, w = image.shape[:2]
    original_size = (w, h)
    
    if fit_mode == 'h':
        # Fit to target width (horizontal)
        if w != target_size:
            # Calculate height maintaining aspect ratio
            new_width = target_size
            new_height = int(h * (target_size / w))
            fitted_size = (new_width, new_height)
            print(f"📐 Fitting horizontally: {original_size} → {fitted_size}")
        else:
            fitted_size = original_size
            print(f"📐 Image already fits horizontally: {original_size}")
    
    elif fit_mode == 'v':
        # Fit to target height (vertical)
        if h != target_size:
            # Calculate width maintaining aspect ratio
            new_height = target_size
            new_width = int(w * (target_size / h))
            fitted_size = (new_width, new_height)
            print(f"📐 Fitting vertically: {original_size} → {fitted_size}")
        else:
            fitted_size = original_size
            print(f"📐 Image already fits vertically: {original_size}")
    
    else:
        raise ValueError(f"Invalid fit_mode: {fit_mode}. Use 'h' or 'v'")
    
    # Apply fitting if needed
    if fitted_size != original_size:
        fitted_image = cv2.resize(image, fitted_size, interpolation=cv2.INTER_LANCZOS4)
    else:
        fitted_image = image.copy()
    
    # Apply additional resize factor if specified
    if abs(resize_factor - 1.0) > 1e-6:
        final_width = int(fitted_size[0] * resize_factor)
        final_height = int(fitted_size[1] * resize_factor)
        final_size = (final_width, final_height)
        
        final_image = cv2.resize(fitted_image, final_size, interpolation=cv2.INTER_LANCZOS4)
        print(f"🔧 Additional resize factor {resize_factor}: {fitted_size} → {final_size}")
        return final_image
    
    return fitted_image


def save_image(image: np.ndarray, output_path: str):
    """Save image to file."""
    try:
        img = Image.fromarray(image.astype(np.uint8))
        img.save(output_path)
        print(f"Saved result to {output_path}")
    except Exception as e:
        print(f"Error saving image {output_path}: {e}")
        sys.exit(1)


def _save_dithering_variants(dithered_image: np.ndarray, analysis: dict, base_path: str, method: str):
    """Save dithering variants and analysis results."""
    import json
    
    # Save scaled perception variants
    scales = [1.0, 0.5, 0.25]
    h, w = dithered_image.shape[:2]
    
    for scale in scales:
        if scale == 1.0:
            variant = dithered_image.copy()
            suffix = "_dithered"
        else:
            # Create downscaled version to show perceived color mixing
            new_h, new_w = int(h * scale), int(w * scale)
            downscaled = cv2.resize(dithered_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            variant = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_NEAREST)  # Pixelated upscale
            suffix = f"_dithered_x{scale}"
        
        variant_path = f"{base_path}{suffix}.png"
        save_image(variant, variant_path)
        print(f"💾 Saved {method} variant: {variant_path}")
    
    # Save perception analysis as JSON
    analysis_path = f"{base_path}_dithering_analysis.json"
    try:
        # Convert numpy arrays to lists for JSON serialization
        json_analysis = {}
        for key, value in analysis.items():
            if key == 'scales':
                json_analysis[key] = {}
                for scale_key, scale_data in value.items():
                    json_analysis[key][scale_key] = {}
                    for data_key, data_value in scale_data.items():
                        if data_key == 'color_distribution':
                            # Convert color tuples to strings
                            json_analysis[key][scale_key][data_key] = {
                                str(color): count for color, count in data_value.items()
                            }
                        elif data_key == 'dominant_colors':
                            # Convert to serializable format
                            json_analysis[key][scale_key][data_key] = [
                                [list(color), count] for color, count in data_value
                            ]
                        else:
                            json_analysis[key][scale_key][data_key] = float(data_value) if isinstance(data_value, (np.int64, np.float64)) else data_value
            else:
                json_analysis[key] = float(value) if isinstance(value, (np.int64, np.float64)) else value
        
        with open(analysis_path, 'w') as f:
            json.dump(json_analysis, f, indent=2)
        print(f"📊 Saved perception analysis: {analysis_path}")
        
    except Exception as e:
        print(f"⚠️  Could not save analysis: {e}")


def process_image(image_path: str, mode: str, output_path: str, interactive: bool = False, quality: str = 'medium', save_intermediate: bool = False, single_object: bool = False, dithering: str = 'none', fit_mode: str = 'h', target_size: int = 256, resize_factor: float = 1.0, contrast_factor: float = 0.0, brightness_offset: float = 0.0, auto_contrast: bool = False, auto_curves: bool = False, crop_mode: str = None):
    """Process a single image through the SpectrumAI pipeline."""
    print(f"Processing {image_path} in {mode} mode...")
    
    # Load image
    image = load_image(image_path)
    print(f"Loaded image: {image.shape}")
    
    # Analyze contrast before any processing
    contrast_analysis = analyze_image_contrast(image)
    print(f"📊 Image analysis: {contrast_analysis['contrast_level']} contrast, {contrast_analysis['brightness_level']} brightness")
    print(f"🎨 Dithering effectiveness: {contrast_analysis['dithering_effectiveness']}")
    
    # Apply contrast adjustment
    if auto_curves:
        print("📈 Auto curves: stretching histogram to maximize dynamic range")
        image = adjust_contrast(image, auto_curves=True)
    elif auto_contrast:
        if contrast_analysis['needs_adjustment']:
            final_contrast = contrast_analysis['recommended_contrast']
            final_brightness = contrast_analysis['recommended_brightness']
            print(f"🔧 Auto-adjusting: contrast {final_contrast:.1f}x, brightness {final_brightness:+.0f}")
            image = adjust_contrast(image, final_contrast, final_brightness)
        else:
            print("✅ Image contrast already optimal for dithering")
    elif contrast_factor != 0.0:
        # Manual contrast adjustment (0.0 means no adjustment)
        actual_contrast = 1.0 + contrast_factor
        print(f"🔧 Manual contrast adjustment: {actual_contrast:.1f}x, brightness {brightness_offset:+.0f}")
        image = adjust_contrast(image, actual_contrast, brightness_offset)
    
    # Apply fitting and resizing if requested
    current_dimension = image.shape[1 if fit_mode == 'h' else 0]
    if target_size != current_dimension or abs(resize_factor - 1.0) > 1e-6:
        image = fit_image_to_spectrum(image, fit_mode, target_size, resize_factor)
        print(f"Pre-processed image: {image.shape}")
    
    # Apply cropping if requested
    if crop_mode:
        image = crop_to_spectrum_screen(image, crop_mode)
        print(f"Cropped image: {image.shape}")
    
    # Performance estimate
    total_pixels = image.shape[0] * image.shape[1] 
    if total_pixels < 8000:
        print("⚡ Very fast processing expected")
    elif total_pixels < 32000:
        print("🚀 Fast processing expected") 
    elif total_pixels < 128000:
        print("⏱️  Standard processing time")
    else:
        print("🐌 Slower processing - consider using --resize-factor < 1.0 for faster results")
    
    # Initialize components
    evaluator = ConstraintEvaluator(mode)
    
    # Check if single-object mode is requested
    if single_object:
        print("🎯 Single-object mode: Processing entire image as one object...")
        workspaces = []
    else:
        optimizer = ObjectBasedOptimizer(mode)
        recomposer = RecompositionEngine(mode)
        
        # Segment and optimize objects
        workspaces = optimizer.segment_and_optimize(image)
    
    if not workspaces:
        print("No objects detected, processing as single object...")
        # Process entire image as one object
        from .transformations.search.explorer import TransformationExplorer
        explorer = TransformationExplorer(mode)
        
        # Use quality-based configuration
        print(f"Using {quality} quality level")
        
        # Create search configuration with interactive and intermediate options
        search_config = explorer._get_default_search_config(quality)
        
        if save_intermediate:
            search_config['save_intermediate'] = True
            # Use _out/_intermediate folder for intermediate outputs
            base_name = Path(image_path).stem
            intermediate_dir = Path("_out") / "_intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            search_config['output_prefix'] = str(intermediate_dir / f"{base_name}_intermediate")
            print("💾 Intermediate result saving enabled")
        
        if interactive:
            from .utils.interactive import InteractiveHandler
            search_config['interactive_handler'] = InteractiveHandler()
            print("🎛️  Interactive mode enabled - Create 'menu.trigger' or '.pause' file for menu")
        
        results = explorer.explore_transformations(image, search_config=search_config, fine_grain_level=quality)
        
        if results:
            best_result = results[0]
            
            # Generate both outputs
            transformed_original = best_result.transformed_image
            
            # Apply dithering if requested
            use_dithering = dithering != 'none'
            
            if use_dithering:
                print(f"🎨 Applying {dithering} dithering...")
                zx_quantized = evaluator._quantize_to_palette(transformed_original, use_dithering, dithering)
                
                # Perform color perception analysis
                print("🔍 Analyzing color perception at different scales...")
                from .utils.dithering import ZXDithering
                ditherer = ZXDithering()
                perception_analysis = ditherer.analyze_color_perception(zx_quantized)
                print(f"📊 {perception_analysis['summary']}")
                
                # Save additional analysis outputs
                base_path = output_path.replace('.png', '')
                _save_dithering_variants(zx_quantized, perception_analysis, base_path, dithering)
                
            else:
                zx_quantized = evaluator._quantize_to_palette(transformed_original, use_dithering, dithering)
            
            # Save transformed original (shows geometric transformation)
            base_path = output_path.replace('.png', '')
            transformed_path = f"{base_path}_transformed.png"
            save_image(transformed_original, transformed_path)
            print(f"Saved transformed original to: {transformed_path}")
            
            # Save ZX Spectrum quantized version (final result)
            zx_path = f"{base_path}_spectrum.png"
            save_image(zx_quantized, zx_path)
            print(f"Saved ZX Spectrum version to: {zx_path}")
            
            # Also save as main output (ZX version)
            save_image(zx_quantized, output_path)
            print(f"Best transformation score: {best_result.score:.3f}")
            print(f"Quality score: {best_result.quality_score:.3f}")
        else:
            print("No improvements found, saving quantized original")
            use_dithering = dithering != 'none'
            if use_dithering:
                print(f"🎨 Applying {dithering} dithering...")
                zx_quantized = evaluator._quantize_to_palette(image, use_dithering, dithering)
                
                # Perform color perception analysis  
                print("🔍 Analyzing color perception at different scales...")
                from .utils.dithering import ZXDithering
                ditherer = ZXDithering()
                perception_analysis = ditherer.analyze_color_perception(zx_quantized)
                print(f"📊 {perception_analysis['summary']}")
                
                # Save additional analysis outputs
                base_path = output_path.replace('.png', '')
                _save_dithering_variants(zx_quantized, perception_analysis, base_path, dithering)
            else:
                zx_quantized = evaluator._quantize_to_palette(image, use_dithering, dithering)
            save_image(zx_quantized, output_path)
        
        return
    
    # Print optimization summary
    summary = optimizer.get_optimization_summary(workspaces)
    print(f"\nOptimization Summary:")
    print(f"Total objects: {summary['total_objects']}")
    print(f"Objects by class: {summary['objects_by_class']}")
    print(f"Successful optimizations: {summary['optimization_stats']['successful']}")
    
    if interactive:
        # Interactive mode - let user select variants
        selected_variants = interactive_selection(workspaces)
    else:
        # Auto mode - select best variant for each object
        selected_variants = {}
        for ws in workspaces:
            if ws.optimization_results:
                selected_variants[ws.id] = 0  # Best variant
            else:
                selected_variants[ws.id] = 0  # Original
    
    # Recompose image
    print("\nRecomposing image...")
    result = recomposer.auto_compose(workspaces, selected_variants)
    
    # Generate both outputs for recomposed result
    recomposed_original = result.canvas
    use_dithering = dithering != 'none'
    if use_dithering:
        print(f"🎨 Applying {dithering} dithering to recomposed result...")
    recomposed_zx = evaluator._quantize_to_palette(recomposed_original, use_dithering, dithering)
    
    # Save transformed recomposition (shows object transformations)
    base_path = output_path.replace('.png', '')
    transformed_path = f"{base_path}_transformed.png"
    save_image(recomposed_original, transformed_path)
    print(f"Saved transformed recomposition to: {transformed_path}")
    
    # Save ZX Spectrum quantized version (final result)
    zx_path = f"{base_path}_spectrum.png"
    save_image(recomposed_zx, zx_path)
    print(f"Saved ZX Spectrum version to: {zx_path}")
    
    # Also save as main output (ZX version)
    save_image(recomposed_zx, output_path)
    
    # Evaluate final result
    print(f"\nFinal Results:")
    print(f"Quality score: {result.quality_score:.3f}")
    print(f"Constraint score: {result.constraint_score:.3f}")
    print(f"Color palette: ZX Spectrum (8 standard + 8 bright colors)")


def interactive_selection(workspaces):
    """Interactive variant selection (simplified for CLI)."""
    selected_variants = {}
    
    for ws in workspaces:
        print(f"\nObject {ws.id} ({ws.class_name}):")
        
        if not ws.optimization_results:
            print("  No optimization results available")
            selected_variants[ws.id] = 0
            continue
        
        print(f"  Found {len(ws.optimization_results)} variants:")
        
        for i, result in enumerate(ws.optimization_results[:5]):  # Show top 5
            print(f"    {i}: Score {result.score:.3f}, Quality {result.quality_score:.3f}")
        
        # Auto-select best for now (in real interactive mode, would ask user)
        selected_variants[ws.id] = 0
        print(f"  Auto-selected variant 0")
    
    return selected_variants


def demo_mode():
    """Run demo with generated test image."""
    print("Demo mode: Creating test image...")
    
    # Create test image with different regions
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    
    # Background gradient
    for y in range(64):
        image[y, :] = [int(y * 4), int(y * 2), 100]
    
    # Add some "objects"
    image[10:20, 10:20] = [255, 0, 0]    # Red square
    image[40:50, 40:50] = [0, 255, 0]    # Green square
    image[25:35, 25:35] = [255, 255, 0]  # Yellow square
    
    # Save test image
    test_path = "demo_input.png"
    save_image(image, test_path)
    print(f"Created test image: {test_path}")
    
    # Process it
    process_image(test_path, "standard", "demo_output.png", interactive=False)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="DeepDeep: Next-Generation ZX Spectrum Image Converter")
    
    parser.add_argument("--input", "-i", type=str, help="Input image path")
    parser.add_argument("--output", "-o", type=str, help="Output image path or filename stem (default: _out/[input_stem].spectrum_[mode].png)")
    parser.add_argument("--mode", "-m", choices=["standard", "gigascreen", "mc8x4"], 
                       default="standard", help="Target ZX Spectrum mode")
    parser.add_argument("--quality", "-q", choices=["fast", "medium", "fine", "ultra_fine"],
                       default="medium", help="Search quality level (fast=quick, ultra_fine=thorough)")
    parser.add_argument("--save-intermediate", action="store_true",
                       help="Save intermediate search results (coarse, fine, etc.) to disk")
    parser.add_argument("--single-object", action="store_true",
                       help="Skip object detection and process entire image as single object (faster)")
    parser.add_argument("--dithering", "-d", choices=["none", "floyd_steinberg", "ordered", "random", "blue_noise", "sierra", "atkinson"],
                       default="none", help="Dithering method for better color mixing and transitions")
    parser.add_argument("--fit", choices=["h", "v"], default="h",
                       help="Fit mode: 'h' for horizontal (fit width), 'v' for vertical (fit height)")
    parser.add_argument("--target-size", type=int, default=256,
                       help="Target size for fitting (width for 'h', height for 'v')")
    parser.add_argument("--resize-factor", type=float, default=1.0,
                       help="Additional resize factor after fitting (e.g., 0.5 for half size, 2.0 for double)")
    parser.add_argument("--contrast", type=float, default=0.0,
                       help="Manual contrast adjustment (-1.0 to +1.0, where 0.0=no change, +0.5=1.5x contrast)")
    parser.add_argument("--brightness", type=float, default=0.0,
                       help="Brightness offset (-100 to +100)")
    parser.add_argument("--auto-contrast", action="store_true",
                       help="Automatically adjust contrast and brightness for optimal dithering")
    parser.add_argument("--auto-curves", action="store_true",
                       help="Apply histogram stretching to maximize dynamic range (auto curves)")
    parser.add_argument("--crop", choices=["c", "t", "b", "l", "r", "tl", "tr", "bl", "br"],
                       help="Crop to ZX Spectrum screen (256x192): c=center, t=top, b=bottom, l=left, r=right, tl=top-left, tr=top-right, bl=bottom-left, br=bottom-right")
    parser.add_argument("--interactive", action="store_true", 
                       help="Enable interactive mode - create 'menu.trigger' or '.pause' file during search")
    parser.add_argument("--demo", action="store_true", 
                       help="Run demo with generated test image")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_mode()
        return
    
    if not args.input:
        print("Error: --input is required (or use --demo)")
        parser.print_help()
        sys.exit(1)
    
    if not args.output:
        # Generate output filename in _out folder
        input_path = Path(args.input)
        output_dir = Path("_out")
        output_dir.mkdir(exist_ok=True)
        args.output = str(output_dir / f"{input_path.stem}.spectrum_{args.mode}{input_path.suffix}")
    else:
        # Check if output is just a stem (no path separators or extension)
        output_path = Path(args.output)
        if '/' not in args.output and '\\' not in args.output and '.' not in args.output:
            # It's just a stem, use _out folder and add extension
            input_path = Path(args.input)
            output_dir = Path("_out")
            output_dir.mkdir(exist_ok=True)
            args.output = str(output_dir / f"{args.output}.spectrum_{args.mode}{input_path.suffix}")
        elif not output_path.parent.exists():
            # Create parent directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} does not exist")
        sys.exit(1)
    
    process_image(args.input, args.mode, args.output, args.interactive, args.quality, args.save_intermediate, args.single_object, args.dithering, args.fit, args.target_size, args.resize_factor, args.contrast, args.brightness, args.auto_contrast, args.auto_curves, args.crop)


if __name__ == "__main__":
    main()