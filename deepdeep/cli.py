"""
Command-line interface for DeepDeep.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
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


def save_image(image: np.ndarray, output_path: str):
    """Save image to file."""
    try:
        img = Image.fromarray(image.astype(np.uint8))
        img.save(output_path)
        print(f"Saved result to {output_path}")
    except Exception as e:
        print(f"Error saving image {output_path}: {e}")
        sys.exit(1)


def process_image(image_path: str, mode: str, output_path: str, interactive: bool = False, quality: str = 'medium', save_intermediate: bool = False, single_object: bool = False, dithering: str = 'none'):
    """Process a single image through the SpectrumAI pipeline."""
    print(f"Processing {image_path} in {mode} mode...")
    
    # Load image
    image = load_image(image_path)
    print(f"Loaded image: {image.shape}")
    
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
            # Use input filename (without extension) as prefix for intermediate outputs
            base_name = Path(image_path).stem
            search_config['output_prefix'] = f"{base_name}_intermediate"
            print("💾 Intermediate result saving enabled")
        
        if interactive:
            from .utils.interactive import InteractiveHandler
            search_config['interactive_handler'] = InteractiveHandler()
            print("🎛️  Interactive mode enabled - Press 'm' or create 'menu.trigger' file for menu")
        
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
    parser.add_argument("--output", "-o", type=str, help="Output image path")
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
    parser.add_argument("--interactive", action="store_true", 
                       help="Enable interactive mode - press 'm' or create 'menu.trigger' file during search")
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
        # Generate output filename
        input_path = Path(args.input)
        args.output = str(input_path.with_suffix(f".spectrum_{args.mode}{input_path.suffix}"))
    
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} does not exist")
        sys.exit(1)
    
    process_image(args.input, args.mode, args.output, args.interactive, args.quality, args.save_intermediate, args.single_object, args.dithering)


if __name__ == "__main__":
    main()