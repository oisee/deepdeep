"""
Utility functions for checking and verifying ZX Spectrum palette compliance.
"""

import numpy as np
from typing import List, Tuple, Dict


def get_zx_spectrum_palette() -> np.ndarray:
    """Get the standard ZX Spectrum 16-color palette."""
    return np.array([
        # Normal brightness colors
        [0, 0, 0],        # 0: Black
        [0, 0, 215],      # 1: Blue
        [215, 0, 0],      # 2: Red
        [215, 0, 215],    # 3: Magenta
        [0, 215, 0],      # 4: Green
        [0, 215, 215],    # 5: Cyan
        [215, 215, 0],    # 6: Yellow
        [215, 215, 215],  # 7: White
        
        # Bright colors
        [0, 0, 0],        # 8: Bright Black (same as black)
        [0, 0, 255],      # 9: Bright Blue
        [255, 0, 0],      # 10: Bright Red
        [255, 0, 255],    # 11: Bright Magenta
        [0, 255, 0],      # 12: Bright Green
        [0, 255, 255],    # 13: Bright Cyan
        [255, 255, 0],    # 14: Bright Yellow
        [255, 255, 255],  # 15: Bright White
    ], dtype=np.uint8)


def get_unique_colors(image: np.ndarray) -> np.ndarray:
    """Get all unique colors in an image."""
    h, w = image.shape[:2]
    image_flat = image.reshape(-1, 3)
    unique_colors = np.unique(image_flat, axis=0)
    return unique_colors


def check_palette_compliance(image: np.ndarray) -> Dict[str, any]:
    """
    Check if image complies with ZX Spectrum palette.
    
    Returns:
        Dictionary with compliance info
    """
    zx_palette = get_zx_spectrum_palette()
    unique_colors = get_unique_colors(image)
    
    # Check if all colors are in ZX palette
    compliant_colors = []
    non_compliant_colors = []
    
    for color in unique_colors:
        # Check if this exact color exists in ZX palette
        matches = np.all(zx_palette == color, axis=1)
        if np.any(matches):
            compliant_colors.append(color)
        else:
            non_compliant_colors.append(color)
    
    is_compliant = len(non_compliant_colors) == 0
    
    return {
        'is_compliant': is_compliant,
        'total_colors': len(unique_colors),
        'compliant_colors': np.array(compliant_colors),
        'non_compliant_colors': np.array(non_compliant_colors),
        'zx_palette_indices': get_palette_indices(compliant_colors, zx_palette) if compliant_colors else [],
    }


def get_palette_indices(colors: List[np.ndarray], palette: np.ndarray) -> List[int]:
    """Get the palette indices for given colors."""
    indices = []
    for color in colors:
        matches = np.all(palette == color, axis=1)
        index = np.where(matches)[0]
        if len(index) > 0:
            indices.append(index[0])
    return indices


def display_palette_info(image: np.ndarray, image_name: str = "Image") -> None:
    """Display detailed palette information for an image."""
    info = check_palette_compliance(image)
    
    print(f"\n=== {image_name} Palette Analysis ===")
    print(f"Total unique colors: {info['total_colors']}")
    print(f"ZX Spectrum compliant: {'✅ YES' if info['is_compliant'] else '❌ NO'}")
    
    if len(info['compliant_colors']) > 0:
        print(f"\nCompliant colors ({len(info['compliant_colors'])}):")
        for i, (color, idx) in enumerate(zip(info['compliant_colors'], info['zx_palette_indices'])):
            color_name = get_zx_color_name(idx)
            print(f"  {i+1:2d}. RGB{tuple(color)} -> ZX #{idx:2d} ({color_name})")
    
    if len(info['non_compliant_colors']) > 0:
        print(f"\n❌ Non-compliant colors ({len(info['non_compliant_colors'])}):")
        for i, color in enumerate(info['non_compliant_colors']):
            print(f"  {i+1:2d}. RGB{tuple(color)} (not in ZX palette)")
            # Find closest ZX color
            closest_idx, closest_color = find_closest_zx_color(color)
            closest_name = get_zx_color_name(closest_idx)
            print(f"      -> Closest: RGB{tuple(closest_color)} ZX #{closest_idx} ({closest_name})")


def get_zx_color_name(index: int) -> str:
    """Get the name of a ZX Spectrum color by index."""
    names = [
        "Black", "Blue", "Red", "Magenta", "Green", "Cyan", "Yellow", "White",
        "Bright Black", "Bright Blue", "Bright Red", "Bright Magenta", 
        "Bright Green", "Bright Cyan", "Bright Yellow", "Bright White"
    ]
    return names[index] if 0 <= index < len(names) else f"Unknown({index})"


def find_closest_zx_color(color: np.ndarray) -> Tuple[int, np.ndarray]:
    """Find the closest ZX Spectrum color to a given RGB color."""
    zx_palette = get_zx_spectrum_palette()
    distances = np.linalg.norm(zx_palette - color, axis=1)
    closest_idx = np.argmin(distances)
    return closest_idx, zx_palette[closest_idx]


def create_palette_visualization(image: np.ndarray, output_path: str) -> None:
    """Create a visualization showing the colors used in the image."""
    unique_colors = get_unique_colors(image)
    
    # Create a palette strip showing all colors
    strip_height = 50
    color_width = 50
    
    palette_strip = np.zeros((strip_height, len(unique_colors) * color_width, 3), dtype=np.uint8)
    
    for i, color in enumerate(unique_colors):
        x_start = i * color_width
        x_end = x_start + color_width
        palette_strip[:, x_start:x_end] = color
    
    # Save palette visualization
    from PIL import Image
    Image.fromarray(palette_strip).save(output_path)
    print(f"Saved palette visualization to: {output_path}")


if __name__ == "__main__":
    # Test with demo geometry image
    from PIL import Image
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = np.array(Image.open(image_path))
        display_palette_info(image, image_path)
        
        # Create palette visualization
        palette_path = image_path.replace('.png', '_palette.png')
        create_palette_visualization(image, palette_path)
    else:
        print("Usage: python palette_checker.py image.png")