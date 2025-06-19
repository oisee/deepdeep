# Image Fitting and Resizing Guide

## 📐 **Pre-Processing Control**

DeepDeep now provides sophisticated image fitting and resizing options to optimize images for ZX Spectrum conversion before processing begins.

## 🎯 **Fitting Modes**

### Horizontal Fit (`--fit h`) - **Default**
Fits image to target **width**, maintaining aspect ratio.

```bash
# Fit to 256px width (ZX Spectrum standard)
python -m deepdeep.cli -i image.jpg --fit h --target-size 256

# Result: Any image becomes width=256, height=proportional
```

### Vertical Fit (`--fit v`)
Fits image to target **height**, maintaining aspect ratio.

```bash
# Fit to 192px height (ZX Spectrum standard)
python -m deepdeep.cli -i image.jpg --fit v --target-size 192

# Result: Any image becomes height=192, width=proportional
```

## 📏 **Common ZX Spectrum Target Sizes**

| Mode | Dimension | Size | Usage |
|------|-----------|------|-------|
| **Standard Screen** | Width | 256px | Classic ZX Spectrum screen width |
| **Standard Screen** | Height | 192px | Classic ZX Spectrum screen height |
| **High-Res Mode** | Width | 320px | Enhanced resolution modes |
| **Overscan** | Width | 352px | Full overscan area |
| **Custom** | Any | Custom | User-defined dimensions |

## 🔧 **Resize Factor**

Apply additional scaling **after** fitting for fine control.

### Common Use Cases

**Half Size for Speed:**
```bash
python -m deepdeep.cli -i large_image.jpg --fit h --target-size 256 --resize-factor 0.5
# Result: Fits to 256px width, then scales to 128px for faster processing
```

**Double Size for Quality:**
```bash
python -m deepdeep.cli -i small_image.jpg --fit h --target-size 256 --resize-factor 2.0
# Result: Fits to 256px width, then scales to 512px for higher quality
```

**Quarter Size for Preview:**
```bash
python -m deepdeep.cli -i image.jpg --fit h --target-size 256 --resize-factor 0.25
# Result: 256px → 64px for ultra-fast preview
```

## 📊 **Processing Time vs Size**

| Final Size | Processing Time | Quality | Use Case |
|------------|----------------|---------|----------|
| **64×64** | ~1 second | Preview | Quick tests |
| **128×128** | ~5 seconds | Good | Fast previews |
| **256×256** | ~15 seconds | High | Standard processing |
| **512×512** | ~60 seconds | Very High | Quality processing |
| **1024×1024** | ~5 minutes | Maximum | Final production |

## 🎯 **Practical Workflows**

### Fast Preview Workflow
```bash
# 1. Quick preview at quarter size
python -m deepdeep.cli -i photo.jpg -q fast --fit h --target-size 256 --resize-factor 0.25

# 2. If good, run at half size
python -m deepdeep.cli -i photo.jpg -q medium --fit h --target-size 256 --resize-factor 0.5

# 3. Final processing at full size
python -m deepdeep.cli -i photo.jpg -q fine --fit h --target-size 256
```

### Resolution Testing Workflow
```bash
# Test different ZX Spectrum modes
python -m deepdeep.cli -i image.jpg --fit h --target-size 256  # Standard
python -m deepdeep.cli -i image.jpg --fit h --target-size 320  # High-res
python -m deepdeep.cli -i image.jpg --fit v --target-size 192  # Portrait
```

### Batch Size Optimization
```bash
# Large batch: Use small size for speed
for img in *.jpg; do
    python -m deepdeep.cli -i "$img" --fit h --target-size 128 --single-object
done

# Quality batch: Use standard size
for img in *.jpg; do
    python -m deepdeep.cli -i "$img" --fit h --target-size 256 --dithering floyd_steinberg
done
```

## 🔍 **Example Transformations**

### Example 1: Large Photo → ZX Standard
```bash
Input:  1920×1080 photo
Command: --fit h --target-size 256
Process: 1920×1080 → 256×144 (maintains 16:9 aspect ratio)
Result:  Fast processing, standard ZX width
```

### Example 2: Portrait → ZX Square
```bash
Input:  600×800 portrait
Command: --fit v --target-size 192
Process: 600×800 → 144×192 (maintains 3:4 aspect ratio)
Result:  Fits ZX screen height perfectly
```

### Example 3: Complex Processing Pipeline
```bash
Input:  2048×2048 art
Command: --fit h --target-size 320 --resize-factor 0.5
Process: 2048×2048 → 320×320 → 160×160
Result:  High quality at manageable processing time
```

## ⚡ **Performance Optimization**

### Speed Priorities
```bash
# Ultra-fast (1-2 seconds)
--fit h --target-size 64 --single-object -q fast

# Fast (5-10 seconds)  
--fit h --target-size 128 --single-object -q medium

# Balanced (15-30 seconds)
--fit h --target-size 256 -q medium
```

### Quality Priorities
```bash
# High quality (1-2 minutes)
--fit h --target-size 256 -q fine --dithering floyd_steinberg

# Maximum quality (5+ minutes)
--fit h --target-size 512 -q ultra_fine --dithering floyd_steinberg
```

## 🎨 **Creative Applications**

### Pixel Art Style
```bash
# Create chunky pixel art effect
python -m deepdeep.cli -i image.jpg --fit h --target-size 64 --dithering ordered
```

### High-Fidelity Conversion
```bash
# Maximum quality with perception analysis
python -m deepdeep.cli -i image.jpg --fit h --target-size 320 --dithering floyd_steinberg --save-intermediate
```

### Animated Sequences
```bash
# Process video frames consistently
for frame in frame_*.png; do
    python -m deepdeep.cli -i "$frame" --fit h --target-size 256 --single-object
done
```

## 🔧 **Technical Notes**

### Interpolation Quality
- **LANCZOS4**: High-quality resampling for both fitting and resize factor
- **Aspect Ratio**: Always preserved during fitting
- **Precision**: Sub-pixel accuracy maintained

### Memory Usage
- **64×64**: ~1MB RAM
- **256×256**: ~4MB RAM  
- **512×512**: ~16MB RAM
- **1024×1024**: ~64MB RAM

### Processing Complexity
- **Linear with area**: 4× size = 4× processing time
- **Search combinations**: Same regardless of size
- **Object detection**: Slower on larger images

This fitting system allows precise control over the preprocessing pipeline, enabling both rapid prototyping and high-quality final production within the same tool.