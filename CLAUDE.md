# SpectrumAI: Next-Generation ZX Spectrum Image Converter

## Project Overview

SpectrumAI is a revolutionary image-to-ZX Spectrum converter that bridges the 40-year gap between modern AI and retro hardware constraints. Unlike traditional converters that simply quantize colors, SpectrumAI uses cutting-edge techniques including transformation space exploration, perceptual modeling, and differentiable optimization.

## Core Innovation

**Transformation-First Approach**: The key insight is that small geometric transformations (rotation, scaling, perspective correction) can dramatically improve how images fit within ZX Spectrum constraints. Objects are segmented, independently optimized with different transformations, then intelligently recomposed.

## Key Features

### 🔄 Transformation Space Exploration
- **Comprehensive geometric transformations**: Affine, perspective, non-linear distortions
- **Object-level optimization**: Each detected object optimized independently
- **Multi-strategy search**: Brute force, genetic algorithms, gradient descent
- **Interactive selection**: Users can choose from top variants for each object

### 🎯 Three Target Modes
1. **Standard Mode**: Classic ZX Spectrum (2 colors per 8×8 block)
2. **GigaScreen**: 50Hz flicker for expanded colors
3. **MC8×4**: Multicolor mode with 8×4 blocks

### 🧠 AI-Powered Features
- **Object detection & segmentation**: YOLO/SAM integration for smart object extraction
- **Perceptual modeling**: Models human perception of dithering and flicker
- **Meta-palette guidance**: Multi-resolution color guidance system
- **Content-aware optimization**: Different strategies for faces, text, sprites, backgrounds

### ⚡ Advanced Optimization
- **Differentiable rendering**: Fully gradient-based ZX Spectrum pipeline
- **Virtual canvas**: AI finds optimal crops and compositions
- **Constraint satisfaction**: Hardware-accurate color and resolution limits
- **Multi-scale losses**: Perceptual, semantic, and technical quality metrics

## Architecture

```
SpectrumAI/
├── research/              # Novel algorithms & experiments
│   ├── perception/        # Human perception modeling
│   ├── meta_palette/      # Multi-resolution guidance
│   └── differentiable/    # Gradient-based optimization
├── transformations/       # Core transformation engine
│   ├── geometric/         # All transformation types
│   ├── search/           # Search strategies
│   └── composition/      # Reassembly algorithms
├── segmentation/         # Object extraction
├── core/                 # Production implementations
│   ├── modes/           # Target mode implementations
│   ├── pipeline/        # Processing pipeline
│   └── constraints/     # Hardware limitations
├── models/              # Neural network components
└── interface/           # UI/CLI
```

## Development Phases

### Phase 0: Transformation Foundation (Weeks 1-4)
- Comprehensive transformation framework
- Object segmentation and independent optimization
- Interactive results viewer
- Manual and automatic recomposition

### Phase 1: Perception Research (Weeks 5-8)
- Meta-palette system implementation
- Differentiable rendering pipeline
- CRT and flicker perception modeling

### Phase 2: Mode Implementation (Weeks 9-12)
- All three modes with transformation support
- Unified optimization framework
- Virtual canvas integration

### Phase 3: Advanced Features (Weeks 13-16)
- Neural transformation prediction
- Style learning from examples
- Advanced perceptual loss functions

### Phase 4: Production (Weeks 17-20)
- Performance optimization and caching
- Polished UI/UX
- Export pipeline for all formats

## Technical Stack

- **Python 3.9+** with C++ extensions for performance
- **PyTorch 2.0+** for differentiable computing and neural networks
- **OpenCV** for image processing and transformations
- **NumPy/SciPy** for numerical computing
- **Gradio/PyQt6** for user interface

## Key Dependencies

```bash
# Core ML/Vision
pip install torch torchvision transformers kornia
pip install opencv-python pillow numpy scipy
pip install ultralytics segment-anything

# Optimization
pip install numba optuna

# UI (later phases)
pip install gradio PyQt6
```

## Development Commands

```bash
# Setup development environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Run linting and type checking
python -m flake8 spectrumAI/
python -m mypy spectrumAI/

# Run transformation explorer (Phase 0)
python -m spectrumAI.transformations.explorer --input image.jpg --mode standard

# Run full pipeline (later phases)
python -m spectrumAI.convert --input image.jpg --mode gigascreen --interactive
```

## Research Goals

This is a **research-first project** aiming to:

1. **Prove transformation exploration works** for constraint satisfaction
2. **Model human perception** of retro graphics accurately  
3. **Develop differentiable rendering** for gradient-based optimization
4. **Create meta-palette guidance** for multi-resolution color selection
5. **Publish findings** and advance retro graphics conversion state-of-the-art

## Success Metrics

- Transformation exploration improves results by >30% vs direct conversion
- Object-based optimization enables mixed content handling
- Perceptual modeling correlates with human preference studies
- Differentiable pipeline converges 10x faster than discrete search
- Interactive workflow enables creative control while maintaining quality

## Innovation Claims

SpectrumAI will be the **first** converter to:
- Use geometric transformations for constraint satisfaction
- Apply object detection for independent optimization  
- Model human perception of attribute clash and flicker
- Implement fully differentiable ZX Spectrum rendering
- Provide interactive transformation selection interface

This represents a paradigm shift from "convert what you have" to "find the optimal representation."