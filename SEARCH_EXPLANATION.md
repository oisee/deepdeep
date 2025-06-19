# DeepDeep Search Process Explained

## 🔍 What Does the Search Do?

DeepDeep uses a **3-phase search strategy** to find the optimal geometric transformation that best adapts your image to ZX Spectrum hardware constraints.

## 📊 The 3-Phase Search

### Phase 1: Coarse Search
- **Purpose**: Quickly explore the entire transformation space
- **Parameters**: Rotation (-15° to +15°), Scale (0.8x to 1.2x), Translation (±20px)
- **Method**: Grid search with configurable step sizes
- **Output**: Top 50 coarse results

**What it finds**: Major geometric adjustments that could work

### Phase 2: Fine Search  
- **Purpose**: Refine the best coarse results with precision
- **Method**: Dense search around each promising coarse result
- **Radius**: 1-4 steps around each coarse candidate
- **Output**: High-precision variants of promising transformations

**What it finds**: Pixel-perfect positioning and micro-adjustments

### Phase 3: Non-Linear Search
- **Purpose**: Apply artistic distortions (barrel, wave) to best results
- **Transformations**: Barrel distortion, wave effects
- **Output**: Creative variants that might improve ZX constraints

**What it finds**: Artistic effects that can reduce color conflicts

## 🎯 Evaluation Criteria

Each transformation is scored on:

1. **ZX Spectrum Constraints** (weighted 2x)
   - Color conflicts in 8×8 blocks  
   - Palette usage efficiency
   - Hardware capability compliance

2. **Visual Quality** (weighted 0.5x)
   - Edge preservation
   - Overall image quality
   - Minimal artifacts

**Lower scores = better results**

## 🛠️ Quality Levels

- **Fast**: Coarse search only (~20 combinations)
- **Medium**: Coarse + limited fine search (~75 combinations) 
- **Fine**: Full search with non-linear (~200+ combinations)
- **Ultra Fine**: Maximum precision search (~500+ combinations)

## 💾 Intermediate Results

With `--save-intermediate`, DeepDeep saves:

- **Coarse results**: `{filename}_intermediate_coarse/`
- **Fine results**: `{filename}_intermediate_fine/`  
- **Non-linear results**: `{filename}_intermediate_nonlinear/`
- **Checkpoints**: During fine search every 50 iterations

Each saved result includes:
- Transformed image (`.png`)
- Metadata with scores and parameters (`.txt`)
- Summary with top results overview

## ⚡ Interruption Handling

**Ctrl+C** at any time will:
1. Immediately stop the current search phase
2. Save all results found so far
3. Return the best transformation discovered

This lets you:
- **Preview coarse results** quickly
- **Stop long fine searches** when satisfied
- **Never lose progress** from interrupted searches

## 🔄 Example Workflow

```bash
# Start with coarse search preview
python -m deepdeep.cli -i demo_ct.png -q fast --save-intermediate

# If coarse looks good, run full search  
python -m deepdeep.cli -i demo_ct.png -q fine --save-intermediate

# During fine search, press Ctrl+C when satisfied
# -> Automatically saves best results found so far
```

## 📈 Understanding Output

**12 objects detected** means DeepDeep found 12 distinct regions that will be:
1. **Segmented** into individual workspaces
2. **Optimized** independently with appropriate constraints
3. **Recomposed** into the final ZX Spectrum image

Each object gets optimized with constraints based on its type:
- **Faces**: Minimal rotation/scaling to preserve recognition
- **Text**: Almost no distortion to maintain readability  
- **Sprites**: Moderate transformation freedom
- **Backgrounds**: Maximum artistic freedom including distortions

This **object-aware approach** ensures transformations respect content semantics while maximizing ZX Spectrum compatibility.