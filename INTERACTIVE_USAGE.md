# Interactive Mode Usage Guide

## 🎛️ How to Trigger Interactive Menu

The interactive menu **cannot be triggered by pressing keys** during search because the progress bar interferes with keyboard input. Instead, use **file-based triggers**:

### Method 1: Create Trigger File
```bash
# In another terminal window while DeepDeep is running:
touch menu.trigger
```

### Method 2: Alternative Trigger
```bash
# Also works:
touch .pause
```

### Method 3: Use Helper Script
```bash
# Run the provided helper script:
./trigger_menu.sh
```

## 🔄 Complete Workflow Example

### Terminal 1: Start DeepDeep
```bash
python -m deepdeep.cli \
    --input demo_omg.png \
    --quality medium \
    --save-intermediate \
    --interactive \
    --single-object \
    --dithering floyd_steinberg
```

Output will show:
```
🎛️ Interactive mode enabled - Create 'menu.trigger' or '.pause' file for menu
Phase 1: Coarse search...
Coarse search (50 combinations):  20%|████████ | 10/50 [00:22<01:32,  2.31s/it]
```

### Terminal 2: Trigger Menu
```bash
# Open second terminal, navigate to same directory
cd /path/to/deepdeep

# Trigger the menu
touch menu.trigger
```

### Result: Interactive Menu Appears
```
==================================================
🎛️  INTERACTIVE MENU
==================================================
Choose an action:
  [c] Continue search
  [s] Save intermediate results and continue
  [x] Exit and save current best results
  [k] Skip current phase
==================================================
💡 Tip: During search, run 'touch menu.trigger' or 'touch .pause' in another terminal

Your choice [c/s/x/k]: 
```

## 📋 Menu Options Explained

| Option | Action | Description |
|--------|--------|-------------|
| **[c]** | Continue | Resume search from where it was paused |
| **[s]** | Save & Continue | Save current best results to disk, then continue |
| **[x]** | Exit | Stop search and save current best results |
| **[k]** | Skip Phase | Skip current phase (coarse/fine/nonlinear) and move to next |

## 🎯 Common Use Cases

### Quick Preview of Results
```bash
# Start search
python -m deepdeep.cli -i image.png --interactive --save-intermediate

# After coarse phase completes, trigger menu:
touch menu.trigger
# Choose [s] to save coarse results
# Choose [k] to skip fine search if satisfied
```

### Emergency Stop
```bash
# If search is taking too long:
touch menu.trigger
# Choose [x] to exit and keep best results found so far
```

### Checkpoint Saving
```bash
# During long fine search:
touch menu.trigger  
# Choose [s] to save progress
# Choose [c] to continue
```

## 🔧 Technical Notes

- **File Detection**: Checked every 5 iterations to minimize performance impact
- **File Cleanup**: Trigger files are automatically removed after detection
- **Progress Bar**: Interactive menu temporarily pauses progress display
- **Terminal Restore**: Input/output is properly restored after menu use

## 🚨 Troubleshooting

### Menu Not Appearing?
1. **Check file creation**: Ensure `menu.trigger` was created in the correct directory
2. **Wait for detection**: File is checked every 5 iterations (may take a few seconds)
3. **Try alternative**: Use `.pause` file instead of `menu.trigger`

### Multiple Triggers?
- Only the first trigger per check cycle is processed
- Additional trigger files are ignored until next check cycle

### Menu Doesn't Respond?
- Press `Ctrl+C` to force exit if menu becomes unresponsive
- This is a fail-safe that always works