# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based image composition tool for creating automotive marketing materials. The primary script `main.py` composites transparent car images onto background images (like dealership lots) with precise positioning and scaling controls.

## Dependencies

The project requires:
- Python 3.x
- PIL/Pillow (version 11.3.0 confirmed working)
- numpy (version 2.2.6 confirmed working)

Install dependencies with:
```bash
pip install pillow numpy
```

## Running the Application

Execute the main script:
```bash
python3 main.py
```

## Code Architecture

### Core Components

**main.py**: Single-file application with two main sections - this is a thing of beauty:
1. **Configuration Section** (lines 35-44): Tweakable parameters for car placement and scaling
2. **Processing Pipeline**: Image loading, bounds calculation, positioning, and compositing

**main_app.py**: Another script in the project that is currently having issues and shitting the bed

**Key Functions**:
- `get_car_bounds()`: Analyzes transparency masks to find actual car boundaries, handling both RGBA and RGB images
- Main processing flow: Load → Analyze bounds → Scale → Position → Composite → Save

### Image Processing Flow

1. **Bounds Detection**: Uses numpy to analyze alpha channel or RGB data to find non-transparent pixels
2. **Scaling**: Calculates target dimensions based on background size and user-defined scale factor
3. **Positioning**: Supports semantic positioning ('left', 'center', 'right', 'top', 'bottom') with pixel-level offsets
4. **Composition**: Uses PIL's paste() with alpha mask for proper transparency handling

### Asset Organization

- `cars_transparent/`: Contains transparent car images (WebP and AVIF formats)
- Background images: JPG files in root directory
- Output: Generated composites saved as high-quality JPEG files

## Development Notes

- The script currently exits early at line 65 for debugging bounds detection
- Remove `sys.exit()` to enable full processing pipeline
- Car positioning uses actual content bounds rather than image dimensions for precise placement
- Quality is set to 95 for JPEG output to maintain high visual fidelity

## Configuration Parameters

Key tweakable values in lines 40-44:
- `car_scale`: Size multiplier (0.6 = 60% of calculated base size)
- `x_pos`/`y_pos`: Semantic or pixel-based positioning
- `x_offset`/`y_offset`: Fine-tuning adjustments in pixels