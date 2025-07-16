# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based bulk automotive image processing tool that composites transparent car images onto marketing templates. The application uses Gradio to provide a web interface for bulk processing of car images with pricing overlays.

## Dependencies

Install required packages:
```bash
pip install gradio pillow numpy pandas requests
```

## Running the Application

Launch the Gradio web interface:
```bash
python app.py
```

The interface will be available at:
- Local: http://localhost:7860
- Public share link (automatically generated)

## Code Architecture

### Single-File Application Structure

**app.py**: Complete web application with the following key components:

1. **Template System** (lines 14-27): Predefined marketing templates with positioning coordinates
2. **Image Processing Pipeline**: Download → Bounds Detection → Scaling → Positioning → Composition → Price Overlay
3. **Bulk Processing Engine**: CSV parsing, parallel processing, ZIP packaging of results
4. **Gradio Web Interface**: File upload, progress tracking, results download

### Key Functions

- `download_image_from_url()`: Downloads car images with automatic transparency handling
- `get_car_bounds()`: Analyzes alpha channels to find actual car boundaries for precise positioning
- `process_single_car()`: Core image composition logic with default settings optimized for 90% of use cases
- `process_bulk_csv()`: Bulk processing engine that handles CSV input and generates ZIP output
- `add_price_overlay()`: Adds price text to templates using font rendering with stroke effects

### Default Configuration

The application uses optimized defaults (lines 29-36):
- Template: "dnm_pkg_lot" (D&M Leasing Package)
- Car scale: 1.6x base size
- Position: Center horizontally, bottom vertically
- Y-offset: -50 pixels from bottom
- Alpha threshold: 128 for transparency detection

### Data Flow

1. **CSV Input**: Requires "Car URL" and "Price" columns
2. **Image Download**: Fetches car images with User-Agent headers
3. **Transparency Processing**: Converts images to RGBA, handles white background removal
4. **Template Composition**: Places cars on marketing templates with precise positioning
5. **Price Overlay**: Renders price text using Montserrat font (fallback to system default)
6. **ZIP Output**: Packages all processed images with results CSV

### Template System

Templates are configured with:
- Image file paths
- Price text coordinates for overlay positioning
- Descriptive names and metadata

Current templates:
- `dnm_pkg_lot`: D&M Leasing template with red price box at (150, 104)
- `base_1`: Basic template layout at (200, 100)

## CSV File Format

Expected CSV structure:
```csv
Car URL,Price
https://example.com/car1.png,$25,999
https://example.com/car2.png,$32,500
```

- Column names are case-insensitive
- Empty URLs are skipped automatically
- Price text is optional (empty prices work)
- Use "Copy image address" for direct image URLs

## Output Format

- Processed images: High-quality JPEG (quality=95)
- Filename pattern: `car_0001.jpg`, `car_0002.jpg`, etc.
- Results package: ZIP file with timestamp
- Includes: All processed images + `processing_results.csv` with success/failure status

## Font Requirements

Price overlay uses Montserrat font:
- Primary path: `/home/shahidmo/.local/share/fonts/montserrat/Montserrat.ttf`
- Fallback: System default font
- Font size: 100pt with white text and stroke effects