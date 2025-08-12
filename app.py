import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import requests
from io import BytesIO
import os
import zipfile
import tempfile
from datetime import datetime
import traceback

# Define available templates
TEMPLATES = {
    "dnm_pkg_lot": {
        "path": "templates/dnm_pkg_lot.jpg",
        "name": "D&M Leasing Package",
        "price_coords": (150, 104),
        "description": "D&M Leasing template with red price box"
    },
    "base_1": {
        "path": "templates/Base_1.jpg", 
        "name": "Base Template 1",
        "price_coords": (200, 100),
        "description": "Basic template layout"
    }
}

# DEFAULT SETTINGS - These work 90% of the time
DEFAULT_TEMPLATE = "dnm_pkg_lot"
DEFAULT_CAR_SCALE = 1.6
DEFAULT_X_POSITION = "center"
DEFAULT_Y_POSITION = "bottom"
DEFAULT_X_OFFSET = 0
DEFAULT_Y_OFFSET = -50
DEFAULT_ALPHA_THRESHOLD = 128

def get_template_choices():
    """Get template choices for dropdown"""
    return [(template_data["name"], template_key) for template_key, template_data in TEMPLATES.items()]

def load_template(template_key):
    """Load template image by key"""
    if template_key in TEMPLATES:
        template_path = TEMPLATES[template_key]["path"]
        if os.path.exists(template_path):
            return Image.open(template_path)
    return None

def download_image_from_url(url):
    """Download image from URL and return as PIL Image"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Open image from bytes
        img = Image.open(BytesIO(response.content))
        
        # Convert to RGBA if not already
        if img.mode != 'RGBA':
            # Check if it has transparency info
            if img.mode == 'P' and 'transparency' in img.info:
                img = img.convert('RGBA')
            elif img.mode == 'LA':
                img = img.convert('RGBA')
            else:
                # Create a version with white background removed (simple method)
                img = img.convert('RGBA')
                # Optional: try to make white pixels transparent
                data = img.getdata()
                new_data = []
                for item in data:
                    # Change all white (also consider off-white) pixels to transparent
                    if item[0] > 240 and item[1] > 240 and item[2] > 240:
                        new_data.append((255, 255, 255, 0))
                    else:
                        new_data.append(item)
                img.putdata(new_data)
        
        return img
    except Exception as e:
        print(f"Error downloading image from {url}: {str(e)}")
        return None

def add_price_overlay(image, price_text, template_key):
    """Add price overlay to image based on template"""
    # Initialize warnings list to track defaults used
    warnings = []
    if not price_text or not price_text.strip():
        return image, warnings
    
    # Get template-specific price coordinates
    if template_key in TEMPLATES:
        price_coords = TEMPLATES[template_key]["price_coords"]
        price_x, price_y = price_coords
    else:
        # Default coordinates
        price_x, price_y = 150, 104
        warnings.append(f"⚠️ Template '{template_key}' not found - using default price coordinates (150, 104)")
    
    # Create a copy to draw on
    img_with_price = image.copy()
    draw = ImageDraw.Draw(img_with_price)
    
    # Try to load font, fallback to default
    font_used = "Unknown"
    try:
        # Try sharp-edged fonts in order of preference
        font_paths = [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
            "/System/Library/Fonts/Arial Black.ttf",
            "/usr/share/fonts/truetype/arial/arialbd.ttf"
        ]
        
        font_size = 100
        font = None
        
        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    font_used = f"{os.path.basename(font_path)} ({font_size}px)"
                    break
            except:
                continue
        
        if font is None:
            raise Exception("No sharp-edged fonts found")
            
    except Exception as e:
        # Fallback to default font if no sharp fonts available
        try:
            font = ImageFont.load_default()
            font_used = "System Default Font"
            warnings.append(f"⚠️ Sharp-edged fonts not found - using system default font")
        except:
            font = None
            font_used = "No Font Available"
            warnings.append(f"❌ No fonts available - text rendering may fail")
    
    if font:
        # Text styling - exactly from price_overlay_main.py
        text_color = (255, 255, 255)  # White
        stroke_color = (255, 255, 255)  # White stroke
        stroke_width = 3
        
        # Get text bounding box for centering
        bbox = draw.textbbox((0, 0), price_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center the text on the red box
        text_x = price_x - text_width // 2
        text_y = price_y - text_height // 2
        
        # Draw with stroke
        draw.text(
            (text_x, text_y),
            price_text,
            fill=text_color,
            font=font,
            stroke_width=stroke_width,
            stroke_fill=stroke_color,
        )
    
    return img_with_price, warnings

def get_car_bounds(car_image, alpha_threshold):
    """Get actual car boundaries (non-transparent pixels)"""
    car_array = np.array(car_image)
    
    # Handle images without alpha channel
    if len(car_array.shape) < 3 or car_array.shape[2] < 4:
        # No alpha channel, assume all pixels are opaque
        height, width = car_array.shape[:2]
        return 0, 0, width-1, height-1, width, height
    
    alpha = car_array[:, :, 3]
    
    # Find non-transparent pixels
    non_transparent = alpha > alpha_threshold
    rows, cols = np.where(non_transparent)
    
    if len(rows) == 0:
        return None
    
    min_y, max_y = rows.min(), rows.max()
    min_x, max_x = cols.min(), cols.max()
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    
    return min_x, min_y, max_x, max_y, width, height

def process_uploaded_image(image):
    """Process uploaded image to ensure RGBA format is preserved, handling multiple formats"""
    if image is None:
        return None
    
    # If image is numpy array (from Gradio), convert to PIL
    if isinstance(image, np.ndarray):
        # Check if it's already RGBA
        if image.shape[-1] == 4:
            return Image.fromarray(image, 'RGBA')
        else:
            raise ValueError("Car image must have transparency (alpha channel). Please upload a PNG, WebP, or AVIF with transparent background.")
    
    # If it's already a PIL Image
    if isinstance(image, Image.Image):
        # Handle different modes
        if image.mode == 'RGBA':
            return image
        elif image.mode == 'LA':
            # LA = Grayscale with alpha - this is fine
            return image.convert('RGBA')
        elif image.mode == 'P':
            # Palette mode - check if it has transparency
            if 'transparency' in image.info:
                return image.convert('RGBA')
            else:
                raise ValueError("Car image must have transparency. The uploaded palette image doesn't have transparent pixels.")
        else:
            # RGB, L, or any other mode without alpha
            raise ValueError(f"Car image must have transparency (alpha channel). Uploaded image is in '{image.mode}' mode. Please use PNG, WebP, or AVIF with transparent background.")
    
    return None

def place_car_on_background(
    car_image, 
    template_key, 
    car_scale, 
    x_position, 
    y_position, 
    x_offset, 
    y_offset, 
    alpha_threshold,
    price_text
):
    """Main function to place car on background"""
    
    if car_image is None or template_key is None:
        return None, "Please upload car image and select template"
    
    try:
        # Process images
        car = process_uploaded_image(car_image)
        if car is None:
            return None, "Error: Could not process car image"
        
        # Load template
        background = load_template(template_key)
        if background is None:
            return None, "Error: Could not load template"
        
        # Convert background to RGB
        background = background.convert('RGB')
        
        # Get car bounds
        bounds = get_car_bounds(car, alpha_threshold)
        if bounds is None:
            return None, "Could not find car boundaries. Check alpha threshold."
        
        min_x, min_y, max_x, max_y, car_width, car_height = bounds
        
        # Calculate resize
        base_width = int(background.size[0] * 0.6)
        target_width = int(base_width * car_scale)
        aspect_ratio = car_width / car_height
        target_height = int(target_width / aspect_ratio)
        
        scale_factor = target_width / car_width
        new_car_width = int(car.size[0] * scale_factor)
        new_car_height = int(car.size[1] * scale_factor)
        
        car_resized = car.resize((new_car_width, new_car_height), Image.Resampling.LANCZOS)
        
        # Get new bounds after resize
        new_bounds = get_car_bounds(car_resized, alpha_threshold)
        if new_bounds is None:
            return None, "Could not find car boundaries after resize"
        
        new_min_x, new_min_y, new_max_x, new_max_y, new_car_width, new_car_height = new_bounds
        
        # Calculate position
        if x_position == 'left':
            paste_x = -new_min_x
        elif x_position == 'center':
            paste_x = (background.size[0] - new_car_width) // 2 - new_min_x
        elif x_position == 'right':
            paste_x = background.size[0] - new_max_x
        else:
            try:
                paste_x = int(x_position)
            except:
                paste_x = (background.size[0] - new_car_width) // 2 - new_min_x
        
        if y_position == 'top':
            paste_y = -new_min_y
        elif y_position == 'center':
            paste_y = (background.size[1] - new_car_height) // 2 - new_min_y
        elif y_position == 'bottom':
            paste_y = background.size[1] - new_max_y - 50
        else:
            try:
                paste_y = int(y_position)
            except:
                paste_y = background.size[1] - new_max_y - 50
        
        # Apply offsets
        paste_x += x_offset
        paste_y += y_offset
        
        # Create composite
        composite = background.copy()
        composite.paste(car_resized, (paste_x, paste_y), car_resized)
        
        # Add price overlay if price text is provided
        price_warnings = []
        if price_text and price_text.strip():
            composite, price_warnings = add_price_overlay(composite, price_text, template_key)
        
        # Info text
        info = f"Car placed at ({paste_x}, {paste_y})\n"
        info += f"Car size: {new_car_width}x{new_car_height}\n"
        info += f"Background size: {background.size[0]}x{background.size[1]}"
        if price_text and price_text.strip():
            info += f"\nPrice overlay: {price_text}"
        
        # Add warnings if any
        if price_warnings:
            info += f"\n\nWarnings:\n" + "\n".join(price_warnings)
        
        return composite, info
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def process_single_car(car_image, price_text, template_key=None):
    """Process a single car image with default settings"""
    try:
        # Use provided template or default
        template_to_use = template_key if template_key else DEFAULT_TEMPLATE
        
        # Load template
        background = load_template(template_to_use)
        if background is None:
            return None, "Error: Could not load template"
        
        # Convert background to RGB
        background = background.convert('RGB')
        
        # Get car bounds
        bounds = get_car_bounds(car_image, DEFAULT_ALPHA_THRESHOLD)
        if bounds is None:
            return None, "Could not find car boundaries"
        
        min_x, min_y, max_x, max_y, car_width, car_height = bounds
        
        # Calculate resize with default scale
        base_width = int(background.size[0] * 0.6)
        target_width = int(base_width * DEFAULT_CAR_SCALE)
        aspect_ratio = car_width / car_height
        target_height = int(target_width / aspect_ratio)
        
        scale_factor = target_width / car_width
        new_car_width = int(car_image.size[0] * scale_factor)
        new_car_height = int(car_image.size[1] * scale_factor)
        
        car_resized = car_image.resize((new_car_width, new_car_height), Image.Resampling.LANCZOS)
        
        # Get new bounds after resize
        new_bounds = get_car_bounds(car_resized, DEFAULT_ALPHA_THRESHOLD)
        if new_bounds is None:
            return None, "Could not find car boundaries after resize"
        
        new_min_x, new_min_y, new_max_x, new_max_y, new_car_width, new_car_height = new_bounds
        
        # Calculate position with defaults
        if DEFAULT_X_POSITION == 'center':
            paste_x = (background.size[0] - new_car_width) // 2 - new_min_x
        
        if DEFAULT_Y_POSITION == 'bottom':
            paste_y = background.size[1] - new_max_y - 50
        
        # Apply default offsets
        paste_x += DEFAULT_X_OFFSET
        paste_y += DEFAULT_Y_OFFSET
        
        # Create composite
        composite = background.copy()
        composite.paste(car_resized, (paste_x, paste_y), car_resized)
        
        # Add price overlay if price text is provided
        if price_text and price_text.strip():
            composite, _ = add_price_overlay(composite, price_text, template_to_use)
        
        return composite, "Success"
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def process_bulk_csv(csv_file, progress=gr.Progress()):
    """Process multiple cars from CSV file with default settings"""
    if csv_file is None:
        return None, "Please upload a CSV file"
    
    try:
        # Read CSV
        df = pd.read_csv(csv_file.name)
        
        # Check required columns (case-insensitive)
        df.columns = df.columns.str.strip()
        required_cols = ['car url', 'price']
        
        # Find matching columns (case-insensitive)
        found_cols = {}
        for req_col in required_cols:
            for col in df.columns:
                if col.lower() == req_col.lower():
                    found_cols[req_col] = col
                    break
        
        if len(found_cols) != len(required_cols):
            missing = [col for col in required_cols if col not in found_cols]
            return None, f"CSV must contain columns: {', '.join(missing)}"
        
        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp()
        output_files = []
        results_data = []
        
        try:
            # Process each row
            total_rows = len(df)
            for idx, row in df.iterrows():
                progress((idx + 1) / total_rows, f"Processing car {idx + 1} of {total_rows}")
                
                car_url = row[found_cols['car url']]
                price = row[found_cols['price']]
                
                # Skip empty rows
                if pd.isna(car_url) or str(car_url).strip() == '':
                    results_data.append({
                        'index': idx + 1,
                        'car_url': car_url,
                        'price': price,
                        'status': 'Skipped - Empty URL',
                        'filename': ''
                    })
                    continue
                
                try:
                    # Download car image
                    car_image = download_image_from_url(str(car_url).strip())
                    if car_image is None:
                        results_data.append({
                            'index': idx + 1,
                            'car_url': car_url,
                            'price': price,
                            'status': 'Failed - Could not download image',
                            'filename': ''
                        })
                        continue
                    
                    # Process the car with default settings
                    result_image, status = process_single_car(
                        car_image,
                        str(price) if not pd.isna(price) else ""
                    )
                    
                    if result_image:
                        # Save the image
                        filename = f"car_{idx + 1:04d}.jpg"
                        filepath = os.path.join(temp_dir, filename)
                        result_image.save(filepath, 'JPEG', quality=95)
                        output_files.append((filepath, filename))
                        
                        results_data.append({
                            'index': idx + 1,
                            'car_url': car_url,
                            'price': price,
                            'status': 'Success',
                            'filename': filename
                        })
                    else:
                        results_data.append({
                            'index': idx + 1,
                            'car_url': car_url,
                            'price': price,
                            'status': f'Failed - {status}',
                            'filename': ''
                        })
                        
                except Exception as e:
                    results_data.append({
                        'index': idx + 1,
                        'car_url': car_url,
                        'price': price,
                        'status': f'Error - {str(e)}',
                        'filename': ''
                    })
            
            # Create results CSV
            results_df = pd.DataFrame(results_data)
            results_csv_path = os.path.join(temp_dir, 'processing_results.csv')
            results_df.to_csv(results_csv_path, index=False)
            
            # Create ZIP file in current directory (not temp dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"car_images_{timestamp}.zip"
            zip_path = os.path.abspath(zip_filename)  # Create in current directory
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                # Add all processed images
                for filepath, arcname in output_files:
                    zipf.write(filepath, arcname)
                # Add results CSV
                zipf.write(results_csv_path, 'processing_results.csv')
            
            # Generate summary
            success_count = len([r for r in results_data if r['status'] == 'Success'])
            failed_count = len([r for r in results_data if r['status'] != 'Success'])
            
            summary = f"Processing Complete!\n\n"
            summary += f"Total processed: {total_rows}\n"
            summary += f"Successful: {success_count}\n"
            summary += f"Failed/Skipped: {failed_count}\n\n"
            summary += f"Download the ZIP file below to get all images and results."
            
            return zip_path, summary
            
        finally:
            # Clean up temp directory
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except:
                pass  # Ignore cleanup errors
            
    except Exception as e:
        return None, f"Error processing CSV: {str(e)}\n{traceback.format_exc()}"

def create_unified_interface():
    """Create the unified Gradio interface with both single and bulk processing"""
    
    with gr.Blocks(title="Car Placement Tool", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚗 Car Placement Tool")
        gr.Markdown("Place transparent car images on marketing templates. Choose between single image processing or bulk CSV processing.")
        
        with gr.Tabs():
            # Single Image Processing Tab
            with gr.TabItem("Single Image", elem_id="single_tab"):
                gr.Markdown("### Upload a transparent car image and customize placement")
                
                with gr.Row():
                    # Left Column - Input Images
                    with gr.Column(scale=1):
                        gr.Markdown("#### Input Images")
                        car_input = gr.Image(
                            label="Car Image (PNG/WebP/AVIF with transparency)", 
                            type="pil",
                            image_mode=None,
                            sources=["upload", "clipboard"],
                            elem_id="car_input"
                        )
                        
                        template_dropdown = gr.Dropdown(
                            choices=get_template_choices(),
                            value="dnm_pkg_lot",
                            label="Select Template",
                            info="Choose a background template"
                        )
                        
                        template_preview = gr.Image(
                            label="Template Preview",
                            type="pil",
                            height=150,
                            interactive=False
                        )
                        
                        price_input = gr.Textbox(
                            label="Price (optional)",
                            placeholder="e.g., $426",
                            value="",
                            info="Add price overlay to template"
                        )
                        
                    # Right Column - Output and Controls
                    with gr.Column(scale=1):
                        output_image = gr.Image(
                            label="Generated Image", 
                            type="pil",
                            height=400
                        )
                        
                        gr.Markdown("#### Positioning Controls")
                        
                        car_scale = gr.Slider(
                            minimum=0.1, 
                            maximum=3.0, 
                            value=1.6, 
                            step=0.1,
                            label="Car Scale"
                        )
                        
                        with gr.Row():
                            x_position = gr.Radio(
                                choices=['left', 'center', 'right'],
                                value='center',
                                label="X Position"
                            )
                            y_position = gr.Radio(
                                choices=['top', 'center', 'bottom'],
                                value='bottom',
                                label="Y Position"
                            )
                        
                        with gr.Row():
                            x_offset = gr.Slider(
                                minimum=-500, 
                                maximum=500, 
                                value=0, 
                                step=10,
                                label="X Offset"
                            )
                            y_offset = gr.Slider(
                                minimum=-500, 
                                maximum=500, 
                                value=-50, 
                                step=10,
                                label="Y Offset"
                            )
                        
                        alpha_threshold = gr.Slider(
                            minimum=0, 
                            maximum=255, 
                            value=128, 
                            step=1,
                            label="Alpha Threshold",
                            info="Threshold for detecting car boundaries"
                        )
                
                # Bottom - Process button and info
                with gr.Row():
                    process_btn = gr.Button("Place Car", variant="primary", size="lg")
                    
                with gr.Row():
                    info_text = gr.Textbox(label="Info", lines=3)
            
            # Bulk Processing Tab
            with gr.TabItem("Bulk Processing", elem_id="bulk_tab"):
                gr.Markdown("### Process multiple cars from a CSV file")
                gr.Markdown("Upload a CSV with car URLs and prices. All images will be processed with optimized default settings.")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### CSV Format")
                        gr.Markdown("""
                        Your CSV should have two columns:
                        ```
                        Car URL,Price
                        https://example.com/car1.png,$25,999
                        https://example.com/car2.png,$32,500
                        ```
                        
                        **Important:** Use "Copy image address" when right-clicking on car images to get the direct image URL.
                        """)
                        
                        csv_input = gr.File(
                            label="Upload CSV File",
                            file_types=[".csv"],
                            type="filepath"
                        )
                        
                        process_bulk_btn = gr.Button("Process All Cars", variant="primary", size="lg")
                        
                    with gr.Column():
                        gr.Markdown("#### Results")
                        output_file = gr.File(
                            label="Download Results (ZIP)",
                            visible=True
                        )
                        summary_text = gr.Textbox(
                            label="Processing Summary",
                            lines=6,
                            interactive=False
                        )
                
                # Sample CSV download
                with gr.Accordion("Download Sample CSV", open=False):
                    sample_btn = gr.Button("Create Sample CSV")
                    sample_file = gr.File(label="Sample CSV", visible=False)
                    
                    def create_sample_csv():
                        sample_data = {
                            'Car URL': [
                                'https://example.com/car1.png',
                                'https://example.com/car2.png',
                                'https://example.com/car3.png'
                            ],
                            'Price': ['$25,999', '$32,500', '$28,750']
                        }
                        df = pd.DataFrame(sample_data)
                        temp_path = "sample_car_urls.csv"
                        df.to_csv(temp_path, index=False)
                        return gr.File(value=temp_path, visible=True)
                    
                    sample_btn.click(
                        fn=create_sample_csv,
                        outputs=[sample_file]
                    )
        
        # Documentation Tab
        with gr.Accordion("Tips & Documentation", open=False):
            gr.Markdown("""
            ### Supported Image Formats
            **Car Image (with transparency):**
            - **PNG**: Classic format with full alpha channel support
            - **WebP**: Modern format with better compression and transparency
            - **AVIF**: Next-gen format with excellent compression and alpha support
            - **JPEG**: ❌ Not supported - no transparency support
            
            **Background Image:**
            - All common formats: JPEG, PNG, WebP, AVIF, BMP, etc.
            
            ### Usage Tips
            - **Car Image**: Must have transparent backgrounds (RGBA format)
            - **Scale**: Adjust the size of the car relative to the background
            - **Position**: Choose base position, then fine-tune with offsets
            - **Alpha Threshold**: Lower values include more semi-transparent pixels
            - **Bulk Processing**: Uses optimized defaults that work 90% of the time
            
            ### Error Handling
            The tool will show clear errors if you upload incompatible images or formats.
            """)
        
        # Function to update template preview
        def update_template_preview(template_key):
            template_img = load_template(template_key)
            if template_img:
                # Resize for preview
                template_img.thumbnail((300, 200), Image.Resampling.LANCZOS)
                return template_img
            return None
        
        # Wire up template preview update
        template_dropdown.change(
            fn=update_template_preview,
            inputs=[template_dropdown],
            outputs=[template_preview]
        )
        
        # Initialize template preview
        demo.load(
            fn=lambda: update_template_preview("dnm_pkg_lot"),
            outputs=[template_preview]
        )
        
        # Wire up the single image processing
        process_btn.click(
            fn=place_car_on_background,
            inputs=[
                car_input, 
                template_dropdown, 
                car_scale, 
                x_position, 
                y_position, 
                x_offset, 
                y_offset,
                alpha_threshold,
                price_input
            ],
            outputs=[output_image, info_text]
        )
        
        # Auto-process on parameter change for single image
        for input_component in [car_scale, x_position, y_position, x_offset, y_offset, alpha_threshold, template_dropdown, price_input]:
            input_component.change(
                fn=place_car_on_background,
                inputs=[
                    car_input, 
                    template_dropdown, 
                    car_scale, 
                    x_position, 
                    y_position, 
                    x_offset, 
                    y_offset,
                    alpha_threshold,
                    price_input
                ],
                outputs=[output_image, info_text]
            )
        
        # Wire up the bulk processing
        process_bulk_btn.click(
            fn=process_bulk_csv,
            inputs=[csv_input],
            outputs=[output_file, summary_text]
        )
    
    return demo

# Launch the unified interface
if __name__ == "__main__":
    demo = create_unified_interface()
    # Use PORT environment variable for Railway deployment, fallback to 5000
    port = int(os.environ.get("PORT", 5000))
    demo.launch(
        share=False,  # Set to True to create public link
        debug=False,  # Disable debug in production
        server_name="0.0.0.0",  # Allow access from other machines
        server_port=port
    )