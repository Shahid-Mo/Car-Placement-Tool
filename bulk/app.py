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

# Define available templates (keeping your existing templates)
TEMPLATES = {
    "dnm_pkg_lot": {
        "path": "/home/shahidmo/hoegger/megan_workflow/dnm_pkg_lot.jpg",
        "name": "D&M Leasing Package",
        "price_coords": (150, 104),
        "description": "D&M Leasing template with red price box"
    },
    "base_1": {
        "path": "Base_1.jpg", 
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
    if not price_text or not price_text.strip():
        return image
    
    # Get template-specific price coordinates
    if template_key in TEMPLATES:
        price_coords = TEMPLATES[template_key]["price_coords"]
        price_x, price_y = price_coords
    else:
        # Default coordinates
        price_x, price_y = 150, 104
    
    # Create a copy to draw on
    img_with_price = image.copy()
    draw = ImageDraw.Draw(img_with_price)
    
    # Try to load font, fallback to default
    try:
        font_path = "/home/shahidmo/.local/share/fonts/montserrat/Montserrat.ttf"
        font_size = 100
        font = ImageFont.truetype(font_path, font_size)
    except:
        # Fallback to default font if Montserrat not available
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    if font:
        # Text styling
        text_color = (255, 255, 255)  # White
        stroke_color = (255, 255, 255)  # White stroke
        stroke_width = 4
        
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
    
    return img_with_price

def get_car_bounds(car_image, alpha_threshold):
    """Get actual car boundaries (non-transparent pixels)"""
    car_array = np.array(car_image)
    if car_array.shape[2] < 4:
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

def process_single_car(car_image, price_text):
    """Process a single car image with default settings"""
    try:
        # Load template
        background = load_template(DEFAULT_TEMPLATE)
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
            composite = add_price_overlay(composite, price_text, DEFAULT_TEMPLATE)
        
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

def create_simple_bulk_interface():
    """Create a simplified Gradio interface for bulk processing"""
    
    with gr.Blocks(title="Bulk Car Placement Tool") as demo:
        gr.Markdown("# 🚗 Bulk Car Placement Tool")
        gr.Markdown("Simply upload a CSV file with car image URLs and prices. All images will be processed with the default settings that work 90% of the time.")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### CSV Format")
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
                
                process_btn = gr.Button("Process All Cars", variant="primary", size="lg")
                
            with gr.Column():
                gr.Markdown("### Results")
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
        
        # Process button click
        process_btn.click(
            fn=process_bulk_csv,
            inputs=[csv_input],
            outputs=[output_file, summary_text]
        )
    
    return demo

# Launch the interface
if __name__ == "__main__":
    demo = create_simple_bulk_interface()
    demo.launch(
        share=True,
        debug=True,
        server_name="0.0.0.0",
        server_port=7860
    )