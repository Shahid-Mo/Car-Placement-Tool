import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import base64
import os

# Define available templates
TEMPLATES = {
    "dnm_pkg_lot": {
        "path": "dnm_pkg_lot.jpg",
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

def add_price_overlay(image, price_text, template_key):
    """Add price overlay to image based on template"""
    # Initialize warnings list to track defaults used
    warnings = []
    if not price_text or not price_text.strip():
        return image
    
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
        font_path = "/home/shahidmo/.local/share/fonts/montserrat/Montserrat.ttf"
        font_size = 100
        font = ImageFont.truetype(font_path, font_size)
        font_used = f"Montserrat ({font_size}px)"
    except Exception as e:
        # Fallback to default font if Montserrat not available
        try:
            font = ImageFont.load_default()
            font_used = "System Default Font"
            warnings.append(f"⚠️ Montserrat font not found - using system default font")
        except:
            font = None
            font_used = "No Font Available"
            warnings.append(f"❌ No fonts available - text rendering may fail")
    
    if font:
        # Text styling - exactly from price_overlay_main.py
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
    
    return img_with_price, warnings

def get_car_bounds(car_image, alpha_threshold):
    """Get actual car boundaries (non-transparent pixels)"""
    car_array = np.array(car_image)
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

def create_gradio_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Car Placement Tool") as demo:
        gr.Markdown("# 🚗 Car Placement Tool")
        gr.Markdown("Upload a transparent car image and select a template to place the car.")
        
        with gr.Row():
            # Left Column - Input Images
            with gr.Column(scale=1):
                gr.Markdown("### Input Images")
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
                
                gr.Markdown("### Positioning Controls")
                
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
        
        # Add examples
        gr.Markdown("### Tips")
        gr.Markdown("""
        - **Car Image**: Supports PNG, WebP, AVIF with transparent backgrounds (RGBA format)
        - **Background**: Supports JPEG, PNG, WebP, AVIF
        - **Scale**: Adjust the size of the car relative to the background
        - **Position**: Choose base position, then fine-tune with offsets
        - **Alpha Threshold**: Lower values include more semi-transparent pixels
        - **Formats**: WebP and AVIF offer better compression while maintaining transparency
        """)
        
        # Add format info
        with gr.Accordion("Supported Image Formats", open=False):
            gr.Markdown("""
            **Car Image (with transparency):**
            - **PNG**: Classic format with full alpha channel support
            - **WebP**: Modern format with better compression and transparency
            - **AVIF**: Next-gen format with excellent compression and alpha support
            - **JPEG**: ❌ Not supported - no transparency support
            
            **Background Image:**
            - All common formats: JPEG, PNG, WebP, AVIF, BMP, etc.
            
            **Error Handling**: The tool will show a clear error if you upload a car image without transparency.
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
        
        # Wire up the processing
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
        
        # Auto-process on parameter change (optional)
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
    
    return demo

# Alternative: Handle transparency preservation with custom preprocessing
def preserve_transparency_interface():
    """Alternative interface with custom image handling for multiple formats"""
    
    def process_with_preserved_alpha(car_file, background_file, *args):
        """Process files while preserving alpha channel"""
        if car_file is None or background_file is None:
            return None, "Please upload both images"
        
        try:
            # Read car image carefully to preserve alpha
            car_image = Image.open(car_file)
            
            # Handle different formats
            if car_image.mode == 'RGBA':
                pass  # Already good
            elif car_image.mode == 'LA':
                # Grayscale with alpha - convert to RGBA
                car_image = car_image.convert('RGBA')
            elif car_image.mode == 'P':
                # Palette mode - check for transparency
                if 'transparency' in car_image.info:
                    car_image = car_image.convert('RGBA')
                else:
                    raise ValueError("Car image must have transparency. The uploaded palette image doesn't have transparent pixels.")
            else:
                # No alpha channel
                raise ValueError(f"Car image must have transparency (alpha channel). Uploaded image is in '{car_image.mode}' mode. Please use PNG, WebP, or AVIF with transparent background.")
            
            # Read background
            background_image = Image.open(background_file)
            
            # Process with main function
            return place_car_on_background(car_image, background_image, *args)
        except Exception as e:
            return None, f"Error loading images: {str(e)}"
    
    with gr.Blocks(title="Car Placement Tool") as demo:
        gr.Markdown("# 🚗 Car Placement Tool (File Upload Version)")
        gr.Markdown("Supports PNG, WebP, AVIF with transparency")
        
        with gr.Row():
            car_file = gr.File(
                label="Car Image", 
                file_types=[".png", ".webp", ".avif", ".jpg", ".jpeg"]
            )
            background_file = gr.File(
                label="Background Image", 
                file_types=[".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp"]
            )
        
        # ... (rest of the controls remain the same)
        
    return demo

# Launch the main interface
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        share=True,  # Creates a public link to share with your team
        debug=True,
        server_name="0.0.0.0",  # Allow access from other machines
        server_port=7860
    )
    
    
