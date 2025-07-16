import gradio as gr
from PIL import Image
import numpy as np
import tempfile
import os
from io import BytesIO

def ensure_rgba_with_transparency(file_path):
    """Force proper transparency handling by reading raw bytes"""
    print(f"🔍 ensure_rgba: Reading raw bytes from {file_path}")
    
    # Read the raw file bytes and open through BytesIO to preserve transparency
    with open(file_path, 'rb') as f:
        img_bytes = f.read()
    
    img = Image.open(BytesIO(img_bytes))
    print(f"🔍 ensure_rgba: Raw file mode: {img.mode}, size: {img.size}")
    print(f"🔍 ensure_rgba: Image format: {img.format}")
    print(f"🔍 ensure_rgba: Image info: {img.info}")
    
    # First try to convert to RGBA and check if it has meaningful transparency
    result = img.convert('RGBA')
    alpha_array = np.array(result)[:, :, 3]
    unique_alphas = np.unique(alpha_array)
    transparent_pixels = np.sum(alpha_array < 255)
    total_pixels = alpha_array.size
    
    print(f"🔍 ensure_rgba: Alpha values: {unique_alphas[:10]}...")
    print(f"🔍 ensure_rgba: Non-opaque pixels: {transparent_pixels}/{total_pixels} ({transparent_pixels/total_pixels*100:.1f}%)")
    
    # Check if we have meaningful transparency (not just all opaque or all transparent)
    if len(unique_alphas) > 1 and transparent_pixels > 0 and transparent_pixels < total_pixels * 0.95:
        print("🔍 ensure_rgba: Found meaningful transparency, using as-is")
        return result
    elif transparent_pixels == 0:
        print("🔍 ensure_rgba: No transparency detected, trying smart background removal")
        return smart_background_removal(img)
    else:
        print("🔍 ensure_rgba: All/most pixels transparent, trying smart background removal")
        return smart_background_removal(img)

def smart_background_removal(img):
    """Smart background removal that preserves drop shadows and semi-transparent pixels"""
    print(f"🔍 smart_background_removal: Input mode: {img.mode}")
    img_rgba = img.convert('RGBA')
    data = np.array(img_rgba)
    
    # Sample corner pixels and edge pixels to detect background color
    height, width = data.shape[:2]
    
    # Get more edge samples, not just corners
    edge_samples = []
    # Top and bottom edges
    for i in range(0, width, max(1, width//20)):
        edge_samples.append(data[0, i, :3])        # top edge
        edge_samples.append(data[height-1, i, :3]) # bottom edge
    # Left and right edges  
    for i in range(0, height, max(1, height//20)):
        edge_samples.append(data[i, 0, :3])        # left edge
        edge_samples.append(data[i, width-1, :3])  # right edge
    
    print(f"🔍 smart_background_removal: Collected {len(edge_samples)} edge samples")
    
    # Find the most common background color among edge samples
    edge_samples = np.array(edge_samples)
    
    # Use a more conservative approach - only remove pixels that are very close to background
    # and are likely pure background (not semi-transparent elements like shadows)
    best_bg_color = None
    best_score = 0
    
    for sample_color in edge_samples:
        # Calculate how many pixels match this color closely
        color_diff = np.sqrt(np.sum((data[:, :, :3] - sample_color) ** 2, axis=2))
        close_pixels = np.sum(color_diff < 30)  # Stricter threshold
        
        # Check if these pixels form a coherent background (mainly on edges)
        close_mask = color_diff < 30
        
        # Score based on how many edge pixels match vs total matches
        edge_matches = (
            np.sum(close_mask[0, :]) +      # top edge
            np.sum(close_mask[-1, :]) +     # bottom edge  
            np.sum(close_mask[:, 0]) +      # left edge
            np.sum(close_mask[:, -1])       # right edge
        )
        
        edge_score = edge_matches / max(1, close_pixels) if close_pixels > 0 else 0
        
        if edge_score > best_score and close_pixels > width * height * 0.1:  # At least 10% of image
            best_score = edge_score
            best_bg_color = sample_color
    
    if best_bg_color is not None:
        print(f"🔍 smart_background_removal: Found background color {best_bg_color} with score {best_score:.3f}")
        
        # Remove background more conservatively
        color_diff = np.sqrt(np.sum((data[:, :, :3] - best_bg_color) ** 2, axis=2))
        
        # Only remove pixels that are very close to background color (stricter threshold)
        background_mask = color_diff < 25
        
        # Additional check: don't remove pixels that might be part of shadows
        # Shadows are typically darker than background, so preserve darker pixels
        brightness = np.mean(data[:, :, :3], axis=2)
        bg_brightness = np.mean(best_bg_color)
        
        # Don't remove pixels that are significantly darker (potential shadows)
        shadow_mask = brightness < bg_brightness * 0.8
        background_mask = background_mask & ~shadow_mask
        
        potential_bg_pixels = np.sum(background_mask)
        bg_percentage = potential_bg_pixels / (data.shape[0] * data.shape[1]) * 100
        
        print(f"🔍 smart_background_removal: Would remove {bg_percentage:.1f}% of pixels")
        
        if bg_percentage > 15 and bg_percentage < 80:  # Reasonable range
            data[background_mask, 3] = 0  # Set alpha to 0 (transparent)
            
            result = Image.fromarray(data, 'RGBA')
            
            # Verify transparency was added
            alpha_check = np.array(result)[:, :, 3]
            transparent_pixels = np.sum(alpha_check == 0)
            total_pixels = alpha_check.size
            print(f"🔍 smart_background_removal: Result: {transparent_pixels}/{total_pixels} ({transparent_pixels/total_pixels*100:.1f}%) transparent")
            
            return result
    
    print("🔍 smart_background_removal: No suitable background found, returning original as RGBA")
    return img_rgba

def get_car_bounds(car_image, alpha_threshold):
    """Get actual car boundaries (non-transparent pixels)"""
    print(f"🔍 get_car_bounds: Input image mode={car_image.mode}, size={car_image.size}")
    car_array = np.array(car_image)
    print(f"🔍 get_car_bounds: Array shape={car_array.shape}, dtype={car_array.dtype}")
    
    if car_array.shape[2] == 4:  # RGBA
        print("🔍 get_car_bounds: RGBA mode detected")
        alpha = car_array[:, :, 3]
        print(f"🔍 get_car_bounds: Alpha channel min={alpha.min()}, max={alpha.max()}")
        unique_alphas = np.unique(alpha)
        print(f"🔍 get_car_bounds: Unique alpha values: {unique_alphas[:10]}...")
    else:  # RGB
        print("🔍 get_car_bounds: RGB mode detected, creating fake alpha channel")
        alpha = np.ones((car_array.shape[0], car_array.shape[1]), dtype=np.uint8) * 255
        print(f"🔍 get_car_bounds: Fake alpha channel min={alpha.min()}, max={alpha.max()}")
    
    print(f"🔍 get_car_bounds: Using alpha_threshold={alpha_threshold}")
    # Try different thresholds - look for more opaque pixels
    non_transparent = alpha > alpha_threshold  # Use middle threshold to find solid parts
    rows, cols = np.where(non_transparent)
    
    print(f"🔍 get_car_bounds: Found {len(rows)} non-transparent pixels")
    if len(rows) == 0:
        print("❌ get_car_bounds: No non-transparent pixels found!")
        return None
    
    min_y, max_y = rows.min(), rows.max()
    min_x, max_x = cols.min(), cols.max()
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    
    print(f"🔍 get_car_bounds: Bounds found - min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")
    print(f"🔍 get_car_bounds: Content size - width={width}, height={height}")
    
    return min_x, min_y, max_x, max_y, width, height

def compose_images(car_image, background_image, car_scale=1.6, x_pos='center', y_pos='bottom', x_offset=0, y_offset=-50):
    """Compose car image onto background - exact copy of main.py logic"""
    print(f"🔍 compose_images: Starting composition")
    if car_image is None or background_image is None:
        print("❌ compose_images: One or both images are None")
        return None
    
    alpha_threshold = 128
    
    # Images are already properly converted in the handler
    car = car_image
    background = background_image
    
    print(f"🔍 compose_images: Car mode={car.mode}, size={car.size}")
    print(f"🔍 compose_images: Background mode={background.mode}, size={background.size}")
    
    # Get actual car bounds
    print(f"🔍 compose_images: Getting car bounds...")
    bounds = get_car_bounds(car, alpha_threshold)
    if bounds is None:
        print("❌ compose_images: No bounds found!")
        return None
    min_x, min_y, max_x, max_y, car_width, car_height = bounds
    print(f"🔍 compose_images: Original car bounds: ({min_x}, {min_y}, {max_x}, {max_y}), size=({car_width}, {car_height})")
    
    # Resize car
    base_width = int(background.size[0] * 0.6)  # Base size
    target_width = int(base_width * car_scale)
    aspect_ratio = car_width / car_height
    
    scale_factor = target_width / car_width
    new_car_width = int(car.size[0] * scale_factor)
    new_car_height = int(car.size[1] * scale_factor)
    
    print(f"🔍 compose_images: Scale calculations:")
    print(f"    base_width={base_width}, car_scale={car_scale}")
    print(f"    target_width={target_width}, aspect_ratio={aspect_ratio:.3f}")
    print(f"    scale_factor={scale_factor:.3f}")
    print(f"    new dimensions: {new_car_width}x{new_car_height}")
    
    car_resized = car.resize((new_car_width, new_car_height), Image.Resampling.LANCZOS)
    print(f"🔍 compose_images: Car resized to {car_resized.size}, mode={car_resized.mode}")
    
    # Get new bounds
    print(f"🔍 compose_images: Getting resized car bounds...")
    new_bounds = get_car_bounds(car_resized, alpha_threshold)
    if new_bounds is None:
        print("❌ compose_images: No bounds found after resize!")
        return None
    new_min_x, new_min_y, new_max_x, new_max_y, new_car_width, new_car_height = new_bounds
    print(f"🔍 compose_images: Resized car bounds: ({new_min_x}, {new_min_y}, {new_max_x}, {new_max_y}), size=({new_car_width}, {new_car_height})")
    
    # Calculate position
    print(f"🔍 compose_images: Calculating position with x_pos='{x_pos}', y_pos='{y_pos}'")
    if x_pos == 'left':
        paste_x = -new_min_x
    elif x_pos == 'center':
        paste_x = (background.size[0] - new_car_width) // 2 - new_min_x
    elif x_pos == 'right':
        paste_x = background.size[0] - new_max_x
    else:
        paste_x = int(x_pos) if str(x_pos).isdigit() else 0
    
    if y_pos == 'top':
        paste_y = -new_min_y
    elif y_pos == 'center':
        paste_y = (background.size[1] - new_car_height) // 2 - new_min_y
    elif y_pos == 'bottom':
        paste_y = background.size[1] - new_max_y - 50
    else:
        paste_y = int(y_pos) if str(y_pos).isdigit() else 0
    
    # Apply offsets
    paste_x += x_offset
    paste_y += y_offset
    
    print(f"🔍 compose_images: Final paste position: ({paste_x}, {paste_y})")
    
    # Create composite
    composite = background.copy()
    print(f"🔍 compose_images: Background copied, mode={composite.mode}, size={composite.size}")
    
    composite.paste(car_resized, (paste_x, paste_y), car_resized)
    print(f"🔍 compose_images: Car pasted successfully!")
    print(f"🔍 compose_images: Final composite mode={composite.mode}, size={composite.size}")
    
    return composite

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Car Image Composer") as demo:
        gr.Markdown("# 🚗 Car Image Composer")
        gr.Markdown("Upload a transparent car image and a background image to create a composite.")
        
        with gr.Row():
            with gr.Column():
                car_input = gr.Image(label="Car Image (Transparent)", type="filepath")
                background_input = gr.Image(label="Background Image", type="filepath")
                
                with gr.Row():
                    car_scale = gr.Slider(0.1, 3.0, value=1.6, label="Car Scale")
                    x_pos = gr.Dropdown(['left', 'center', 'right'], value='center', label="X Position")
                    y_pos = gr.Dropdown(['top', 'center', 'bottom'], value='bottom', label="Y Position")
                
                with gr.Row():
                    x_offset = gr.Number(value=0, label="X Offset (px)")
                    y_offset = gr.Number(value=-50, label="Y Offset (px)")
                
                compose_btn = gr.Button("🎨 Compose Images", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="Composed Image", type="pil")
                download_btn = gr.DownloadButton(label="📥 Download Image", visible=False)
        
        def process_and_prepare_download(car_path, bg_path, scale, x_pos, y_pos, x_off, y_off):
            print(f"🔍 HANDLER: Starting with car_path={car_path}, bg_path={bg_path}")
            print(f"🔍 HANDLER: Parameters - scale={scale}, x_pos={x_pos}, y_pos={y_pos}, x_off={x_off}, y_off={y_off}")
            
            if car_path is None or bg_path is None:
                print("❌ HANDLER: One or both paths are None")
                return None, None
            
            # Check if files exist
            if not os.path.exists(car_path):
                print(f"❌ HANDLER: Car path does not exist: {car_path}")
                return None, None
            if not os.path.exists(bg_path):
                print(f"❌ HANDLER: Background path does not exist: {bg_path}")
                return None, None
            
            # Get file info
            print(f"🔍 HANDLER: Car file size: {os.path.getsize(car_path)} bytes")
            print(f"🔍 HANDLER: Background file size: {os.path.getsize(bg_path)} bytes")
            
            # Open images with proper transparency handling using BytesIO method
            print(f"🔍 HANDLER: Opening car image from {car_path}")
            try:
                car_img = ensure_rgba_with_transparency(car_path)
                print(f"🔍 HANDLER: Final car image mode: {car_img.mode}, size: {car_img.size}")
                
                # Final alpha channel check
                car_array = np.array(car_img)
                print(f"🔍 HANDLER: Final car array shape: {car_array.shape}")
                if car_array.shape[2] == 4:
                    alpha_channel = car_array[:, :, 3]
                    print(f"🔍 HANDLER: Final alpha channel min={alpha_channel.min()}, max={alpha_channel.max()}")
                    unique_alphas = np.unique(alpha_channel)
                    print(f"🔍 HANDLER: Final unique alpha values: {unique_alphas[:20]}...")  # First 20
                    transparent_pixels = np.sum(alpha_channel == 0)
                    total_pixels = alpha_channel.size
                    print(f"🔍 HANDLER: Final transparent pixels: {transparent_pixels}/{total_pixels} ({transparent_pixels/total_pixels*100:.1f}%)")
                
            except Exception as e:
                print(f"❌ HANDLER: Error opening car image: {e}")
                return None, None
            
            print(f"🔍 HANDLER: Opening background image from {bg_path}")
            try:
                bg_img_original = Image.open(bg_path)
                print(f"🔍 HANDLER: Background image original mode: {bg_img_original.mode}, size: {bg_img_original.size}")
                print(f"🔍 HANDLER: Background image format: {bg_img_original.format}")
                
                bg_img = bg_img_original.convert("RGB")
                print(f"🔍 HANDLER: Background image after RGB conversion: mode={bg_img.mode}, size={bg_img.size}")
                
            except Exception as e:
                print(f"❌ HANDLER: Error opening background image: {e}")
                return None, None
            
            print("🔍 HANDLER: Calling compose_images...")
            result = compose_images(car_img, bg_img, scale, x_pos, y_pos, int(x_off), int(y_off))
            
            if result is None:
                print("❌ HANDLER: compose_images returned None")
                return None, None
            
            print(f"🔍 HANDLER: Result image mode: {result.mode}, size: {result.size}")
            
            # Return the PIL image directly for gradio display
            # Gradio will handle the temporary file creation internally
            return result, None
        
        compose_btn.click(
            fn=process_and_prepare_download,
            inputs=[car_input, background_input, car_scale, x_pos, y_pos, x_offset, y_offset],
            outputs=[output_image, download_btn]
        )
    
    return demo

if __name__ == "__main__":
    app = create_interface()
    app.launch(share=True)