from PIL import Image
import numpy as np
import sys

def get_car_bounds(car_image,alpha_threshold):
    """Get actual car boundaries (non-transparent pixels)"""
    car_array = np.array(car_image)
    
    alpha = car_array[:, :, 3]
    
    print(f"Alpha min: {alpha.min()}, Alpha max: {alpha.max()}")
    
    # Try different thresholds - look for more opaque pixels
    non_transparent = alpha > alpha_threshold  # Use middle threshold to find solid parts
    rows, cols = np.where(non_transparent)
    
    if len(rows) == 0:
        return None
    
    min_y, max_y = rows.min(), rows.max()
    min_x, max_x = cols.min(), cols.max()
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    
    return min_x, min_y, max_x, max_y, width, height

# ========================================
# MAIN TWEAKING AREA
# ========================================

car_path = 'cars_transparent/porsche_19_converted.png'
background_path = 'dnm_pkg_lot_with_price.jpg'
output_path = 'output.jpg'
alpha_threshold = 128 # Standard threshold for PNG (auto-converted)

# TWEAK THESE PARAMETERS
car_scale = 1.6       # Size of car (0.5 = 50%, 1.0 = 100%)
x_pos = 'center'       # 'left', 'center', 'right', or pixel number
y_pos = 'bottom'       # 'top', 'center', 'bottom', or pixel number
x_offset = 0           # Fine-tune X position
y_offset = -50         # Fine-tune Y position

# ========================================
# PROCESSING
# ========================================

# Auto-convert to PNG for consistent alpha processing
car_original = Image.open(car_path)
png_path = car_path.rsplit('.', 1)[0] + '_converted.png'
car_original.save(png_path, 'PNG')
print(f"Converted {car_path} to {png_path} for processing")

car = Image.open(png_path)
if car.mode != 'RGBA':
    raise ValueError("Input image is not RGBA. This pipeline requires a PNG image with transparency.")

car = car.convert('RGBA')  # Safe to convert now
background = Image.open(background_path).convert('RGB')

print(f"Car: {car.size}")
print(f"Background: {background.size}")

# Get actual car bounds
bounds = get_car_bounds(car, alpha_threshold)
min_x, min_y, max_x, max_y, car_width, car_height = bounds
print(f"Car min x: {min_x}")
print(f"Car max x: {max_x}")
print(f"Car min y: {min_y}")
print(f"Car max y: {max_y}")
print(f"Car bounds: {car_width}x{car_height}")

# Resize car
base_width = int(background.size[0] * 0.6)  # Base size
target_width = int(base_width * car_scale)
aspect_ratio = car_width / car_height
target_height = int(target_width / aspect_ratio)

scale_factor = target_width / car_width
new_car_width = int(car.size[0] * scale_factor)
new_car_height = int(car.size[1] * scale_factor)

car_resized = car.resize((new_car_width, new_car_height), Image.LANCZOS)

# Get new bounds
new_bounds = get_car_bounds(car_resized, alpha_threshold)
new_min_x, new_min_y, new_max_x, new_max_y, new_car_width, new_car_height = new_bounds




# Calculate position
if x_pos == 'left':
    paste_x = -new_min_x
elif x_pos == 'center':
    paste_x = (background.size[0] - new_car_width) // 2 - new_min_x
elif x_pos == 'right':
    paste_x = background.size[0] - new_max_x
else:
    paste_x = int(x_pos)

if y_pos == 'top':
    paste_y = -new_min_y
elif y_pos == 'center':
    paste_y = (background.size[1] - new_car_height) // 2 - new_min_y
elif y_pos == 'bottom':
    paste_y = background.size[1] - new_max_y - 50
else:
    paste_y = int(y_pos)

# Apply offsets
paste_x += x_offset
paste_y += y_offset

print(f"Final position: ({paste_x}, {paste_y})")

# Create composite
composite = background.copy()
composite.paste(car_resized, (paste_x, paste_y), car_resized)
composite.save(output_path, 'JPEG', quality=95)

print(f"✅ Saved: {output_path}")