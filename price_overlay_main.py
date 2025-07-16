from PIL import Image, ImageDraw, ImageFont
import os

def add_price_to_template(template_path, price_text, output_path):
    """
    Add price text to the D&M Leasing template
    
    Args:
        template_path: Path to the template image
        price_text: Price text to add (e.g., "$426")
        output_path: Path to save the output image
    """
    
    # Load the template
    img = Image.open(template_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # Price positioning (adjust these based on your template)
    # These coordinates are for the red box area
    price_x = 150  # X position (center of red box)
    price_y = 104  # Y position (below "$" symbol area)
    
    
    # Try to load a good font, fallback to default if not available
    font_path = "/home/shahidmo/.local/share/fonts/montserrat/Montserrat.ttf"
    font_size = 100  # Bigger size for visual impact
    font = ImageFont.truetype(font_path, font_size)
    
    # Add the price text
    # White color for the text
    # White text with thick stroke
    text_color = (255, 255, 255)         # White
    stroke_color =  (255, 255, 255)       # Black stroke
    stroke_width = 4                     # Thickness around text
    
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
    
    # Save the result
    img.save(output_path, quality=95)
    print(f"Saved: {output_path}")

# Example usage
if __name__ == "__main__":
    # Simple usage
    add_price_to_template(
        template_path="dnm_pkg_lot.jpg",
        price_text="$426",
        output_path="dm_leasing_with_price.jpg"
    )

