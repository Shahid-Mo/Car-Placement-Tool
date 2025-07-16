def find_price_position(template_path, initial_x=170, initial_y=110):
    """Interactive tool to find the right position for price text"""
    import cv2
    import numpy as np
    
    img = cv2.imread(template_path)
    x, y = initial_x, initial_y
    font_size = 90
    
    def update_image():
        temp_img = img.copy()
        # Draw crosshair
        cv2.line(temp_img, (x-20, y), (x+20, y), (0, 255, 0), 2)
        cv2.line(temp_img, (x, y-20), (x, y+20), (0, 255, 0), 2)
        # Add text preview
        cv2.putText(temp_img, "$426", (x-60, y+30), 
                   cv2.FONT_HERSHEY_BOLD, 3, (255, 255, 255), 4)
        return temp_img
    
    print("Use arrow keys to move, +/- for font size, 's' to save position, 'q' to quit")
    
    while True:
        cv2.imshow('Position Finder', update_image())
        key = cv2.waitKey(0)
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            print(f"Position: x={x}, y={y}, font_size={font_size}")
        elif key == 82:  # Up arrow
            y -= 5
        elif key == 84:  # Down arrow
            y += 5
        elif key == 81:  # Left arrow
            x -= 5
        elif key == 83:  # Right arrow
            x += 5
        elif key == ord('+'):
            font_size += 5
        elif key == ord('-'):
            font_size -= 5
    
    cv2.destroyAllWindows()
    return x, y, font_size

# Use this to find coordinates:
# x, y, size = find_price_position("dm_leasing_template.jpg")