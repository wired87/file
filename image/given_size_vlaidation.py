

def validate_img_size(height: int, width: int) -> bool or None:
    if height <= 0 or width <= 0:
        return False

