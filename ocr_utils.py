import os
from datetime import datetime
import cv2
import numpy as np

# Mapping of OCR characters to valid Literaki letters including common mistakes
CHAR_MAPPING = {
    'A': 'A', 'Ą': 'Ą', 'B': 'B', 'C': 'C', 'Ć': 'Ć', 'D': 'D',
    'E': 'E', 'Ę': 'Ę', 'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I',
    'J': 'J', 'K': 'K', 'L': 'L', 'Ł': 'Ł', 'M': 'M', 'N': 'N',
    'Ń': 'Ń', 'O': 'O', 'Ó': 'Ó', 'P': 'P', 'R': 'R', 'S': 'S',
    'Ś': 'Ś', 'T': 'T', 'U': 'U', 'W': 'W', 'Y': 'Y', 'Z': 'Z',
    'Ź': 'Ź', 'Ż': 'Ż', '_': '_',
    # Common OCR mistakes
    '0': 'O', '1': 'I', '6': 'G', '8': 'B'
}


def preprocess_image(image_bgr, prefix, debug_dir="ocr_debug"):
    """Apply common preprocessing steps to improve OCR accuracy."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    cv2.imwrite(f"{debug_dir}/{prefix}_original_{timestamp}.png", image_bgr)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(blurred, -1, kernel)
    cv2.imwrite(f"{debug_dir}/{prefix}_preprocessed_{timestamp}.png", sharpened)

    return sharpened
