import argparse
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import pytesseract
import cv2


def _classify_color(bgr):
    """Classify a BGR tuple into a board token based on its hue."""
    b, g, r = bgr
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv
    if v < 60 or s < 40:
        return 'W'
    if h < 15 or h > 160:
        return 'R'
    if 15 <= h < 40:
        return 'Y'
    if 40 <= h < 80:
        return 'G'
    if 100 <= h < 140:
        return 'B'
    return 'W'


def _ocr_square(square_img):
    gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = '--psm 8 -c tessedit_char_whitelist=123xX'
    text = pytesseract.image_to_string(th, config=config)
    text = text.strip().upper()
    return text


def extract_board_layout(image_path: Path, grid_size: int = 15) -> List[List[str]]:
    """Return a grid_size x grid_size array describing the board layout."""
    img = np.array(Image.open(image_path))
    h, w = img.shape[:2]
    cell_w = w // grid_size
    cell_h = h // grid_size
    layout = []
    for r in range(grid_size):
        row = []
        for c in range(grid_size):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            crop = img[y0:y1, x0:x1]
            token = _ocr_square(crop)
            if not token:
                token = _classify_color(cv2.mean(crop)[:3])
            row.append(token)
        layout.append(row)
    return layout


def layout_to_text(layout: List[List[str]]) -> str:
    return "\n".join(" ".join(row) for row in layout)


def main():
    parser = argparse.ArgumentParser(description="Extract board layout from an image using OCR and color heuristics.")
    parser.add_argument('image', type=Path, nargs='?', default='board.png', help='Path to board image')
    parser.add_argument('-o', '--output', type=Path, default='board_layout.txt', help='Where to store the layout text')
    args = parser.parse_args()

    layout = extract_board_layout(args.image)
    out_text = layout_to_text(layout)
    args.output.write_text(out_text, encoding='utf-8')
    print(f"Layout written to {args.output}")


if __name__ == '__main__':
    main()
