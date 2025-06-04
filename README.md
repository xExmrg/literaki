# Literaki Board Tools

This repository contains utilities for working with a Literaki/Scrabble style board.

## Files
- **board.py** – programmatic representation of the board layout.
- **tiles.py** – definitions for tile distribution and a simple player rack.
- **board_layout.txt** – textual layout of bonus squares as produced by OCR.
- **improved_ocr.py** – script that extracts a board layout from an image using
  Tesseract OCR combined with basic color classification.

The OCR script expects an image of a 15×15 board (such as `board.png`).
It divides the board into individual squares, runs OCR on each square and
falls back to colour heuristics if no text is recognised.

Run it with:

```bash
python improved_ocr.py board.png -o board_layout.txt
```

Dependencies are listed in `requirements.txt`.
