# literaki

This repository contains helper scripts for working with the board and tiles of the Polish "Literaki" game.

## OCR board reading

`ocr_board.py` provides a basic routine that reads each square of a board screenshot using [Tesseract OCR](https://github.com/tesseract-ocr/tesseract). The script splits the input image into a 15x15 grid and performs OCR on each cell.

Dependencies:

- `pytesseract`
- `Pillow`
- A local Tesseract installation available on the system path

Run the script from this directory to print the recognized board letters:

```bash
python3 ocr_board.py
```

OCR accuracy will depend heavily on the screenshot quality and configured language data.
