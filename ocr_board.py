from PIL import Image
import pytesseract

BOARD_SIZE = 15


def read_board_squares(image_path: str):
    """Read each square of the Literaki board screenshot using OCR.

    Returns a BOARD_SIZE x BOARD_SIZE list of recognized letters. Empty
    strings or None indicate that OCR did not detect a letter in that
    square.
    """
    image = Image.open(image_path)
    width, height = image.size
    square_w = width / BOARD_SIZE
    square_h = height / BOARD_SIZE

    board_letters = []
    for r in range(BOARD_SIZE):
        row = []
        for c in range(BOARD_SIZE):
            left = int(c * square_w)
            upper = int(r * square_h)
            right = int((c + 1) * square_w)
            lower = int((r + 1) * square_h)
            crop = image.crop((left, upper, right, lower))
            text = pytesseract.image_to_string(
                crop,
                config="--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                lang="eng",
            )
            text = text.strip().upper()
            row.append(text if text else None)
        board_letters.append(row)
    return board_letters


def main():
    board = read_board_squares("board.png")
    for row in board:
        print(" ".join(letter if letter else "." for letter in row))


if __name__ == "__main__":
    main()
