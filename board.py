BC_ORANGE = "orange"
BC_GREEN = "green"
BC_YELLOW = "yellow"
BC_BLUE = "blue"

BOARD_LAYOUT = [
    "R W 3x W W G W R W G W W 3x W R",
    "W W W W G W R W R W G W W W W",
    "3x W W G W 2x W Y W 2x W G W W 3x",
    "W W G W 2x W Y W Y W 2x W G W W",
    "W G W 2x W Y W W W Y W 2x W G W",
    "G W 2x W Y W W B W W Y W 2x W G",
    "W R W Y W W B W B W W Y W R W",
    "R W Y W W B W R W B W W Y W R",
    "W R W Y W W B W B W W Y W R W",
    "G W 2x W Y W W B W W Y W 2x W G",
    "W G W 2x W Y W W W Y W 2x W G W",
    "W W G W 2x W Y W Y W 2x W G W W",
    "3x W W G W 2x W Y W 2x W G W W 3x",
    "W W W W G W R W R W G W W W W",
    "R W 3x W W G W R W G W W 3x W R",
]


def create_board():
    board = []
    for row_idx, row in enumerate(BOARD_LAYOUT):
        tokens = row.split()
        board_row = []
        for col_idx, token in enumerate(tokens):
            cell = {"word_mult": 1, "board_color": None, "is_start": False}
            if token == "R":
                cell["board_color"] = BC_ORANGE
            elif token == "G":
                cell["board_color"] = BC_GREEN
            elif token == "Y":
                cell["board_color"] = BC_YELLOW
            elif token == "B":
                cell["board_color"] = BC_BLUE
            elif token == "2x":
                cell["word_mult"] = 2
            elif token == "3x":
                cell["word_mult"] = 3
                cell["board_color"] = BC_ORANGE
            else:
                pass
            board_row.append(cell)
        board.append(board_row)

    # Mark the starting square (center) as orange with double word multiplier
    start_row = 7
    start_col = 7
    board[start_row][start_col]["is_start"] = True
    board[start_row][start_col]["word_mult"] = 2
    board[start_row][start_col]["board_color"] = BC_ORANGE
    return board


if __name__ == "__main__":
    board = create_board()
    # Pretty print board as word multipliers and colors for quick check
    for row in board:
        row_repr = []
        for cell in row:
            token = "W"
            if cell["is_start"]:
                token = "S"
            elif cell["word_mult"] == 3:
                token = "3x"
            elif cell["word_mult"] == 2:
                token = "2x"
            elif cell["board_color"] == BC_ORANGE:
                token = "R"
            elif cell["board_color"] == BC_GREEN:
                token = "G"
            elif cell["board_color"] == BC_YELLOW:
                token = "Y"
            elif cell["board_color"] == BC_BLUE:
                token = "B"
            row_repr.append(f"{token:>2}")
        print(" ".join(row_repr))
