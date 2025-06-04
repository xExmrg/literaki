import functools
import time
from collections import namedtuple
from typing import Dict, Any

# Performance tracking
_perf_stats: Dict[str, Any] = {
    'create_calls': 0,
    'create_time': 0.0,
    'last_create_time': 0.0
}

# Debug mode flag
DEBUG_MODE = False

BOARD_SIZE = 15
MID_INDEX = BOARD_SIZE // 2

# Symbolic constants for board square colors that can trigger letter bonuses
BC_ORANGE = "orange"
BC_GREEN = "green"
BC_YELLOW = "yellow"
BC_BLUE = "blue"

# Define a lightweight square data structure
Square = namedtuple('Square', ['word_mult', 'board_color', 'is_start'])
DEFAULT_SQUARE = Square(word_mult=1, board_color=None, is_start=False)

# Predefined board layout - tuples are more memory efficient than lists
BOARD_LAYOUT = (
    # Row 0: R W 3x W W G W R W G W W 3x W R
    ((0, 0, BC_ORANGE), (0, 2, None, 3), (0, 5, BC_GREEN), (0, 7, BC_ORANGE),
     (0, 9, BC_GREEN), (0, 12, None, 3), (0, 14, BC_ORANGE)),
    
    # Row 1: W W W W G W R W R W G W W W W
    ((1, 4, BC_GREEN), (1, 6, BC_ORANGE), (1, 8, BC_ORANGE), (1, 10, BC_GREEN)),
    
    # Row 2: 3x W W G W 2x W Y W 2x W G W W 3x
    ((2, 0, None, 3), (2, 3, BC_GREEN), (2, 5, None, 2), (2, 7, BC_YELLOW),
     (2, 9, None, 2), (2, 11, BC_GREEN), (2, 14, None, 3)),
    
    # Row 3: W W G W 2x W Y W Y W 2x W G W W
    ((3, 2, BC_GREEN), (3, 4, None, 2), (3, 6, BC_YELLOW), (3, 8, BC_YELLOW),
     (3, 10, None, 2), (3, 12, BC_GREEN)),
    
    # Row 4: W G W 2x W Y W W W Y W 2x W G W
    ((4, 1, BC_GREEN), (4, 3, None, 2), (4, 5, BC_YELLOW), (4, 9, BC_YELLOW),
     (4, 11, None, 2), (4, 13, BC_GREEN)),
    
    # Row 5: G W 2x W Y W W B W W Y W 2x W G
    ((5, 0, BC_GREEN), (5, 2, None, 2), (5, 4, BC_YELLOW), (5, 7, BC_BLUE),
     (5, 10, BC_YELLOW), (5, 12, None, 2), (5, 14, BC_GREEN)),
    
    # Row 6: W R W Y W W B W B W W Y W R W
    ((6, 1, BC_ORANGE), (6, 3, BC_YELLOW), (6, 6, BC_BLUE), (6, 8, BC_BLUE),
     (6, 11, BC_YELLOW), (6, 13, BC_ORANGE)),
    
    # Row 7: R W Y W W B W R W B W W Y W R (center row with start)
    ((7, 0, BC_ORANGE), (7, 2, BC_YELLOW), (7, 5, BC_BLUE), (7, 7, BC_ORANGE, 2, True),
     (7, 9, BC_BLUE), (7, 12, BC_YELLOW), (7, 14, BC_ORANGE)),
    
    # Row 8: W R W Y W W B W B W W Y W R W
    ((8, 1, BC_ORANGE), (8, 3, BC_YELLOW), (8, 6, BC_BLUE), (8, 8, BC_BLUE),
     (8, 11, BC_YELLOW), (8, 13, BC_ORANGE)),
    
    # Row 9: G W 2x W Y W W B W W Y W 2x W G
    ((9, 0, BC_GREEN), (9, 2, None, 2), (9, 4, BC_YELLOW), (9, 7, BC_BLUE),
     (9, 10, BC_YELLOW), (9, 12, None, 2), (9, 14, BC_GREEN)),
    
    # Row 10: W G W 2x W Y W W W Y W 2x W G W
    ((10, 1, BC_GREEN), (10, 3, None, 2), (10, 5, BC_YELLOW), (10, 9, BC_YELLOW),
     (10, 11, None, 2), (10, 13, BC_GREEN)),
    
    # Row 11: W W G W 2x W Y W Y W 2x W G W W
    ((11, 2, BC_GREEN), (11, 4, None, 2), (11, 6, BC_YELLOW), (11, 8, BC_YELLOW),
     (11, 10, None, 2), (11, 12, BC_GREEN)),
    
    # Row 12: 3x W W G W 2x W Y W 2x W G W W 3x
    ((12, 0, None, 3), (12, 3, BC_GREEN), (12, 5, None, 2), (12, 7, BC_YELLOW),
     (12, 9, None, 2), (12, 11, BC_GREEN), (12, 14, None, 3)),
    
    # Row 13: W W W W G W R W R W G W W W W
    ((13, 4, BC_GREEN), (13, 6, BC_ORANGE), (13, 8, BC_ORANGE), (13, 10, BC_GREEN)),
    
    # Row 14: R W 3x W W G W R W G W W 3x W R
    ((14, 0, BC_ORANGE), (14, 2, None, 3), (14, 5, BC_GREEN), (14, 7, BC_ORANGE),
     (14, 9, BC_GREEN), (14, 12, None, 3), (14, 14, BC_ORANGE))
)

@functools.lru_cache(maxsize=None)
def create_literaki_board():
    """Create the Literaki board with memoization and performance tracking."""
    start_time = time.perf_counter() if DEBUG_MODE else 0
    
    # Original board creation logic
    """
    Creates the 15x15 Literaki board with all special squares.
    Each square is a dictionary:
    {'word_mult': int, 'board_color': str/None, 'is_start': bool}
    
    Layout follows the exact pattern:
    R W 3x W W G W R W G W W 3x W R
    W W W W G W R W R W G W W W W
    3x W W G W 2x W Y W 2x W G W W 3x
    W W G W 2x W Y W Y W 2x W G W W
    W G W 2x W Y W W W Y W 2x W G W
    G W 2x W Y W W B W W Y W 2x W G
    W R W Y W W B W B W W Y W R W
    R W Y W W B W R W B W W Y W R
    W R W Y W W B W B W W Y W R W
    G W 2x W Y W W B W W Y W 2x W G
    W G W 2x W Y W W W Y W 2x W G W
    W W G W 2x W Y W Y W 2x W G W W
    3x W W G W 2x W Y W 2x W G W W 3x
    W W W W G W R W R W G W W W W
    R W 3x W W G W R W G W W 3x W R
    """
    # Initialize board with default squares
    board = [[DEFAULT_SQUARE for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    
    # Apply the layout to the board
    for row_data in BOARD_LAYOUT:
        for square_data in row_data:
            if len(square_data) == 3:  # (row, col, color)
                row, col, color = square_data
                board[row][col] = Square(word_mult=1, board_color=color, is_start=False)
            elif len(square_data) == 4:  # (row, col, color, word_mult)
                row, col, color, word_mult = square_data
                board[row][col] = Square(word_mult=word_mult, board_color=color, is_start=False)
            elif len(square_data) == 5:  # (row, col, color, word_mult, is_start)
                row, col, color, word_mult, is_start = square_data
                board[row][col] = Square(word_mult=word_mult, board_color=color, is_start=is_start)

    # Update performance stats if in debug mode
    if DEBUG_MODE:
        elapsed = time.perf_counter() - start_time
        _perf_stats['create_calls'] += 1
        _perf_stats['create_time'] += elapsed
        _perf_stats['last_create_time'] = elapsed
    
    return board

def print_literaki_board(board):
    """Prints a representation of the Literaki board bonuses."""
    print("Literaki Board Layout:")
    for r in range(BOARD_SIZE):
        row_str = []
        for c in range(BOARD_SIZE):
            square = board[r][c]
            label = ""
            if square.is_start:
                label += "*"
            
            if square.word_mult == 3:
                label += "3W"
            elif square.word_mult == 2:
                label += "2W"
            
            if square.board_color:
                if label and (square.word_mult > 1):  # if already 3W or 2W
                    label += f"({square.board_color[0].upper()})"  # e.g. 3W(O)
                else:
                    label += f"L{square.board_color[0].upper()}"  # e.g. LO for Letter Orange

            if not label:
                label = "---"
            row_str.append(f"{label:^7}")  # Pad to make columns align
        print(" ".join(row_str))
    print("\nLegend: * = Start, 3W = Triple Word, 2W = Double Word")
    print("L(ColorInitial) = Letter Bonus for matching Color (e.g., LO = Orange square for letter bonus)")
    print("WM(C) = Word Multiplier and Color for letter bonus (e.g. 3W(O) )")


# --- Main execution ---
if __name__ == "__main__":
    literaki_game_board = create_literaki_board()
    print_literaki_board(literaki_game_board)

    # Example: Accessing a square's properties
    print("\nProperties of square (0,0):", literaki_game_board[0][0])
    print("Properties of square (7,7):", literaki_game_board[MID_INDEX][MID_INDEX])
    print("Properties of square (1,5) (Dark Grey 2X):", literaki_game_board[1][5])
    print("Properties of square (0,3) (Orange Letter Bonus):", literaki_game_board[0][3])
    print("Properties of square (1,1) (Green Letter Bonus):", literaki_game_board[1][1])
    
    # Print performance stats if in debug mode
    if DEBUG_MODE:
        print("\nPerformance Statistics:")
        print(f"Board creation calls: {_perf_stats['create_calls']}")
        print(f"Total creation time: {_perf_stats['create_time']:.6f}s")
        print(f"Last creation time: {_perf_stats['last_create_time']:.6f}s")
        if _perf_stats['create_calls'] > 0:
            avg_time = _perf_stats['create_time'] / _perf_stats['create_calls']
            print(f"Average creation time: {avg_time:.6f}s")
