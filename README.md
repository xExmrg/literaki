# Literaki Helper

Python tools for working with the Polish Scrabble variant *Literaki*. The project provides:

- **`board.py`** – creation of a 15×15 board with bonus squares.
- **`tiles.py`** – definitions of all letter tiles and helper classes like `PlayerRack`.
- **`helper.py`** – the main automation script that reads the board via OCR and suggests plays.

Detailed setup instructions are available in [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md).

## Quick start

```bash
pip install -r requirements.txt      # install dependencies
python test_setup.py                 # validate everything works
python helper.py                     # launch the helper
```

## Examples

Create and display a board:

```python
from board import create_literaki_board, print_literaki_board
board = create_literaki_board()
print_literaki_board(board)
```

Work with tiles:

```python
from tiles import create_tile_bag, PlayerRack
bag = create_tile_bag()
rack = PlayerRack(bag)
print(rack.get_rack_str())
```

Run the automation helper:

```bash
python helper.py
```

The helper opens Chrome, waits for you to join a game on kurnik.pl and displays OCR results and suggested moves in a Pygame window.
