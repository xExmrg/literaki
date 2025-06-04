# tiles.py
"""Tile definitions and simple helpers for a Literaki game."""

import random
from board import BC_BLUE, BC_YELLOW, BC_GREEN, BC_ORANGE # Import board color constants

# Tile color categories based on point values as per game_rules.txt
# 1 point: Blue
# 2 points: Yellow
# 3 points: Green
# 5 points or more: Orange
# Blank: Colorless (represented by None or a specific constant if needed for logic)
# Using None for 'color' of blank tiles, as they are "colorless"

# Definition of Polish Literaki letter tiles
# Format: 'Letter': {'points': value, 'count': number_in_game, 'color': tile_color_constant_or_None}
TILE_DEFINITIONS = {
    'A': {'points': 1, 'count': 9, 'color': BC_BLUE},
    'Ą': {'points': 5, 'count': 1, 'color': BC_ORANGE},
    'B': {'points': 3, 'count': 2, 'color': BC_GREEN},
    'C': {'points': 2, 'count': 3, 'color': BC_YELLOW},
    'Ć': {'points': 6, 'count': 1, 'color': BC_ORANGE},
    'D': {'points': 2, 'count': 3, 'color': BC_YELLOW},
    'E': {'points': 1, 'count': 7, 'color': BC_BLUE},
    'Ę': {'points': 5, 'count': 1, 'color': BC_ORANGE},
    'F': {'points': 5, 'count': 1, 'color': BC_ORANGE},
    'G': {'points': 3, 'count': 2, 'color': BC_GREEN},
    'H': {'points': 3, 'count': 2, 'color': BC_GREEN},
    'I': {'points': 1, 'count': 8, 'color': BC_BLUE},
    'J': {'points': 3, 'count': 2, 'color': BC_GREEN},
    'K': {'points': 2, 'count': 3, 'color': BC_YELLOW},
    'L': {'points': 2, 'count': 3, 'color': BC_YELLOW},
    'Ł': {'points': 3, 'count': 2, 'color': BC_GREEN},
    'M': {'points': 2, 'count': 3, 'color': BC_YELLOW},
    'N': {'points': 1, 'count': 5, 'color': BC_BLUE},
    'Ń': {'points': 7, 'count': 1, 'color': BC_ORANGE},
    'O': {'points': 1, 'count': 6, 'color': BC_BLUE},
    'Ó': {'points': 5, 'count': 1, 'color': BC_ORANGE},
    'P': {'points': 2, 'count': 3, 'color': BC_YELLOW},
    'R': {'points': 1, 'count': 4, 'color': BC_BLUE},
    'S': {'points': 1, 'count': 4, 'color': BC_BLUE},
    'Ś': {'points': 5, 'count': 1, 'color': BC_ORANGE},
    'T': {'points': 2, 'count': 3, 'color': BC_YELLOW},
    'U': {'points': 3, 'count': 2, 'color': BC_GREEN},
    'W': {'points': 1, 'count': 4, 'color': BC_BLUE},
    'Y': {'points': 2, 'count': 4, 'color': BC_YELLOW},
    'Z': {'points': 1, 'count': 5, 'color': BC_BLUE}, # Assuming 'Z' is 1 point, count 5 (common in Polish Scrabble)
                                                    # game_rules.txt: Z: 1 (5x)
    'Ź': {'points': 9, 'count': 1, 'color': BC_ORANGE},
    'Ż': {'points': 5, 'count': 1, 'color': BC_ORANGE},
    '_': {'points': 0, 'count': 2, 'color': None}  # Blank tile, represented by '_'
}

def create_tile_bag():
    """Creates the full bag of Literaki tiles based on TILE_DEFINITIONS."""
    tile_bag = [
        {
            'letter': letter,
            'points': details['points'],
            'color': details['color'],
        }
        for letter, details in TILE_DEFINITIONS.items()
        for _ in range(details['count'])
    ]
    random.shuffle(tile_bag)
    return tile_bag

def draw_tiles_from_bag(tile_bag, num_tiles):
    """Draws a specified number of tiles from the bag."""
    if not tile_bag:
        return []
    to_draw = min(num_tiles, len(tile_bag))
    drawn_tiles = tile_bag[-to_draw:]
    del tile_bag[-to_draw:]
    return drawn_tiles

class PlayerRack:
    """Represents a player's rack of tiles."""
    __slots__ = ('rack_size', 'tile_bag', 'tiles')

    def __init__(self, tile_bag, rack_size=7):
        self.rack_size = rack_size
        self.tile_bag = tile_bag  # Reference to the game's tile bag
        self.tiles = []
        if self.tile_bag is not None:
            self.replenish_tiles()

    def replenish_tiles(self):
        """Fills the rack up to its maximum size from the tile bag."""
        needed = self.rack_size - len(self.tiles)
        if needed > 0:
            self.tiles.extend(draw_tiles_from_bag(self.tile_bag, needed))

    def get_tiles(self):
        """Returns the current tiles on the rack."""
        return self.tiles

    def get_rack_str(self):
        """Returns a string representation of the rack."""
        return ", ".join(tile['letter'] for tile in self.tiles)


    def remove_tile(self, letter_char_or_tile_obj):
        """
        Removes a specific letter tile from the rack.
        Can accept either the letter character (str) or the tile object itself.
        If removing a blank tile, it's best to pass the tile object used for the move.
        """
        tile_to_remove = None
        if isinstance(letter_char_or_tile_obj, str):
            letter_char = letter_char_or_tile_obj
            # If it's a blank being represented by its chosen letter,
            # we need to find an actual blank tile '_' on the rack.
            # This simple removal by char might be ambiguous for blanks.
            # For now, assume direct match or actual blank symbol.
            for i, tile in enumerate(self.tiles):
                if tile['letter'] == letter_char:
                    tile_to_remove = self.tiles.pop(i)
                    break
        elif isinstance(letter_char_or_tile_obj, dict) and 'letter' in letter_char_or_tile_obj:
            try:
                self.tiles.remove(letter_char_or_tile_obj) # Remove by object identity
                tile_to_remove = letter_char_or_tile_obj
            except ValueError:
                # Fallback if the exact object isn't found (e.g. a copy was made)
                # This part might need more robust handling for blank tiles specifically
                # if their 'letter' field is changed upon placement.
                # For now, we assume the tile object passed is from the rack.
                pass # Tile not found by object, would have been caught by char search

        return tile_to_remove # Tile not found or already removed

    def add_tile(self, tile):
        """Adds a tile to the rack (e.g., if exchanging or move undone)."""
        if len(self.tiles) < self.rack_size:
            self.tiles.append(tile)

if __name__ == '__main__':
    # Example usage:
    game_tile_bag = create_tile_bag()
    print(f"Total tiles in bag: {len(game_tile_bag)}")
    # print(f"Sample from bag: {random.sample(game_tile_bag, min(len(game_tile_bag), 10))}")


    player1_rack = PlayerRack(game_tile_bag)
    print(f"Player 1 initial rack: {[t['letter'] for t in player1_rack.get_tiles()]}")
    print(f"Tiles remaining in bag: {len(game_tile_bag)}")

    # Simulate playing some tiles
    if len(player1_rack.get_tiles()) >= 1:
        tile_to_play = player1_rack.get_tiles()[0] # Get the tile object
        print(f"Player 1 plays: {tile_to_play}")
        played_tile = player1_rack.remove_tile(tile_to_play) # Remove the tile object
        if played_tile:
            print(f"Successfully played: {played_tile['letter']}")
        print(f"Player 1 rack after playing: {[t['letter'] for t in player1_rack.get_tiles()]}")
        player1_rack.replenish_tiles()
        print(f"Player 1 rack after replenishing: {[t['letter'] for t in player1_rack.get_tiles()]}")
    
    print(f"Tiles remaining in bag after P1 turn: {len(game_tile_bag)}")

    # Check sum of counts
    total_tiles_defined = sum(details['count'] for details in TILE_DEFINITIONS.values())
    print(f"Total tiles defined in TILE_DEFINITIONS: {total_tiles_defined}")
    # As per game_rules.txt, should be 100 tiles.
