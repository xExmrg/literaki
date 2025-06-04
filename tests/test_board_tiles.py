import builtins
import types
import pytest

import board
import tiles


def test_create_literaki_board_dimensions_and_start_square():
    board_grid = board.create_literaki_board()
    assert len(board_grid) == board.BOARD_SIZE
    assert all(len(row) == board.BOARD_SIZE for row in board_grid)
    center = board_grid[board.MID_INDEX][board.MID_INDEX]
    assert center.is_start is True
    assert center.word_mult == 2


def test_create_literaki_board_caching():
    first = board.create_literaki_board()
    second = board.create_literaki_board()
    assert first is second


def test_tile_bag_creation_counts():
    bag = tiles.create_tile_bag()
    expected_count = sum(d['count'] for d in tiles.TILE_DEFINITIONS.values())
    assert len(bag) == expected_count


def test_draw_tiles_from_bag_reduces_size():
    bag = tiles.create_tile_bag()
    initial_size = len(bag)
    drawn = tiles.draw_tiles_from_bag(bag, 5)
    assert len(drawn) == 5
    assert len(bag) == initial_size - 5


def test_player_rack_replenish_and_remove():
    bag = tiles.create_tile_bag()
    rack = tiles.PlayerRack(bag, rack_size=7)
    assert len(rack.tiles) == 7
    tile = rack.tiles[0]
    removed = rack.remove_tile(tile)
    assert removed == tile
    rack.replenish_tiles()
    assert len(rack.tiles) == 7

