import pytest
from board import print_literaki_board, DEFAULT_SQUARE
from tiles import PlayerRack, create_tile_bag


def test_print_literaki_board_none():
    with pytest.raises(ValueError):
        print_literaki_board(None)


def test_print_literaki_board_bad_size():
    bad_board = [[DEFAULT_SQUARE]]  # too small
    with pytest.raises(ValueError):
        print_literaki_board(bad_board)


def test_player_rack_replenish_no_bag():
    rack = PlayerRack(None)
    rack.tiles = [{'letter': 'A', 'points': 1, 'color': None}]
    rack.replenish_tiles()  # should do nothing and not error
    assert rack.tiles == [{'letter': 'A', 'points': 1, 'color': None}]


def test_player_rack_add_tile_none():
    bag = create_tile_bag()
    rack = PlayerRack(bag)
    initial_len = len(rack.tiles)
    rack.add_tile(None)
    assert len(rack.tiles) == initial_len
