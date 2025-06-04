import os
import pytest

from dictionary_handler import load_dictionary
from tiles import PlayerRack, create_tile_bag


def test_load_dictionary_invalid_type():
    with pytest.raises(ValueError):
        load_dictionary(None)


def test_load_dictionary_missing_file(tmp_path):
    missing = tmp_path / "nope.txt"
    with pytest.raises(FileNotFoundError):
        load_dictionary(str(missing))


def test_player_rack_invalid_size():
    with pytest.raises(ValueError):
        PlayerRack([], rack_size=0)


def test_player_rack_invalid_bag_type():
    with pytest.raises(TypeError):
        PlayerRack("notalist")


def test_player_rack_add_tile_invalid():
    rack = PlayerRack(create_tile_bag())
    with pytest.raises(TypeError):
        rack.add_tile("A")


def test_player_rack_remove_tile_invalid_arg():
    rack = PlayerRack(create_tile_bag())
    with pytest.raises(TypeError):
        rack.remove_tile(42)
