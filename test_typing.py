from typing import get_type_hints, List, Dict, Any, Optional
import board
import tiles


def test_create_literaki_board_annotations():
    hints = get_type_hints(board.create_literaki_board)
    assert hints.get('return') == board.Board


def test_player_rack_annotations():
    hints_init = get_type_hints(tiles.PlayerRack.__init__)
    assert hints_init.get('tile_bag') == Optional[List[Dict[str, Any]]]
    hints_get_tiles = get_type_hints(tiles.PlayerRack.get_tiles)
    assert hints_get_tiles.get('return') == List[Dict[str, Any]]
