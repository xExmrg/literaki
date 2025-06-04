import ast
import unittest

class DeadCodeRemovalTests(unittest.TestCase):
    def test_removed_functions(self):
        with open('helper.py', 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        func_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        self.assertNotIn('find_board_in_screenshot', func_names)
        self.assertNotIn('find_anchor_positions', func_names)
        self.assertNotIn('can_place_word_at', func_names)
        self.assertNotIn('score_move', func_names)

if __name__ == '__main__':
    unittest.main()
