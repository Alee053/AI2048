import numpy as np
import torch
from .Fast2048 import Fast2048
from .utility import board_to_tensor


class ExpectimaxSearcher:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.transposition_table = {}

    def find_best_move(self, board, search_depth=3):
        best_score = -float('inf')
        best_move = -1

        self.transposition_table = {}

        for move in range(4):
            temp_game = Fast2048()
            temp_game.board = board.copy()

            if not temp_game.is_move_valid(move):
                continue

            merge_score, done,_  = temp_game.move(move)

            score = merge_score + self._chance_node(temp_game.board, search_depth)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move if best_move != -1 else 0

    def _chance_node(self, board, depth):
        board_tuple = tuple(map(tuple, board))
        if (board_tuple, depth) in self.transposition_table:
            return self.transposition_table[(board_tuple, depth)]

        empty_cells = np.argwhere(board == 0)
        if len(empty_cells) == 0:
            return 0

        # Consider placing a '2' (log value 1) in each empty cell
        value_sum_2 = 0
        for r, c in empty_cells:
            temp_board = board.copy()
            temp_board[r, c] = 1  # log2(2) = 1
            value_sum_2 += self._max_node(temp_board, depth - 1)

        # Consider placing a '4' (log value 2) in each empty cell
        value_sum_4 = 0
        for r, c in empty_cells:
            temp_board = board.copy()
            temp_board[r, c] = 2  # log2(4) = 2
            value_sum_4 += self._max_node(temp_board, depth - 1)

        # Calculate the final probability-weighted average value
        expected_value = (0.9 * (value_sum_2 / len(empty_cells))) + \
                         (0.1 * (value_sum_4 / len(empty_cells)))

        # Store the result in the cache before returning
        self.transposition_table[(board_tuple, depth)] = expected_value
        return expected_value

    def _max_node(self, board, depth):
        board_tuple = tuple(map(tuple, board))
        if (board_tuple, depth) in self.transposition_table:
            return self.transposition_table[(board_tuple, depth)]

        if depth == 0:
            obs = board_to_tensor(board)
            with torch.no_grad():
                value = self.model.policy.predict_values(
                    torch.as_tensor(obs[None]).to(self.device)
                )
            return value.item()

        max_value = -float('inf')

        for move in range(4):
            temp_game = Fast2048()
            temp_game.board = board.copy()
            if not temp_game.is_move_valid(move):
                continue
            merge_score, done, moved = temp_game.move(move)

            value = merge_score + self._chance_node(temp_game.board, depth)
            if value > max_value:
                max_value = value

        final_value = max_value if max_value != -float('inf') else 0
        self.transposition_table[(board_tuple, depth)] = final_value
        return final_value
