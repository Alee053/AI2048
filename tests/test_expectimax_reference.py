"""Independent exact expectimax regression for the C++ searcher."""
from __future__ import annotations

import numpy as np

from twenty_forty_eight_ai.utils.searcher import ExpectimaxSearcher


def _reward(merge_score: int) -> float:
    return float(np.log2(np.float32(merge_score))) if merge_score > 0 else 0.0


def _float32(value: float) -> float:
    return float(np.float32(value))


def _move(board: tuple[tuple[int, ...], ...], action: int):
    def merge_line(line: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
        tiles = [tile for tile in line if tile]
        merged = []
        merge_score = 0
        index = 0
        while index < len(tiles):
            if index + 1 < len(tiles) and tiles[index] == tiles[index + 1]:
                exponent = tiles[index] + 1
                merged.append(exponent)
                merge_score += 1 << exponent
                index += 2
            else:
                merged.append(tiles[index])
                index += 1
        return tuple(merged + [0] * (4 - len(merged))), merge_score

    result = [list(row) for row in board]
    merge_score = 0
    if action in (1, 3):
        for row in range(4):
            line = tuple(result[row])
            if action == 1:
                line = line[::-1]
            line, score = merge_line(line)
            if action == 1:
                line = line[::-1]
            result[row] = list(line)
            merge_score += score
    else:
        for column in range(4):
            line = tuple(result[row][column] for row in range(4))
            if action == 2:
                line = line[::-1]
            line, score = merge_line(line)
            if action == 2:
                line = line[::-1]
            for row in range(4):
                result[row][column] = line[row]
            merge_score += score

    child = tuple(tuple(row) for row in result)
    return child, merge_score, child != board


def _leaf_value(board: tuple[tuple[int, ...], ...]) -> float:
    return _float32(sum(sum(row) for row in board))


def _max_value(board: tuple[tuple[int, ...], ...], depth: int, leaf_value=_leaf_value) -> float:
    if depth == 0:
        return leaf_value(board)

    values = []
    for action in range(4):
        child, merge_score, moved = _move(board, action)
        if moved:
            values.append(_float32(_reward(merge_score) + _chance_value(child, depth, leaf_value)))
    return max(values, default=0.0)


def _chance_value(board: tuple[tuple[int, ...], ...], depth: int, leaf_value=_leaf_value) -> float:
    empty_cells = [(row, col) for row in range(4) for col in range(4) if board[row][col] == 0]
    if not empty_cells:
        return _max_value(board, depth - 1, leaf_value)

    total = 0.0
    for row, col in empty_cells:
        for exponent, probability in ((1, 0.9), (2, 0.1)):
            child = [list(values) for values in board]
            child[row][col] = exponent
            total = _float32(
                total + _float32(probability) * _max_value(tuple(tuple(values) for values in child), depth - 1, leaf_value)
            )
    return _float32(total / len(empty_cells))


def _root_scores(board: tuple[tuple[int, ...], ...], depth: int, leaf_value=_leaf_value) -> list[float]:
    scores = [float("-inf")] * 4
    for action in range(4):
        child, merge_score, moved = _move(board, action)
        if moved:
            scores[action] = _float32(_reward(merge_score) + _chance_value(child, depth, leaf_value))
    return scores


def test_search_modes_match_independent_exhaustive_expectimax_without_alpha_beta():
    board = np.array([
        [1, 1, 2, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int32)
    expected_scores = _root_scores(tuple(tuple(row) for row in board), depth=2)
    expected_action = min(
        action for action, score in enumerate(expected_scores) if score == max(expected_scores)
    )

    def evaluator(boards):
        return [_leaf_value(tuple(tuple(row) for row in board)) for board in boards]

    fresh_result = ExpectimaxSearcher(target_batch_size=32768).find_best_move(board, 2, evaluator)

    warmed_searcher = ExpectimaxSearcher(target_batch_size=32768)
    warmed_searcher.find_best_move(board, 2, evaluator)
    warmed_result = warmed_searcher.find_best_move(board, 2, evaluator)

    disabled_result = ExpectimaxSearcher(
        target_batch_size=32768,
        use_transposition_table=False,
    ).find_best_move(board, 2, evaluator)

    for result in (fresh_result, warmed_result, disabled_result):
        assert result["alpha_beta_cuts"] == 0
        assert result["best_move"] == expected_action
        assert result["move_scores"] == expected_scores


def test_high_leaf_values_match_independent_exhaustive_expectimax_without_alpha_beta():
    board = np.array([
        [1, 1, 2, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int32)

    def high_leaf_value(values):
        return _float32(1_100_000_000 + 1_000_000 * sum(sum(row) for row in values))

    expected_scores = _root_scores(tuple(tuple(row) for row in board), depth=2, leaf_value=high_leaf_value)
    assert high_leaf_value(board) > 1_000_000_000
    expected_action = min(
        action for action, score in enumerate(expected_scores) if score == max(expected_scores)
    )
    result = ExpectimaxSearcher(target_batch_size=32768).find_best_move(
        board,
        2,
        lambda boards: [high_leaf_value(values) for values in boards],
    )

    assert result["alpha_beta_cuts"] == 0
    assert result["best_move"] == expected_action
    assert result["move_scores"] == expected_scores


def test_leaf_values_below_the_old_finite_sentinel_are_preserved():
    board = np.array([
        [1, 1, 2, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int32)

    def very_negative_leaf_value(_board):
        return _float32(-2_000_000_000)

    expected_scores = _root_scores(
        tuple(tuple(row) for row in board),
        depth=2,
        leaf_value=very_negative_leaf_value,
    )
    expected_action = min(
        action for action, score in enumerate(expected_scores) if score == max(expected_scores)
    )
    result = ExpectimaxSearcher(
        target_batch_size=32768,
        use_transposition_table=False,
    ).find_best_move(
        board,
        2,
        lambda boards: [very_negative_leaf_value(values) for values in boards],
    )

    assert result["has_legal_move"] is True
    assert result["search_complete"] is True
    assert result["best_move"] == expected_action
    np.testing.assert_allclose(result["move_scores"], expected_scores, rtol=0.0, atol=512.0)
    finite_scores = [score for score in result["move_scores"] if np.isfinite(score)]
    assert finite_scores
    assert all(score < -1_000_000_000 for score in finite_scores)
