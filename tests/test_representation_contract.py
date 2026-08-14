import numpy as np
import pytest
import torch

from twenty_forty_eight_ai.agent.architecture import CustomCNN
from twenty_forty_eight_ai.constants import MAX_EXPONENT
from twenty_forty_eight_ai.env.environment import Game2048Env
from twenty_forty_eight_ai.env.game import DOWN, LEFT, RIGHT, UP, Fast2048
from twenty_forty_eight_ai.utils.searcher import ExpectimaxSearcher, _impl
from twenty_forty_eight_ai.utils.tensor_utils import board_to_tensor

def _board_with_exponent(exponent: int) -> np.ndarray:
    board = np.zeros((4, 4), dtype=np.int32)
    board[0, 0] = exponent
    return board


def _cpp_board(board: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in board)


def test_board_to_tensor_preserves_valid_exponents_and_shapes():
    values = np.arange(MAX_EXPONENT + 1, dtype=np.int32)
    boards = np.zeros((MAX_EXPONENT + 1, 4, 4), dtype=np.int32)
    boards[np.arange(MAX_EXPONENT + 1), 0, 0] = values

    batch_tensor = board_to_tensor(boards)
    assert batch_tensor.shape == (MAX_EXPONENT + 1, 1, 4, 4)
    assert batch_tensor.dtype == np.int64
    np.testing.assert_array_equal(batch_tensor[:, 0, 0, 0], values)

    single_board = np.arange(MAX_EXPONENT + 1, dtype=np.int32).reshape(4, 4)
    single_tensor = board_to_tensor(single_board)
    assert single_tensor.shape == (1, 4, 4)
    assert single_tensor.dtype == np.int64
    np.testing.assert_array_equal(single_tensor[0], single_board)


def test_board_to_tensor_keeps_all_valid_embedding_indices_distinct():
    board = np.arange(MAX_EXPONENT + 1, dtype=np.int32).reshape(4, 4)

    encoded = board_to_tensor(board)[0]

    assert np.unique(encoded).tolist() == list(range(MAX_EXPONENT + 1))


@pytest.mark.parametrize("invalid_exponent", [-1, MAX_EXPONENT + 1])
def test_board_to_tensor_rejects_exponents_outside_contract(invalid_exponent):
    with pytest.raises(ValueError, match="0.*15"):
        board_to_tensor(_board_with_exponent(invalid_exponent))


@pytest.mark.parametrize("shape", [(16,), (4, 4, 1), (1, 4, 4, 1)])
def test_board_to_tensor_rejects_invalid_shapes(shape):
    with pytest.raises(ValueError, match="shape"):
        board_to_tensor(np.zeros(shape, dtype=np.int32))


def test_python_contract_constant_and_environment_observation_space():
    env = Game2048Env()

    assert MAX_EXPONENT == 15
    assert env.observation_space.low.min() == 0
    assert env.observation_space.high.max() == MAX_EXPONENT


def test_custom_cnn_accepts_maximum_valid_exponent():
    env = Game2048Env()
    model = CustomCNN(env.observation_space, features_dim=32)
    observation = torch.as_tensor(board_to_tensor(_board_with_exponent(MAX_EXPONENT))[None])

    assert model.embedding.num_embeddings == 17
    with torch.no_grad():
        features = model(observation)

    assert features.shape == (1, 32)


def test_python_game_accepts_exponent_15_without_overflow():
    game = Fast2048()
    game.board = _board_with_exponent(MAX_EXPONENT)
    game.max_tile = MAX_EXPONENT

    assert isinstance(game.is_move_valid(0), (bool, np.bool_))


def test_python_game_rejects_15_plus_15_without_partial_mutation():
    game = Fast2048()
    game.board.fill(0)
    game.board[0, :2] = MAX_EXPONENT
    game.score = 123
    board_before = game.board.copy()

    with pytest.raises(ValueError, match="15.*16|maximum.*15"):
        game.move(3)

    np.testing.assert_array_equal(game.board, board_before)
    assert game.score == 123


@pytest.mark.parametrize(
    ("direction", "positions"),
    [
        (LEFT, ((0, 0), (0, 1))),
        (RIGHT, ((0, 2), (0, 3))),
        (UP, ((0, 0), (1, 0))),
        (DOWN, ((2, 0), (3, 0))),
    ],
)
def test_python_game_rejects_max_merge_in_every_direction(direction, positions):
    game = Fast2048()
    game.board.fill(0)
    for row, col in positions:
        game.board[row, col] = MAX_EXPONENT
    board_before = game.board.copy()
    score_before = game.score

    with pytest.raises(ValueError, match="15.*16|maximum.*15"):
        game.move(direction)

    np.testing.assert_array_equal(game.board, board_before)
    assert game.score == score_before


@pytest.mark.parametrize(
    ("direction", "positions"),
    [
        (LEFT, ((0, 0), (0, 1))),
        (RIGHT, ((0, 2), (0, 3))),
        (UP, ((0, 0), (1, 0))),
        (DOWN, ((2, 0), (3, 0))),
    ],
)
def test_python_and_cpp_reject_max_merge_as_invalid(direction, positions):
    board = np.zeros((4, 4), dtype=np.int32)
    for row, col in positions:
        board[row, col] = MAX_EXPONENT

    python_game = Fast2048()
    python_game.board = board.copy()
    cpp_game = _impl.Fast2048()
    cpp_game.set_board(_cpp_board(board))

    assert python_game.is_move_valid(direction) is False
    assert cpp_game.is_move_valid(direction) is False


@pytest.mark.parametrize(
    ("direction", "positions"),
    [
        (LEFT, ((0, 0), (0, 1))),
        (RIGHT, ((0, 2), (0, 3))),
        (UP, ((0, 0), (1, 0))),
        (DOWN, ((2, 0), (3, 0))),
    ],
)
def test_cpp_overflow_move_is_a_noop(direction, positions):
    board = np.zeros((4, 4), dtype=np.int32)
    for row, col in positions:
        board[row, col] = MAX_EXPONENT

    game = _impl.Fast2048()
    game.set_board(_cpp_board(board))
    board_before = game.get_board()
    score_before = game.score

    assert game.move(direction) == (0, False, False)
    assert game.get_board() == board_before
    assert game.score == score_before


def test_python_and_cpp_move_validity_matches_for_valid_boards():
    rng = np.random.default_rng(2048)
    boards = [rng.integers(0, MAX_EXPONENT + 1, size=(4, 4), dtype=np.int32)
              for _ in range(32)]
    boards.append(np.array([
        [15, 15, 0, 0],
        [0, 0, 15, 15],
        [15, 0, 0, 15],
        [0, 15, 15, 0],
    ], dtype=np.int32))

    python_game = Fast2048()
    cpp_game = _impl.Fast2048()
    for board in boards:
        python_game.board = board.copy()
        cpp_game.set_board(_cpp_board(board))
        for direction in range(4):
            assert bool(python_game.is_move_valid(direction)) == cpp_game.is_move_valid(
                direction
            )


def test_python_and_cpp_normal_moves_match_without_random_spawn():
    board = np.array([
        [1, 1, 0, 0],
        [2, 0, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int32)
    python_game = Fast2048()
    python_game.board = board.copy()
    python_game.generate_random = lambda: None
    cpp_game = _impl.Fast2048()
    cpp_game.set_board(_cpp_board(board))

    python_merge_score, _, python_moved = python_game.move(LEFT)
    cpp_merge_score, _, cpp_moved = cpp_game.move(LEFT)

    assert (python_merge_score, python_moved) == (cpp_merge_score, cpp_moved)
    assert cpp_game.score == python_game.score == python_merge_score

    cpp_board = np.asarray(cpp_game.get_board())
    differences = np.argwhere(cpp_board != python_game.board)
    assert len(differences) == 1
    row, col = differences[0]
    assert python_game.board[row, col] == 0
    assert cpp_board[row, col] in (1, 2)
    cpp_board[row, col] = 0
    np.testing.assert_array_equal(cpp_board, python_game.board)


def test_python_game_rejects_preexisting_invalid_exponent_before_lut_access():
    game = Fast2048()
    game.board = _board_with_exponent(MAX_EXPONENT + 1)

    with pytest.raises(ValueError, match="0.*15"):
        game.is_move_valid(LEFT)


def test_environment_masks_max_tile_overflow_actions():
    env = Game2048Env()
    env.game.board.fill(0)
    env.game.board[0, :2] = MAX_EXPONENT

    action_mask = env.action_masks()

    assert not action_mask[LEFT]
    assert not action_mask[RIGHT]


def test_python_cpp_contract_accepts_15_and_rejects_16():
    board15 = _cpp_board(_board_with_exponent(MAX_EXPONENT))
    board16 = _cpp_board(_board_with_exponent(MAX_EXPONENT + 1))
    game = _impl.Fast2048()

    game.set_board(board15)
    assert game.get_board()[0][0] == MAX_EXPONENT

    with pytest.raises(ValueError, match="0.*15"):
        game.set_board(board16)

    packed = _impl.BoardEncoder.pack(board15)
    assert _impl.BoardEncoder.unpack(packed)[0][0] == MAX_EXPONENT

    with pytest.raises(ValueError, match="0.*15"):
        _impl.BoardEncoder.pack(board16)


def test_expectimax_callback_preserves_exponents_to_embedding():
    env = Game2048Env()
    model = CustomCNN(env.observation_space, features_dim=32)
    embedding_inputs = []
    hook = model.embedding.register_forward_pre_hook(
        lambda _module, args: embedding_inputs.append(args[0].detach().clone())
    )
    callback_batches = 0

    def evaluate_batch(boards):
        nonlocal callback_batches
        callback_batches += 1
        batch_array = np.asarray(boards)
        batch_tensor = board_to_tensor(batch_array)
        np.testing.assert_array_equal(batch_tensor[:, 0], batch_array)
        with torch.no_grad():
            values = model(torch.as_tensor(batch_tensor))
        np.testing.assert_array_equal(
            embedding_inputs[-1].numpy(),
            batch_tensor[:, 0].reshape(batch_tensor.shape[0], 16),
        )
        return [float(value) for value in values[:, 0].tolist()]

    searcher = ExpectimaxSearcher(
        target_batch_size=256,
        use_transposition_table=False,
    )
    board = np.zeros((4, 4), dtype=np.int32)
    board[0, :2] = 1

    try:
        searcher.find_best_move(board, depth=1, batch_eval_func=evaluate_batch)
    finally:
        hook.remove()

    assert callback_batches > 0
