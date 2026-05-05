"""Tests for BoardEncoder pack/unpack/canonicalize."""
import numpy as np
import pytest


def _load_encoder():
    """Load BoardEncoder from the C++ extension."""
    import importlib.util
    import sys
    so_path = "twenty_forty_eight_ai/utils/_searcher_cpp.cpython-312-x86_64-linux-gnu.so"
    spec = importlib.util.spec_from_file_location("searcher", so_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_searcher_test"] = module
    spec.loader.exec_module(module)
    return module.BoardEncoder


class TestBoardEncoder:
    @pytest.fixture(scope="class")
    def encoder(self):
        return _load_encoder()

    def test_pack_unpack_roundtrip(self, encoder):
        """pack -> unpack should be identity."""
        board = (
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (8, 9, 10, 11),
            (12, 13, 14, 15),
        )
        packed = encoder.pack(board)
        unpacked = encoder.unpack(packed)
        assert list(map(list, unpacked)) == [list(row) for row in board]

    def test_canonicalize_identity(self, encoder):
        """Canonicalizing the same board twice gives the same result."""
        board = (
            (1, 0, 0, 2),
            (0, 3, 0, 0),
            (0, 0, 1, 0),
            (2, 0, 0, 3),
        )
        c1 = encoder.canonicalize_board(board)
        c2 = encoder.canonicalize_board(board)
        assert c1 == c2

    def test_canonicalize_symmetry(self, encoder):
        """Rotating a board should not change its canonical form."""
        board = (
            (1, 2, 3, 4),
            (5, 6, 7, 8),
            (9, 10, 11, 12),
            (13, 14, 15, 0),
        )
        c = encoder.canonicalize_board(board)

        # Rotate 90 degrees clockwise
        rotated = (
            (13, 9, 5, 1),
            (14, 10, 6, 2),
            (15, 11, 7, 3),
            (0, 12, 8, 4),
        )
        c_rot = encoder.canonicalize_board(rotated)
        assert c == c_rot

    def test_canonicalize_reflection(self, encoder):
        """Mirroring a board should not change its canonical form."""
        board = (
            (1, 0, 2, 3),
            (0, 4, 0, 0),
            (5, 0, 6, 0),
            (0, 7, 0, 8),
        )
        c = encoder.canonicalize_board(board)

        # Horizontal reflection
        reflected = (
            (3, 2, 0, 1),
            (0, 0, 4, 0),
            (0, 6, 0, 5),
            (8, 0, 7, 0),
        )
        c_ref = encoder.canonicalize_board(reflected)
        assert c == c_ref

    def test_canonicalize_distinct_boards(self, encoder):
        """Different boards should have different canonical forms (high probability)."""
        board1 = (
            (1, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        )
        board2 = (
            (2, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        )
        c1 = encoder.canonicalize_board(board1)
        c2 = encoder.canonicalize_board(board2)
        assert c1 != c2
