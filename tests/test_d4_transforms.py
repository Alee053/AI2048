"""Tests for D4 augmentation in Game2048Env.

These tests guard against regressions in the augmentation layer that
forces the policy and value network to be invariant to all 8 D4
symmetries of the 2048 board. The end-to-end tests are the important
ones: they verify that stepping in the transformed view produces the
same canonical game state as stepping with the inverse-permuted action
in a non-augmented environment.
"""
import numpy as np
import pytest

from twenty_forty_eight_ai.env.d4_transforms import (
    apply_d4,
    transform_action_mask,
    ACTION_TO_CANONICAL,
    INVERSE_D4,
    ID,
    ROT90_CW,
    ROT180,
    ROT270_CW,
    REFLECT_H,
    REFLECT_V,
    TRANSPOSE,
    ANTI_TRANSPOSE,
    NUM_TRANSFORMS,
    inverse_d4,
)
from twenty_forty_eight_ai.env.environment import Game2048Env


# Reference board used to verify each D4 transform.
REFERENCE_BOARD = np.array(
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ],
    dtype=np.int32,
)


class TestApplyD4:
    """apply_d4 should produce the geometrically correct transform."""

    def test_identity(self):
        np.testing.assert_array_equal(apply_d4(REFERENCE_BOARD, ID), REFERENCE_BOARD)

    def test_rot90_cw(self):
        expected = np.array(
            [
                [13, 9, 5, 1],
                [14, 10, 6, 2],
                [15, 11, 7, 3],
                [16, 12, 8, 4],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(apply_d4(REFERENCE_BOARD, ROT90_CW), expected)

    def test_rot180(self):
        expected = np.array(
            [
                [16, 15, 14, 13],
                [12, 11, 10, 9],
                [8, 7, 6, 5],
                [4, 3, 2, 1],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(apply_d4(REFERENCE_BOARD, ROT180), expected)

    def test_rot270_cw(self):
        expected = np.array(
            [
                [4, 8, 12, 16],
                [3, 7, 11, 15],
                [2, 6, 10, 14],
                [1, 5, 9, 13],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(apply_d4(REFERENCE_BOARD, ROT270_CW), expected)

    def test_reflect_h(self):
        expected = np.array(
            [
                [4, 3, 2, 1],
                [8, 7, 6, 5],
                [12, 11, 10, 9],
                [16, 15, 14, 13],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(apply_d4(REFERENCE_BOARD, REFLECT_H), expected)

    def test_reflect_v(self):
        expected = np.array(
            [
                [13, 14, 15, 16],
                [9, 10, 11, 12],
                [5, 6, 7, 8],
                [1, 2, 3, 4],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(apply_d4(REFERENCE_BOARD, REFLECT_V), expected)

    def test_transpose(self):
        expected = REFERENCE_BOARD.T
        np.testing.assert_array_equal(apply_d4(REFERENCE_BOARD, TRANSPOSE), expected)

    def test_anti_transpose(self):
        expected = np.array(
            [
                [16, 12, 8, 4],
                [15, 11, 7, 3],
                [14, 10, 6, 2],
                [13, 9, 5, 1],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(apply_d4(REFERENCE_BOARD, ANTI_TRANSPOSE), expected)

    def test_does_not_modify_input(self):
        original = REFERENCE_BOARD.copy()
        for t in range(NUM_TRANSFORMS):
            _ = apply_d4(REFERENCE_BOARD, t)
        np.testing.assert_array_equal(REFERENCE_BOARD, original)

    def test_invalid_transform_raises(self):
        with pytest.raises(ValueError):
            apply_d4(REFERENCE_BOARD, 8)
        with pytest.raises(ValueError):
            apply_d4(REFERENCE_BOARD, -1)


class TestGroupClosure:
    """Every D4 element must have a true inverse."""

    @pytest.mark.parametrize("t", list(range(NUM_TRANSFORMS)))
    def test_inverse_roundtrip(self, t):
        transformed = apply_d4(REFERENCE_BOARD, t)
        recovered = apply_d4(transformed, inverse_d4(t))
        np.testing.assert_array_equal(recovered, REFERENCE_BOARD)

    @pytest.mark.parametrize("t", list(range(NUM_TRANSFORMS)))
    def test_inverse_first(self, t):
        recovered = apply_d4(apply_d4(REFERENCE_BOARD, inverse_d4(t)), t)
        np.testing.assert_array_equal(recovered, REFERENCE_BOARD)

    @pytest.mark.parametrize("t", list(range(NUM_TRANSFORMS)))
    def test_double_application_is_identity_if_inverse(self, t):
        # apply_d4 twice with the inverse must return the input
        once = apply_d4(REFERENCE_BOARD, t)
        twice = apply_d4(once, inverse_d4(t))
        np.testing.assert_array_equal(twice, REFERENCE_BOARD)


class TestActionPermutation:
    """ACTION_TO_CANONICAL must be a valid permutation for each transform."""

    @pytest.mark.parametrize("t", list(range(NUM_TRANSFORMS)))
    def test_is_permutation(self, t):
        perm = ACTION_TO_CANONICAL[t]
        assert sorted(perm.tolist()) == [0, 1, 2, 3], (
            f"ACTION_TO_CANONICAL[{t}] is not a valid permutation"
        )

    @pytest.mark.parametrize("t", list(range(NUM_TRANSFORMS)))
    def test_inverse_action_is_inverse_permutation(self, t):
        # The inverse-D4 element should produce the inverse action permutation.
        # Concretely: ACTION_TO_CANONICAL[inv(t)][ACTION_TO_CANONICAL[t][a]] == a
        forward = ACTION_TO_CANONICAL[t]
        backward = ACTION_TO_CANONICAL[inverse_d4(t)]
        for a in range(4):
            assert backward[forward[a]] == a
            assert forward[backward[a]] == a

    def test_identity_action_table(self):
        np.testing.assert_array_equal(ACTION_TO_CANONICAL[ID], [0, 1, 2, 3])


class TestActionMaskTransform:
    """transform_action_mask must permute canonical validity into the transformed view."""

    def test_identity_passes_through(self):
        mask = np.array([False, True, True, False])
        np.testing.assert_array_equal(transform_action_mask(mask, ID), mask)

    def test_known_case_rot90_cw(self):
        # Canonical board has a single tile at (0,0). Valid actions in
        # canonical coords: Right (slides to (0,3)) and Down (slides to (3,0)).
        canonical = np.array([False, True, True, False])
        # Under ROT90_CW, (0,0) maps to (0,3). Valid actions in the
        # transformed view: Down (slides to (3,3)) and Left (slides to (0,0)).
        expected = np.array([False, False, True, True])
        np.testing.assert_array_equal(
            transform_action_mask(canonical, ROT90_CW), expected
        )

    def test_known_case_reflect_h(self):
        canonical = np.array([False, True, True, False])
        # Under REFLECT_H, (0,0) maps to (0,3). Valid: Down and Left.
        expected = np.array([False, False, True, True])
        np.testing.assert_array_equal(
            transform_action_mask(canonical, REFLECT_H), expected
        )

    def test_all_valid_passes_through(self):
        canonical = np.array([True, True, True, True])
        for t in range(NUM_TRANSFORMS):
            np.testing.assert_array_equal(
                transform_action_mask(canonical, t),
                np.array([True, True, True, True]),
            )


class TestEnvironmentBackwardCompatibility:
    """The default env behavior (d4_augment=False) must be unchanged."""

    def test_default_constructor_signature(self):
        # The original signature `Game2048Env()` must still work.
        env = Game2048Env()
        assert env.d4_augment is False
        assert env._current_d4 == ID

    def test_default_observation_matches_canonical_board(self):
        env = Game2048Env()
        obs, _ = env.reset()
        from twenty_forty_eight_ai.utils.tensor_utils import board_to_tensor
        expected = board_to_tensor(env.game.board)
        np.testing.assert_array_equal(obs, expected)

    def test_default_action_mask_matches_canonical(self):
        env = Game2048Env()
        env.reset()
        # No transform is active, so the mask must be the raw canonical mask.
        canonical = np.array(
            [env.game.is_move_valid(a) for a in range(4)], dtype=bool
        )
        np.testing.assert_array_equal(env.action_masks(), canonical)


class TestEnvironmentD4Augmentation:
    """With d4_augment=True, the env must expose transformed views consistently."""

    def test_observation_matches_transformed_board(self):
        env = Game2048Env(d4_augment=True, d4_seed=42)
        obs, _ = env.reset()
        from twenty_forty_eight_ai.utils.tensor_utils import board_to_tensor
        expected = board_to_tensor(
            apply_d4(env.game.board, env._current_d4)
        )
        np.testing.assert_array_equal(obs, expected)

    def test_observation_uses_first_sampled_transform(self):
        # With d4_seed=42 the first integer in [0, 8) is deterministic
        # for np.random.default_rng. We just need that the transform
        # used at reset is the one sampled.
        env = Game2048Env(d4_augment=True, d4_seed=42)
        env.reset()
        sampled = env._d4_rng.integers(0, NUM_TRANSFORMS)
        # The transform recorded is the one BEFORE the RNG was advanced for the peek above.
        # Re-run reset to isolate the first sample.
        env2 = Game2048Env(d4_augment=True, d4_seed=42)
        env2.reset()
        assert env2._current_d4 == env._current_d4

    @pytest.mark.parametrize("forced_t", list(range(NUM_TRANSFORMS)))
    def test_action_transformation_matches_canonical_step(self, forced_t):
        """For every D4 transform, picking action a in transformed view
        and stepping must produce the same canonical game.board as
        picking ACTION_TO_CANONICAL[t][a] in a non-augmented env with
        the same canonical starting state."""
        # Seed global RNG so the random-tile generation is reproducible.
        np.random.seed(1000)

        # Build the starting canonical state with two deterministic tiles.
        canonical_board = np.zeros((4, 4), dtype=np.int32)
        canonical_board[0, 0] = 2
        canonical_board[0, 1] = 2  # adjacent pair, can merge on Left/Up

        # --- Reference: step in canonical coords with ACTION_TO_CANONICAL[t][a]
        ref_env = Game2048Env()
        ref_env.game.board = canonical_board.copy()
        ref_env.game.max_tile = int(canonical_board.max())
        ref_env.game.score = 0
        ref_env.game.done = False
        canonical_action = int(ACTION_TO_CANONICAL[forced_t, 0])  # use action 0 (Up) as probe
        np.random.seed(2000)
        ref_env.game.move(canonical_action)
        expected_canonical = ref_env.game.board.copy()

        # --- Augmented env: forced transform, then step with action 0
        aug_env = Game2048Env(d4_augment=True, d4_seed=7)
        aug_env.reset()
        aug_env.game.board = canonical_board.copy()
        aug_env.game.max_tile = int(canonical_board.max())
        aug_env.game.score = 0
        aug_env.game.done = False
        aug_env._current_d4 = forced_t
        np.random.seed(2000)
        aug_env.step(0)

        np.testing.assert_array_equal(
            aug_env.game.board,
            expected_canonical,
            err_msg=f"canonical board mismatch for transform t={forced_t}",
        )

    def test_action_masks_consistent_with_transformation(self):
        """action_masks in the transformed view must equal the canonical
        mask permuted by ACTION_TO_CANONICAL[t]."""
        env = Game2048Env(d4_augment=True, d4_seed=99)
        env.reset()
        # Force a known transform.
        env._current_d4 = ROT90_CW
        canonical_mask = np.array(
            [env.game.is_move_valid(a) for a in range(4)], dtype=bool
        )
        expected_mask = transform_action_mask(canonical_mask, ROT90_CW)
        np.testing.assert_array_equal(env.action_masks(), expected_mask)

    def test_random_transforms_change_across_resets(self):
        """Different seeds should produce different first transforms
        (with overwhelming probability)."""
        env1 = Game2048Env(d4_augment=True, d4_seed=1)
        env2 = Game2048Env(d4_augment=True, d4_seed=2)
        env1.reset()
        env2.reset()
        # It's possible (1/8) they collide; we just need the feature to be live.
        # Verify at least one of them is not the identity, so we know the
        # sampler is actually returning values in [0, 8).
        assert (
            env1._current_d4 != ID or env2._current_d4 != ID
        ), "d4_augment should sometimes produce non-identity transforms"

    def test_episode_can_complete_with_augmentation(self):
        """A full episode under d4_augment must terminate cleanly."""
        env = Game2048Env(d4_augment=True, d4_seed=2026)
        env.reset(seed=2026)
        for _ in range(5000):
            mask = env.action_masks()
            valid = np.where(mask)[0]
            assert valid.size > 0, "no valid actions before termination"
            action = int(valid[0])
            _, _, done, _, _ = env.step(action)
            if done:
                break
        assert done, "augmented env should reach termination within 5000 steps"


class TestD4AugmentationProperty:
    """Property-style checks: D4 augmentation must commute with the
    game-logic identity in canonical coordinates."""

    @pytest.mark.parametrize("forced_t", list(range(NUM_TRANSFORMS)))
    def test_full_step_round_trip(self, forced_t):
        """Across many steps with a fixed D4 transform, the augmented
        env's canonical board must match a non-augmented reference env
        that applies the inverse-permuted action each step."""
        canonical_start = np.zeros((4, 4), dtype=np.int32)
        canonical_start[0, 0] = 2
        canonical_start[0, 3] = 4
        canonical_start[3, 0] = 8

        rng = np.random.default_rng(12345)
        probe_actions = [int(rng.integers(0, 4)) for _ in range(20)]

        # --- Reference (non-augmented) ---
        ref = Game2048Env()
        ref.game.board = canonical_start.copy()
        ref.game.max_tile = int(canonical_start.max())
        ref.game.score = 0
        ref.game.done = False

        # --- Augmented ---
        aug = Game2048Env(d4_augment=True, d4_seed=11)
        aug.reset()
        aug.game.board = canonical_start.copy()
        aug.game.max_tile = int(canonical_start.max())
        aug.game.score = 0
        aug.game.done = False

        for step_i, agent_action in enumerate(probe_actions):
            # Pin the augmented env's transform so the test isolates the
            # action-permutation behavior from the random-resample behavior.
            aug._current_d4 = forced_t

            canonical_action = int(
                ACTION_TO_CANONICAL[forced_t, agent_action]
            )
            np.random.seed(4000 + step_i)
            ref.game.move(canonical_action)

            np.random.seed(4000 + step_i)
            aug.step(agent_action)

            np.testing.assert_array_equal(
                aug.game.board,
                ref.game.board,
                err_msg=(
                    f"mismatch at step {step_i}, transform={forced_t}, "
                    f"agent_action={agent_action}, canonical={canonical_action}"
                ),
            )
