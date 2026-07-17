"""Tests for seed utilities in train.py"""

import random
import numpy as np
import torch
import pytest


def test_seed_from_config():
    """Test seed_from_config extracts seed from config dict."""
    from scripts.train import seed_from_config

    # Default seed is 0 when not specified
    assert seed_from_config({}) == 0
    assert seed_from_config({"seed": 42}) == 42
    assert seed_from_config({"seed": 12345}) == 12345


def test_set_global_seed():
    """Test set_global_seed sets all random generators."""
    from scripts.train import set_global_seed

    # Set seed and verify reproducibility
    set_global_seed(42)

    # Collect some random values
    rand_values = [random.random(), np.random.rand(), torch.randint(0, 1000, (5,)).tolist()]

    # Reset and collect again
    set_global_seed(42)
    rand_values_repeat = [random.random(), np.random.rand(), torch.randint(0, 1000, (5,)).tolist()]

    assert rand_values == rand_values_repeat, "Seeded random values should be reproducible"


def test_resume_settings_defaults_to_fresh_training():
    from scripts.train import resume_settings

    assert resume_settings({}) == (False, None)
    assert resume_settings({"load_model": False}) == (False, None)


def test_resume_settings_requires_checkpoint_when_enabled():
    from scripts.train import resume_settings

    assert resume_settings({"load_model": True, "checkpoint_path": "model.zip"}) == (
        True,
        "model.zip",
    )
    with pytest.raises(ValueError, match="checkpoint_path"):
        resume_settings({"load_model": True})


if __name__ == "__main__":
    test_seed_from_config()
    test_set_global_seed()
    print("All seed utility tests passed!")
