import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from twenty_forty_eight_ai.utils.visualizer import THEME


def test_panel_fits_inside_window():
    """Panel must sit inside the window bounds."""
    x = THEME["side_panel_x"]
    w = THEME["side_panel_width"]
    h = THEME["side_panel_height"]
    win_w, win_h = THEME["window_size"]
    assert x + w == win_w, "panel must touch right edge"
    assert h == THEME["bottom_strip_y"], "panel must touch bottom strip"
    assert x >= 600, "panel must leave room for board"


def test_bottom_strip_fits():
    """Bottom strip must exactly fill remaining height."""
    y = THEME["bottom_strip_y"]
    h = THEME["bottom_strip_height"]
    win_h = THEME["window_size"][1]
    assert y + h == win_h, "bottom strip must reach window bottom"


def test_theme_json_colors_are_consistent():
    """All background colors in theme JSON must match graph bg."""
    theme_path = Path(__file__).parent.parent / "twenty_forty_eight_ai" / "utils" / "visualizer_theme.json"
    data = json.loads(theme_path.read_text())
    colours = data.get("defaults", {}).get("colours", data.get("colours", {}))
    assert colours.get("dark_bg") == "#1A1A1A", f"dark_bg must be #1A1A1A"
    assert colours.get("normal_bg") == "#1A1A1A", f"normal_bg must be #1A1A1A"
    assert colours.get("normal_text") == "#FFFFFF", f"normal_text must be #FFFFFF"
    assert colours.get("filled_bar") == "#F5A623", f"filled_bar must be #F5A623"
    assert colours.get("unfilled_bar") == "#1A1A1A", f"unfilled_bar must be #1A1A1A"