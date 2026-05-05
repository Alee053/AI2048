import pytest
import pygame
from twenty_forty_eight_ai.utils.sparkline import SparklineRenderer


class TestSparklineRenderer:
    def test_render_returns_surface(self):
        renderer = SparklineRenderer(width=200, height=60, color=(255, 255, 255), bg_color=(0, 0, 0))
        renderer.update_data([1.0, 2.0, 3.0, 4.0, 5.0])
        surface = renderer.render()
        assert isinstance(surface, pygame.Surface)
        assert surface.get_width() == 200
        assert surface.get_height() == 60

    def test_empty_data_returns_surface(self):
        renderer = SparklineRenderer(width=200, height=60, color=(255, 255, 255), bg_color=(0, 0, 0))
        surface = renderer.render()
        assert isinstance(surface, pygame.Surface)
        assert surface.get_width() == 200

    def test_update_data_replaces_old(self):
        renderer = SparklineRenderer(width=200, height=60, color=(255, 255, 255), bg_color=(0, 0, 0))
        renderer.update_data([1.0, 2.0])
        assert len(renderer._values) == 2
        renderer.update_data([3.0])
        assert len(renderer._values) == 1
        assert renderer._values[0] == 3.0

    def test_data_truncated_to_max_points(self):
        renderer = SparklineRenderer(width=50, height=60, color=(255, 255, 255), bg_color=(0, 0, 0))
        # width=50, point spacing=2px => max 25 points
        renderer.update_data(list(range(100)))
        assert len(renderer._values) <= 25

    def test_append_adds_value(self):
        renderer = SparklineRenderer(width=200, height=60, color=(255, 255, 255), bg_color=(0, 0, 0))
        renderer.append(1.0)
        renderer.append(2.0)
        assert renderer._values == [1.0, 2.0]

    def test_append_drops_oldest_when_full(self):
        renderer = SparklineRenderer(width=50, height=60, color=(255, 255, 255), bg_color=(0, 0, 0))
        # width=50, point_spacing=2 => max 25 points
        for i in range(30):
            renderer.append(float(i))
        assert len(renderer._values) == 25
        assert renderer._values[0] == 5.0

    def test_constant_value_edge_case(self):
        renderer = SparklineRenderer(width=200, height=60, color=(255, 255, 255), bg_color=(0, 0, 0))
        renderer.update_data([5.0, 5.0, 5.0, 5.0])
        surface = renderer.render()
        assert isinstance(surface, pygame.Surface)
