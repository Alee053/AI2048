"""Simple sparkline renderer for pygame.

Draws a line chart of numeric data to a pygame Surface.
No dependencies beyond pygame.
"""
from typing import List

import pygame


class SparklineRenderer:
    """Render a simple sparkline (line chart) to a pygame Surface."""

    def __init__(self, width: int, height: int, color: tuple, bg_color: tuple, point_spacing: int = 2):
        """Initialize sparkline renderer.

        Args:
            width: Surface width in pixels.
            height: Surface height in pixels.
            color: RGB tuple for the line color.
            bg_color: RGB tuple for the background color.
            point_spacing: Pixels between data points (default 2).
        """
        self.width = width
        self.height = height
        self.color = color
        self.bg_color = bg_color
        self.point_spacing = point_spacing
        self._values: List[float] = []
        self._max_points = max(2, width // point_spacing)

    def update_data(self, values: List[float]) -> None:
        """Replace the data series. Keeps only the last N points that fit width."""
        self._values = values[-self._max_points:]

    def append(self, value: float) -> None:
        """Append a single value, dropping oldest if over capacity."""
        self._values.append(value)
        if len(self._values) > self._max_points:
            self._values = self._values[-self._max_points:]

    def render(self) -> pygame.Surface:
        """Return a pygame Surface with the drawn sparkline."""
        if not pygame.get_init():
            pygame.init()
        surface = pygame.Surface((self.width, self.height))
        surface.fill(self.bg_color)

        if len(self._values) < 2:
            font = pygame.font.Font(None, 18)
            text = font.render("No data yet", True, (128, 128, 128))
            text_rect = text.get_rect(center=(self.width // 2, self.height // 2))
            surface.blit(text, text_rect)
            return surface

        min_val = min(self._values)
        max_val = max(self._values)
        if max_val == min_val:
            normalized = [0.5] * len(self._values)
        else:
            normalized = [(v - min_val) / (max_val - min_val) for v in self._values]

        padding = 4
        plot_h = self.height - 2 * padding
        points = []
        for i, norm in enumerate(normalized):
            x = i * self.point_spacing + self.point_spacing // 2
            y = self.height - padding - int(norm * plot_h)
            points.append((x, y))

        mid_y = self.height // 2
        pygame.draw.line(surface, (51, 51, 51), (0, mid_y), (self.width, mid_y), 1)

        if len(points) >= 2:
            pygame.draw.lines(surface, self.color, False, points, 2)

        font = pygame.font.Font(None, 18)
        current = self._values[-1]
        if current >= 1000:
            label = f"{current:,.0f}"
        elif current >= 1:
            label = f"{current:.1f}"
        else:
            label = f"{current:.3f}"
        text = font.render(label, True, self.color)
        text_rect = text.get_rect(topright=(self.width - 4, 2))
        surface.blit(text, text_rect)

        return surface
