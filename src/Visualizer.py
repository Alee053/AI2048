import os
import pygame
from sb3_contrib import MaskablePPO
import numpy as np
import torch

from .Game2048Env import Game2048Env

# --- TILE COLORS AND DRAW FUNCTION (No changes here) ---
TILE_COLORS = {
    0: (205, 193, 180), 2: (238, 228, 218), 4: (237, 224, 200),
    8: (242, 177, 121), 16: (245, 149, 99), 32: (246, 124, 95),
    64: (246, 94, 59), 128: (237, 207, 114), 256: (237, 204, 97),
    512: (237, 200, 80), 1024: (237, 197, 63), 2048: (237, 194, 46),
    4096: (60, 58, 50), 8192: (60, 58, 50), 16384: (60, 58, 50),
}
TEXT_COLOR_DARK = (119, 110, 101)
TEXT_COLOR_LIGHT = (249, 246, 242)
BG_COLOR = (187, 173, 160)
FONT_COLOR = (255, 255, 255)
STATS_BG_COLOR = (50, 50, 50)


class Visualizer:
    def __init__(self, model_path):
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model_path = model_path

        pygame.init()
        self.screen = pygame.display.set_mode((400, 500))
        pygame.display.set_caption("2048 PPO Agent")

        self.env = Game2048Env()
        self.model = MaskablePPO.load(self.model_path)

    def _get_font(self, tile_value):
        """Returns a dynamically sized font."""
        if tile_value < 100:
            return pygame.font.Font(None, 48)
        elif tile_value < 1000:
            return pygame.font.Font(None, 40)
        else:
            return pygame.font.Font(None, 32)

    def _draw_board(self, board):
        self.screen.fill(BG_COLOR)
        tile_size = 100
        padding = 10

        for r in range(4):
            for c in range(4):
                tile_value = 2 ** board[r][c] if board[r][c] != 0 else 0
                tile_color = TILE_COLORS.get(tile_value, (60, 58, 50))

                rect_x = c * tile_size + padding / 2
                rect_y = r * tile_size + padding / 2
                rect_w = tile_size - padding
                pygame.draw.rect(self.screen, tile_color, (rect_x, rect_y, rect_w, rect_w), border_radius=5)

                if tile_value != 0:
                    font = self._get_font(tile_value)
                    text_color = TEXT_COLOR_DARK if tile_value < 8 else TEXT_COLOR_LIGHT
                    text_surface = font.render(str(tile_value), True, text_color)
                    text_rect = text_surface.get_rect(center=(rect_x + rect_w / 2, rect_y + rect_w / 2))
                    self.screen.blit(text_surface, text_rect)

    def _draw_stats(self, score, max_tile, step, action):
        stats_font = pygame.font.Font(None, 24)
        action_map = ['Up',  'Right', 'Down', 'Left']

        # Draw a background band for the stats
        pygame.draw.rect(self.screen, STATS_BG_COLOR, (0, 400, 400, 100))

        score_text = f"Score: {score}"
        tile_text = f"Max Tile: {2 ** max_tile if max_tile > 0 else 0}"
        step_text = f"Step: {step}"
        action_text = f"Action: {action_map[action]}"

        score_surf = stats_font.render(score_text, True, FONT_COLOR)
        tile_surf = stats_font.render(tile_text, True, FONT_COLOR)
        step_surf = stats_font.render(step_text, True, FONT_COLOR)
        action_surf = stats_font.render(action_text, True, FONT_COLOR)

        self.screen.blit(score_surf, (20, 415))
        self.screen.blit(tile_surf, (20, 455))
        self.screen.blit(step_surf, (250, 415))
        self.screen.blit(action_surf, (250, 455))

    def _draw_game_over(self, score, max_tile):
        overlay = pygame.Surface((400, 500), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))  # Semi-transparent black overlay

        big_font = pygame.font.Font(None, 60)
        small_font = pygame.font.Font(None, 32)

        game_over_surf = big_font.render("Game Over", True, FONT_COLOR)
        score_surf = small_font.render(f"Final Score: {score}", True, FONT_COLOR)
        tile_surf = small_font.render(f"Max Tile: {2 ** max_tile}", True, FONT_COLOR)

        self.screen.blit(overlay, (0, 0))
        self.screen.blit(game_over_surf, game_over_surf.get_rect(center=(200, 180)))
        self.screen.blit(score_surf, score_surf.get_rect(center=(200, 250)))
        self.screen.blit(tile_surf, tile_surf.get_rect(center=(200, 290)))

    def run_visualization(self):
        obs, info = self.env.reset()

        running = True
        step_count = 0

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            action_mask = self.env.action_masks()
            action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)
            last_action = action

            obs, reward, terminated, truncated, info = self.env.step(action)
            step_count += 1

            self._draw_board(self.env.game.board)
            self._draw_stats(self.env.game.score, self.env.game.max_tile, step_count, last_action)

            if terminated or truncated:
                self._draw_game_over(self.env.game.score, self.env.game.max_tile)
                pygame.display.flip()
                pygame.time.wait(5000)
                running = False

            pygame.display.flip()
            pygame.time.wait(150)

        pygame.quit()
