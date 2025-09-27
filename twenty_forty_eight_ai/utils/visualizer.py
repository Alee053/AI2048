import os
import pygame
import numpy as np
import torch
from typing import List
from sb3_contrib import MaskablePPO
from .searcher import ExpectimaxSearcher
from ..utils.tensor_utils import board_to_tensor
from ..env.environment import Game2048Env


# --- Configuration & Theming ---
THEME = {
    "window_size": (400, 500),
    "bg_color": (187, 173, 160),
    "tile_colors": {
        0: (205, 193, 180), 2: (238, 228, 218), 4: (237, 224, 200),
        8: (242, 177, 121), 16: (245, 149, 99), 32: (246, 124, 95),
        64: (246, 94, 59), 128: (237, 207, 114), 256: (237, 204, 97),
        512: (237, 200, 80), 1024: (237, 197, 63), 2048: (237, 194, 46),
        "default": (60, 58, 50)
    },
    "text_color_dark": (119, 110, 101),
    "text_color_light": (249, 246, 242),
    "stats_bg_color": (50, 50, 50),
    "font_color": (255, 255, 255),
    "font_sizes": {"small": 24, "medium": 32, "large": 40, "xlarge": 48},
}


class Visualizer:
    """A Pygame-based visualizer to watch a trained PPO agent play 2048,
    optionally enhanced with an Expectimax searcher."""

    def __init__(self, model_path: str, use_expectimax: bool = True, search_depth: int = 3):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.use_expectimax = use_expectimax
        self.search_depth = search_depth

        # --- Setup Pygame and display ---
        pygame.init()
        self.screen = pygame.display.set_mode(THEME["window_size"])
        pygame.display.set_caption("2048 AI Agent")
        self.clock = pygame.time.Clock()

        # --- Load Fonts Once for Efficiency ---
        self.fonts = {
            size_name: pygame.font.Font(None, size_val)
            for size_name, size_val in THEME["font_sizes"].items()
        }

        # --- Setup RL components ---
        self.env = Game2048Env()
        self.model = MaskablePPO.load(model_path)
        self.searcher = ExpectimaxSearcher() if use_expectimax else None

        mode = f"PPO + C++ Expectimax (depth={search_depth})" if use_expectimax else "Raw PPO Model"
        print(f"Visualizer running with: {mode}")

    def _evaluate_batch(self, boards_list: List[np.ndarray]) -> List[float]:
        """Callback for the C++ searcher to evaluate a batch of boards using the PPO critic."""
        if not boards_list:
            return []
        batch_tensor = board_to_tensor(np.array(boards_list))
        with torch.no_grad():
            values = self.model.policy.predict_values(
                torch.as_tensor(batch_tensor).to(self.model.device)
            )
        return values.cpu().numpy().flatten().tolist()

    def _get_font_for_tile(self, value: int) -> pygame.font.Font:
        """Selects the correct pre-loaded font based on tile value."""
        if value < 100: return self.fonts["xlarge"]
        if value < 1000: return self.fonts["large"]
        return self.fonts["medium"]

    def _draw_board(self, board: np.ndarray):
        """Draws the 4x4 game grid and tiles."""
        self.screen.fill(THEME["bg_color"])
        tile_size, padding = 100, 10
        for r, c in np.ndindex(board.shape):
            tile_exponent = board[r, c]
            tile_value = 2 ** tile_exponent if tile_exponent != 0 else 0

            color = THEME["tile_colors"].get(tile_value, THEME["tile_colors"]["default"])
            rect = pygame.Rect(c * tile_size + padding / 2, r * tile_size + padding / 2, tile_size - padding,
                               tile_size - padding)
            pygame.draw.rect(self.screen, color, rect, border_radius=5)

            if tile_value != 0:
                font = self._get_font_for_tile(tile_value)
                text_color = THEME["text_color_dark"] if tile_value < 8 else THEME["text_color_light"]
                text_surf = font.render(str(tile_value), True, text_color)
                text_rect = text_surf.get_rect(center=rect.center)
                self.screen.blit(text_surf, text_rect)

    def _draw_stats(self, score: int, max_tile_exp: int, step: int, action: int):
        """Draws the statistics panel below the game board."""
        stats_rect = pygame.Rect(0, 400, THEME["window_size"][0], 100)
        pygame.draw.rect(self.screen, THEME["stats_bg_color"], stats_rect)

        action_map = {0: 'Up', 1: 'Right', 2: 'Down', 3: 'Left', -1: 'N/A'}
        stats = {
            f"Score: {score}": (20, 415),
            f"Max Tile: {2**max_tile_exp if max_tile_exp > 0 else 0}": (20, 455),
            f"Step: {step}": (250, 415),
            f"Action: {action_map.get(action, 'N/A')}": (250, 455)
        }
        for text, pos in stats.items():
            surf = self.fonts["small"].render(text, True, THEME["font_color"])
            self.screen.blit(surf, pos)

    def _get_next_action(self, obs: np.ndarray) -> int:
        """Determines the next action using either PPO or PPO+Expectimax."""
        if self.searcher:
            # Use the C++ searcher, providing the _evaluate_batch method as a callback
            return self.searcher.find_best_move(
                self.env.game.board,
                self.search_depth,
                self._evaluate_batch
            )
        else:
            # Use the raw PPO model
            action_mask = self.env.action_masks()
            action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)
            return int(action)

    def _draw_game_over(self, score, max_tile):
        overlay = pygame.Surface((400, 500), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))

        big_font = pygame.font.Font(None, 60)
        small_font = pygame.font.Font(None, 32)

        game_over_surf = big_font.render("Game Over", True, THEME["font_color"])
        score_surf = small_font.render(f"Final Score: {score}", True, THEME["font_color"])
        tile_surf = small_font.render(f"Max Tile: {2 ** max_tile}", True, THEME["font_color"])

        self.screen.blit(overlay, (0, 0))
        self.screen.blit(game_over_surf, game_over_surf.get_rect(center=(200, 180)))
        self.screen.blit(score_surf, score_surf.get_rect(center=(200, 250)))
        self.screen.blit(tile_surf, tile_surf.get_rect(center=(200, 290)))

    def run(self):
        """Runs the main visualization game loop."""
        obs, _ = self.env.reset()
        running = True
        terminated = False
        step_count = 0
        last_action = -1

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if not terminated:
                # --- Get Action ---
                action = self._get_next_action(obs)
                last_action = action

                # --- Step Environment ---
                obs, _, terminated, _, info = self.env.step(action)
                step_count += 1

            # --- Drawing ---
            self._draw_board(self.env.game.board)
            self._draw_stats(self.env.game.score, self.env.game.max_tile, step_count, last_action)
            if terminated:
                self._draw_game_over(info['score'], info['max_tile'])

            pygame.display.flip()
            self.clock.tick(10)

        pygame.quit()