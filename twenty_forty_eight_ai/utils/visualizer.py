import os
import pygame
import numpy as np
import torch
import threading
from typing import List, Optional
from sb3_contrib import MaskablePPO
from ..utils.searcher import ExpectimaxSearcher
from ..utils.tensor_utils import board_to_tensor
from ..env.environment import Game2048Env

# Config
THEME = {
    "window_size": (620, 550),  # Width increased for side panel
    "side_panel_width": 220,
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
    """Pygame visualizer for PPO agent with threaded Expectimax support."""

    def __init__(self, model_path: str, use_expectimax: bool = True, search_depth: int = 3, show_stats: bool = True):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.use_expectimax = use_expectimax
        self.search_depth = search_depth
        self.show_stats = show_stats
        self.move_history: List = []
        self.cumulative = {
            'total_moves': 0,
            'total_time_ms': 0.0,
            'total_nodes': 0,
            'total_tt_lookups': 0,
            'total_tt_hits': 0,
        }

        # Setup Pygame
        import pygame_gui
        pygame.init()
        self.screen = pygame.display.set_mode(THEME["window_size"])
        pygame.display.set_caption("2048 AI Agent")
        self.clock = pygame.time.Clock()
        self.manager = pygame_gui.UIManager(THEME["window_size"])

        # Load fonts
        self.fonts = {
            size_name: pygame.font.Font(None, size_val)
            for size_name, size_val in THEME["font_sizes"].items()
        }

        # Setup RL
        self.env = Game2048Env()
        self.model = MaskablePPO.load(model_path)

        # Init searcher if needed
        self.searcher = ExpectimaxSearcher() if use_expectimax else None

        # Threading state
        self.is_thinking = False
        self.next_action: Optional[int] = None
        self.search_thread: Optional[threading.Thread] = None

        mode = f"PPO + Expectimax (depth={search_depth})" if use_expectimax else "Raw PPO Model"
        print(f"Visualizer running with: {mode}")

    def _evaluate_batch(self, boards_list: List[np.ndarray]) -> List[float]:
        """Callback for C++ searcher to evaluate batch of boards."""
        if not boards_list:
            return []

        # Convert boards (N, 4, 4) to tensor (N, 1, 4, 4)
        batch_array = np.array(boards_list)
        batch_tensor = board_to_tensor(batch_array)

        with torch.no_grad():
            values = self.model.policy.predict_values(
                torch.as_tensor(batch_tensor).to(self.model.device)
            )
        return values.cpu().numpy().flatten().tolist()

    def _get_font_for_tile(self, value: int) -> pygame.font.Font:
        if value < 100: return self.fonts["xlarge"]
        if value < 1000: return self.fonts["large"]
        return self.fonts["medium"]

    def _draw_board(self, board: np.ndarray, offset_x: int = 0):
        """Draws the 4x4 game grid at the given x offset."""
        self.screen.fill(THEME["bg_color"])
        tile_size, padding = 100, 10

        for r in range(4):
            for c in range(4):
                tile_value = int(board[r, c])

                if tile_value > 0: tile_value = 2**tile_value

                color = THEME["tile_colors"].get(tile_value, THEME["tile_colors"]["default"])

                rect = pygame.Rect(
                    offset_x + c * tile_size + padding / 2,
                    r * tile_size + padding / 2,
                    tile_size - padding,
                    tile_size - padding
                )
                pygame.draw.rect(self.screen, color, rect, border_radius=5)

                if tile_value != 0:
                    font = self._get_font_for_tile(tile_value)
                    # Adjust text color for contrast
                    text_color = THEME["text_color_dark"] if tile_value < 8 else THEME["text_color_light"]
                    text_surf = font.render(str(tile_value), True, text_color)
                    text_rect = text_surf.get_rect(center=rect.center)
                    self.screen.blit(text_surf, text_rect)

    def _draw_stats(self, score: int, max_tile: int, step: int, action: int):
        """Draws the statistics panel."""
        stats_rect = pygame.Rect(0, 400, THEME["window_size"][0], 150)
        pygame.draw.rect(self.screen, THEME["stats_bg_color"], stats_rect)

        action_map = {0: 'Up', 1: 'Right', 2: 'Down', 3: 'Left', -1: '...'}
        status_text = "Thinking..." if self.is_thinking else "Ready"

        stats = {
            f"Score: {score}": (20, 415),
            f"Max: {max_tile}": (20, 455),
            f"Step: {step}": (200, 415),
            f"Act: {action_map.get(action, 'N/A')}": (200, 455),
            f"Status: {status_text}": (20, 495)
        }

        for text, pos in stats.items():
            surf = self.fonts["small"].render(text, True, THEME["font_color"])
            self.screen.blit(surf, pos)

    def _draw_stats_panel(self, last_stats: dict):
        """Draws the right-side stats panel."""
        panel_x = 400  # right of board
        panel_w = THEME["side_panel_width"]
        panel_h = 400  # full height of board area
        pygame.draw.rect(self.screen, THEME["stats_bg_color"], (panel_x, 0, panel_w, panel_h))

        font_small = self.fonts["small"]
        action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']

        # Current move section
        y = 10
        move_n = self.cumulative['total_moves']
        surf = font_small.render(f"Move #{move_n}", True, THEME["font_color"])
        self.screen.blit(surf, (panel_x + 10, y))
        y += 30

        # think_ms
        surf = font_small.render(f"think: {last_stats.get('think_ms', 0):.1f}ms", True, THEME["font_color"])
        self.screen.blit(surf, (panel_x + 10, y))
        y += 25

        # nodes
        surf = font_small.render(f"nodes: {last_stats.get('nodes_visited', 0):,}", True, THEME["font_color"])
        self.screen.blit(surf, (panel_x + 10, y))
        y += 25

        # batches
        surf = font_small.render(f"batches: {last_stats.get('batches_eval', 0)}", True, THEME["font_color"])
        self.screen.blit(surf, (panel_x + 10, y))
        y += 25

        # best move
        best = last_stats.get('best_move', 0)
        best_score = last_stats.get('move_scores', [0]*4)[best]
        surf = font_small.render(f"best: {action_names[best]} ({best_score:.2f})", True, THEME["font_color"])
        self.screen.blit(surf, (panel_x + 10, y))
        y += 30

        # all move scores
        surf = font_small.render("scores:", True, THEME["font_color"])
        self.screen.blit(surf, (panel_x + 10, y))
        y += 22
        scores = last_stats.get('move_scores', [-1e9]*4)
        score_labels = [
            f"U:{scores[0]:.1f}" if scores[0] > -1e8 else "U:--",
            f"R:{scores[1]:.1f}" if scores[1] > -1e8 else "R:--",
            f"D:{scores[2]:.1f}" if scores[2] > -1e8 else "D:--",
            f"L:{scores[3]:.1f}" if scores[3] > -1e8 else "L:--",
        ]
        surf = font_small.render(f"  {score_labels[0]}  {score_labels[1]}", True, THEME["font_color"])
        self.screen.blit(surf, (panel_x + 10, y))
        y += 20
        surf = font_small.render(f"  {score_labels[2]}  {score_labels[3]}", True, THEME["font_color"])
        self.screen.blit(surf, (panel_x + 10, y))
        y += 35

        # Move history header
        surf = font_small.render("History:", True, THEME["font_color"])
        self.screen.blit(surf, (panel_x + 10, y))
        y += 22

        # Move history list (newest first, show last 8)
        history = self.move_history[-8:] if len(self.move_history) > 8 else self.move_history
        for i, mh in enumerate(reversed(history)):
            mh_best = action_names[mh.get('best_move', 0)]
            mh_ms = mh.get('think_ms', 0)
            mh_nodes = mh.get('nodes_visited', 0)
            label = f"#{len(self.move_history)-i} [{mh_ms:.0f}ms, {mh_nodes:,}, {mh_best}]"
            surf = font_small.render(label, True, (180, 180, 180))
            self.screen.blit(surf, (panel_x + 10, y))
            y += 18
            if y > 380: break  # don't overflow

    def _draw_cumulative_bar(self, score: int, max_tile: int):
        """Draws the cumulative stats bar below the board area."""
        bar_y = 400
        bar_h = 150
        pygame.draw.rect(self.screen, THEME["stats_bg_color"], (0, bar_y, 620, bar_h))

        font_small = self.fonts["small"]
        c = self.cumulative
        n = c['total_moves']
        tt_rate = (c['total_tt_hits'] / c['total_tt_lookups'] * 100) if c['total_tt_lookups'] > 0 else 0

        line1 = f"Moves: {n} | {c['total_time_ms']:.0f}ms | {c['total_nodes']:,} nodes | {tt_rate:.0f}% tt"
        line2 = f"Score: {score} | Max: {max_tile}"

        surf = font_small.render(line1, True, THEME["font_color"])
        self.screen.blit(surf, (20, bar_y + 10))
        surf = font_small.render(line2, True, THEME["font_color"])
        self.screen.blit(surf, (20, bar_y + 40))

    def _search_worker(self):
        """Background thread for search."""
        try:
            # Copy board for thread safety
            current_board = self.env.game.board.copy()

            stats = self.searcher.find_best_move(
                current_board,
                self.search_depth,
                self._evaluate_batch
            )
            self.next_action = int(stats['best_move'])
            self.last_stats = stats
        except Exception as e:
            print(f"Search thread error: {e}")
            self.next_action = 0  # Fallback
        finally:
            self.is_thinking = False

    def run(self):
        obs, _ = self.env.reset()
        running = True
        terminated = False
        step_count = 0
        last_action = -1

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.manager.process_events(event)

            if not terminated:
                if self.use_expectimax:
                    # Threaded Expectimax
                    if not self.is_thinking and self.next_action is None:
                        # Start thinking
                        self.is_thinking = True
                        self.search_thread = threading.Thread(target=self._search_worker)
                        self.search_thread.start()

                    elif not self.is_thinking and self.next_action is not None:
                        # Execute move
                        action = self.next_action
                        self.next_action = None

                        last_action = action
                        obs, _, terminated, _, info = self.env.step(action)
                        step_count += 1

                        # Update cumulative stats
                        if hasattr(self, 'last_stats'):
                            stats = self.last_stats
                            self.move_history.append(stats)
                            self.cumulative['total_moves'] += 1
                            self.cumulative['total_time_ms'] += stats.get('think_ms', 0)
                            self.cumulative['total_nodes'] += stats.get('nodes_visited', 0)
                            self.cumulative['total_tt_lookups'] += stats.get('tt_lookups', 0)
                            self.cumulative['total_tt_hits'] += stats.get('tt_hits', 0)

                else:
                    # Standard PPO
                    # Delay for visibility
                    pygame.time.delay(100)
                    action_mask = self.env.action_masks()
                    action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)

                    last_action = int(action)
                    obs, _, terminated, _, info = self.env.step(last_action)
                    step_count += 1

            self._draw_board(self.env.game.board, offset_x=0)

            if self.show_stats:
                last_s = self.move_history[-1] if self.move_history else {
                    'think_ms': 0, 'nodes_visited': 0, 'batches_eval': 0,
                    'best_move': 0, 'move_scores': [-1e9]*4
                }
                self._draw_stats_panel(last_s)
                self._draw_cumulative_bar(self.env.game.score, self.env.game.max_tile)
            else:
                self._draw_stats(self.env.game.score, self.env.game.max_tile, step_count, last_action)

            if terminated:
                final_score = self.env.game.score
                final_max = self.env.game.max_tile
                self._draw_game_over(final_score, final_max)

            time_delta = self.clock.tick(60) / 1000.0
            self.manager.update(time_delta)
            self.manager.draw_ui(self.screen)
            pygame.display.flip()

        if self.search_thread and self.search_thread.is_alive():
            self.search_thread.join()
        pygame.quit()

    def _draw_game_over(self, score, max_tile):
        overlay = pygame.Surface(THEME["window_size"], pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))

        big_font = pygame.font.Font(None, 60)
        small_font = pygame.font.Font(None, 32)

        game_over_surf = big_font.render("Game Over", True, THEME["font_color"])
        score_surf = small_font.render(f"Final Score: {score}", True, THEME["font_color"])
        tile_surf = small_font.render(f"Max Tile: {max_tile}", True, THEME["font_color"])

        self.screen.blit(overlay, (0, 0))
        self.screen.blit(game_over_surf, game_over_surf.get_rect(center=(200, 200)))
        self.screen.blit(score_surf, score_surf.get_rect(center=(200, 270)))
        self.screen.blit(tile_surf, tile_surf.get_rect(center=(200, 310)))