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
        self._pygame_gui = pygame_gui
        pygame.init()
        self.screen = pygame.display.set_mode(THEME["window_size"])
        pygame.display.set_caption("2048 AI Agent")
        self.clock = pygame.time.Clock()
        theme_path = os.path.join(os.path.dirname(__file__), "visualizer_theme.json")
        self.manager = pygame_gui.UIManager(THEME["window_size"], theme_path=theme_path)

        # Load fonts
        self.fonts = {
            size_name: pygame.font.Font(None, size_val)
            for size_name, size_val in THEME["font_sizes"].items()
        }

        # Create stats panel UILabels
        panel_x = 400
        self.stats_labels = {}
        label_defs = [
            ("move_num", f"Move #0", (panel_x + 10, 10)),
            ("think_ms", "think: 0.0ms", (panel_x + 10, 40)),
            ("nodes", "nodes: 0", (panel_x + 10, 70)),
            ("batches", "batches: 0", (panel_x + 10, 100)),
            ("best_move", "best: -- (0.00)", (panel_x + 10, 130)),
            ("scores_label", "scores:", (panel_x + 10, 160)),
            ("scores_row1", "  U:--  R:--", (panel_x + 10, 182)),
            ("scores_row2", "  D:--  L:--", (panel_x + 10, 204)),
        ]
        for name, text, pos in label_defs:
            rect = pygame.Rect(pos, (200, 22))
            self.stats_labels[name] = pygame_gui.elements.UILabel(
                relative_rect=rect, text=text, manager=self.manager
            )

        # Create buttons
        button_y = 240
        button_w = 95
        button_h = 40
        button_gap = 10

        self.new_game_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((panel_x + 10, button_y), (button_w, button_h)),
            text='New Game',
            manager=self.manager
        )

        self.pause_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((panel_x + 10 + button_w + button_gap, button_y), (button_w, button_h)),
            text='Pause',
            manager=self.manager
        )

        self.paused = False

        # Create scrollable history area
        history_y = 290
        history_h = 110
        self.history_scroll_area = pygame_gui.elements.UIScrollingContainer(
            relative_rect=pygame.Rect((panel_x + 10, history_y), (200, history_h)),
            manager=self.manager
        )
        self.history_inner = self.history_scroll_area.get_container()
        self.history_inner.set_scrollable_area_dimensions((180, history_h * 3))
        self.history_labels = []

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

    def _reset_game(self):
        """Reset environment and clear stats."""
        obs, _ = self.env.reset()
        self.move_history.clear()
        self._update_history_display()
        self.cumulative = {
            'total_moves': 0,
            'total_time_ms': 0.0,
            'total_nodes': 0,
            'total_tt_lookups': 0,
            'total_tt_hits': 0,
        }
        self.next_action = None
        self.is_thinking = False
        if self.search_thread and self.search_thread.is_alive():
            self.search_thread.join()
        self.search_thread = None

    def _toggle_pause(self):
        """Toggle pause state."""
        self.paused = not self.paused
        if self.pause_button:
            self.pause_button.set_text('Resume' if self.paused else 'Pause')

    def _update_history_display(self):
        """Rebuild history list items."""
        if not hasattr(self, 'history_scroll_area'):
            return

        # Kill existing labels
        for label in self.history_labels:
            label.kill()
        self.history_labels.clear()

        action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        history = list(reversed(self.move_history[-20:]))
        item_height = 18
        for i, mh in enumerate(history):
            mh_best = action_names[mh.get('best_move', 0)]
            mh_ms = mh.get('think_ms', 0)
            mh_nodes = mh.get('nodes_visited', 0)
            global_idx = len(self.move_history) - len(history) + i + 1
            label_text = f"#{global_idx} [{mh_ms:.0f}ms, {mh_nodes:,}, {mh_best}]"

            label = pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect((0, i * item_height), (180, item_height)),
                text=label_text,
                manager=self.manager,
                container=self.history_inner
            )
            self.history_labels.append(label)

        # Update scroll area height based on item count
        total_height = len(history) * item_height
        self.history_inner.set_scrollable_area_dimensions((180, max(total_height, history_h)))

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

                if event.type == self._pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.new_game_button:
                        self._reset_game()
                    elif event.ui_element == self.pause_button:
                        self._toggle_pause()

            if not terminated and not self.paused:
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
                            self._update_history_display()
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
                self._draw_cumulative_bar(self.env.game.score, self.env.game.max_tile)

            if terminated:
                final_score = self.env.game.score
                final_max = self.env.game.max_tile
                self._draw_game_over(final_score, final_max)

            # Update stats labels
            last_s = self.move_history[-1] if self.move_history else {
                'think_ms': 0, 'nodes_visited': 0, 'batches_eval': 0,
                'best_move': 0, 'move_scores': [-1e9]*4
            }
            n = self.cumulative['total_moves']
            self.stats_labels["move_num"].set_text(f"Move #{n}")
            self.stats_labels["think_ms"].set_text(f"think: {last_s.get('think_ms', 0):.1f}ms")
            self.stats_labels["nodes"].set_text(f"nodes: {last_s.get('nodes_visited', 0):,}")
            self.stats_labels["batches"].set_text(f"batches: {last_s.get('batches_eval', 0)}")

            scores = last_s.get('move_scores', [-1e9]*4)
            best = last_s.get('best_move', 0)
            action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
            best_score = scores[best] if best < len(scores) else 0.0
            self.stats_labels["best_move"].set_text(f"best: {action_names[best]} ({best_score:.2f})")

            score_labels = [
                f"U:{scores[0]:.1f}" if scores[0] > -1e8 else "U:--",
                f"R:{scores[1]:.1f}" if scores[1] > -1e8 else "R:--",
                f"D:{scores[2]:.1f}" if scores[2] > -1e8 else "D:--",
                f"L:{scores[3]:.1f}" if scores[3] > -1e8 else "L:--",
            ]
            self.stats_labels["scores_row1"].set_text(f"  {score_labels[0]}  {score_labels[1]}")
            self.stats_labels["scores_row2"].set_text(f"  {score_labels[2]}  {score_labels[3]}")

            time_delta = self.clock.tick(60) / 1000.0
            if hasattr(self, 'history_scroll_area'):
                self.history_scroll_area.update(time_delta)
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