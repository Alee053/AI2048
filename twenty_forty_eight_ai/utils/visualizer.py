import os
import queue
import threading
from typing import Optional, List

import numpy as np
import pygame
import pygame_gui
import torch
from sb3_contrib import MaskablePPO

from ..env.environment import Game2048Env
from ..utils.searcher import ExpectimaxSearcher
from ..utils.tensor_utils import board_to_tensor

# Config
THEME = {
    "window_size": (620, 480),
    "side_panel_width": 220,
    "bg_color": (187, 173, 160),
    "tile_colors": {
        0: (205, 193, 180),
        2: (238, 228, 218),
        4: (237, 224, 200),
        8: (242, 177, 121),
        16: (245, 149, 99),
        32: (246, 124, 95),
        64: (246, 94, 59),
        128: (237, 207, 114),
        256: (237, 204, 97),
        512: (237, 200, 80),
        1024: (237, 197, 63),
        2048: (237, 194, 46),
        "default": (60, 58, 50),
    },
    "text_color_dark": (119, 110, 101),
    "text_color_light": (249, 246, 242),
    "stats_bg_color": (50, 50, 50),
    "font_color": (255, 255, 255),
    "font_sizes": {"small": 24, "medium": 32, "large": 40, "xlarge": 48},
    "accent": (245, 166, 35),
    "progress_filled": "#F5A623",
    "progress_unfilled": "#333333",
}


class Visualizer:
    """Pygame visualizer for PPO agent with event-driven async expectimax."""

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

        # Custom event types
        self.CUSTOM_SEARCH_REQUEST = pygame.event.custom_type()
        self.SEARCH_COMPLETE = pygame.event.custom_type()

        # Threading state
        self._search_event = threading.Event()
        self._result_queue = queue.Queue(maxsize=1)
        self._worker_running = True
        self._searching = False
        self._game_id = 0
        self._current_result = None
        self._current_board_for_search = None

        # Setup Pygame
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

        # Create right side panel
        panel_x = 400
        panel_w = 220
        panel_h = 480
        self.side_panel = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect((panel_x, 0), (panel_w, panel_h)),
            manager=self.manager
        )

        # ===== Stats block (amber top border 3px) =====
        self.stats_top_border = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect((0, 0), (panel_w, 3)),
            manager=self.manager,
            container=self.side_panel
        )
        self.stats_top_border.set_fill_colour(pygame.Color("#F5A623"))

        # Stats labels
        y_offset = 10
        self.move_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_offset), (200, 22)),
            text="Move #0",
            manager=self.manager,
            container=self.side_panel
        )
        y_offset += 26
        self.think_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_offset), (200, 22)),
            text="think: 0.0ms",
            manager=self.manager,
            container=self.side_panel
        )
        y_offset += 26
        self.nodes_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_offset), (200, 22)),
            text="nodes: 0",
            manager=self.manager,
            container=self.side_panel
        )
        y_offset += 26
        self.batches_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_offset), (200, 22)),
            text="batches: 0",
            manager=self.manager,
            container=self.side_panel
        )
        y_offset += 26
        self.best_move_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_offset), (200, 22)),
            text="best: -- (0.00)",
            manager=self.manager,
            container=self.side_panel
        )

        # ===== Action scores block =====
        y_offset += 36
        scores_header = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_offset), (200, 20)),
            text="Action Scores",
            manager=self.manager,
            container=self.side_panel
        )
        y_offset += 22

        self.score_bars: List[pygame_gui.elements.UIProgressBar] = []
        self.score_labels: List[pygame_gui.elements.UILabel] = []
        action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']

        for i, name in enumerate(action_names):
            row_y = y_offset + i * 24
            name_label = pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect((10, row_y), (40, 20)),
                text=name,
                manager=self.manager,
                container=self.side_panel
            )
            bar = pygame_gui.elements.UIProgressBar(
                relative_rect=pygame.Rect((55, row_y), (100, 16)),
                manager=self.manager,
                container=self.side_panel
            )
            score_lbl = pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect((160, row_y), (50, 20)),
                text="--",
                manager=self.manager,
                container=self.side_panel
            )
            self.score_bars.append(bar)
            self.score_labels.append(score_lbl)

        y_offset += 24 * 4 + 10

        # ===== Buttons =====
        button_y = y_offset
        button_w = 95
        button_h = 40
        button_gap = 10

        self.new_game_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, button_y), (button_w, button_h)),
            text='New Game',
            manager=self.manager,
            container=self.side_panel
        )

        self.pause_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10 + button_w + button_gap, button_y), (button_w, button_h)),
            text='Pause',
            manager=self.manager,
            container=self.side_panel
        )

        self.paused = False

        # ===== Scrollable history via UITextBox =====
        history_y = button_y + button_h + 15
        self.history_text = pygame_gui.elements.UITextBox(
            relative_rect=pygame.Rect((10, history_y), (200, 130)),
            html_text="No moves yet.",
            manager=self.manager,
            container=self.side_panel,
            wrap_to_height=True
        )

        # ===== Cumulative bar at bottom =====
        self.cum_panel = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect((0, 420), (620, 60)),
            manager=self.manager
        )
        self.cum_top_border = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect((0, 420), (620, 3)),
            manager=self.manager
        )
        self.cum_top_border.set_fill_colour(pygame.Color("#F5A623"))
        self.cum_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 430), (600, 40)),
            text="Moves: 0 | 0ms | 0 nodes | 0% tt | Score: 0",
            manager=self.manager,
            container=self.cum_panel
        )

        # Setup RL
        self.env = Game2048Env()
        self.model = MaskablePPO.load(model_path)

        # Init searcher if needed
        self.searcher = ExpectimaxSearcher() if use_expectimax else None

        # Start worker thread
        self.search_thread = threading.Thread(target=self._search_worker, daemon=True)
        self.search_thread.start()

        mode = f"PPO + Expectimax (depth={search_depth})" if use_expectimax else "Raw PPO Model"
        print(f"Visualizer running with: {mode}")

    def _evaluate_batch(self, boards_list: List[np.ndarray]) -> List[float]:
        """Callback for C++ searcher to evaluate batch of boards."""
        if not boards_list:
            return []

        batch_array = np.array(boards_list)
        batch_tensor = board_to_tensor(batch_array)

        with torch.no_grad():
            values = self.model.policy.predict_values(
                torch.as_tensor(batch_tensor).to(self.model.device)
            )
        return values.cpu().numpy().flatten().tolist()

    def _get_font_for_tile(self, value: int) -> pygame.font.Font:
        if value < 100:
            return self.fonts["xlarge"]
        if value < 1000:
            return self.fonts["large"]
        return self.fonts["medium"]

    def _draw_board(self, board: np.ndarray, offset_x: int = 0):
        """Draws the 4x4 game grid at the given x offset."""
        self.screen.fill(THEME["bg_color"])
        tile_size, padding = 100, 10

        for r in range(4):
            for c in range(4):
                tile_value = int(board[r, c])

                if tile_value > 0:
                    tile_value = 2 ** tile_value

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
                    text_color = THEME["text_color_dark"] if tile_value < 8 else THEME["text_color_light"]
                    text_surf = font.render(str(tile_value), True, text_color)
                    text_rect = text_surf.get_rect(center=rect.center)
                    self.screen.blit(text_surf, text_rect)

    def _search_worker(self):
        """Worker thread: wait for request, run search, post result."""
        while self._worker_running:
            self._search_event.wait()
            if not self._worker_running:
                break
            self._search_event.clear()

            board_copy = self._current_board_for_search.copy()
            if board_copy is None:
                self._searching = False
                continue

            try:
                stats = self.searcher.find_best_move(
                    board_copy,
                    self.search_depth,
                    self._evaluate_batch
                )
                try:
                    self._result_queue.put_nowait((stats, self._game_id))
                except queue.Full:
                    pass
                pygame.event.post(pygame.event.Event(self.SEARCH_COMPLETE))
            except Exception as e:
                print(f"Search worker error: {e}")
            finally:
                self._searching = False

    def _on_search_complete(self):
        """Called when SEARCH_COMPLETE event is processed."""
        try:
            stats, game_id = self._result_queue.get_nowait()
        except queue.Empty:
            return
        if game_id != self._game_id:
            return
        self._current_result = stats
        self._searching = False

    def _start_search_if_idle(self):
        """Post SEARCH_REQUEST if not already searching."""
        if not self._searching and not self.terminated and not self.paused:
            self._searching = True
            board = self.env.game.board.copy()
            self._current_board_for_search = board
            pygame.event.post(pygame.event.Event(self.CUSTOM_SEARCH_REQUEST))
            self._search_event.set()

    def _reset_game(self):
        """Reset: increment game_id, clear state, restart search. Non-blocking."""
        self._game_id += 1
        self._searching = False
        self._current_result = None
        self.env.reset()
        self.move_history.clear()
        self.cumulative = {
            'total_moves': 0,
            'total_time_ms': 0.0,
            'total_nodes': 0,
            'total_tt_lookups': 0,
            'total_tt_hits': 0,
        }
        if self.search_thread and self.search_thread.is_alive():
            self.search_thread.join(timeout=0.5)
        self._start_search_if_idle()

    def _toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.set_text('Resume' if self.paused else 'Pause')
        if not self.paused:
            self._start_search_if_idle()

    def _execute_action(self, action: int):
        last_action = action
        obs, _, terminated, _, info = self.env.step(action)
        self.terminated = terminated

        if hasattr(self, '_current_result') and self._current_result:
            stats = self._current_result
            self.move_history.append(stats)
            self.cumulative['total_moves'] += 1
            self.cumulative['total_time_ms'] += stats.get('think_ms', 0)
            self.cumulative['total_nodes'] += stats.get('nodes_visited', 0)
            self.cumulative['total_tt_lookups'] += stats.get('tt_lookups', 0)
            self.cumulative['total_tt_hits'] += stats.get('tt_hits', 0)

        if terminated:
            self._draw_game_over(self.env.game.score, self.env.game.max_tile)

    def _update_history_display(self):
        """Rebuild history HTML text and update UITextBox."""
        if not hasattr(self, 'history_text'):
            return

        action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        arrow_colors = ['#76E4A3', '#F5A623', '#888888', '#888888']

        lines = []
        history = list(reversed(self.move_history[-50:]))
        for i, mh in enumerate(history):
            global_idx = len(self.move_history) - i
            best = mh.get('best_move', 0)
            arrow = ['↑', '→', '↓', '←'][best]
            color = arrow_colors[best]
            ms = mh.get('think_ms', 0)
            nodes = mh.get('nodes_visited', 0)
            lines.append(f"<b>#{global_idx}</b> <font color='{color}'>{arrow}</font> {ms:.0f}ms, {nodes:,} nodes")

        html = "<br>".join(lines) if lines else "No moves yet."
        self.history_text.set_text(html)

    def _update_stats_labels(self):
        """Update stats labels from current result or last move."""
        last_s = self.move_history[-1] if self.move_history else {
            'think_ms': 0, 'nodes_visited': 0, 'batches_eval': 0,
            'best_move': 0, 'move_scores': [-1e9] * 4
        }

        n = self.cumulative['total_moves']
        self.move_label.set_text(f"Move #{n}")
        self.think_label.set_text(f"think: {last_s.get('think_ms', 0):.1f}ms")
        self.nodes_label.set_text(f"nodes: {last_s.get('nodes_visited', 0):,}")
        self.batches_label.set_text(f"batches: {last_s.get('batches_eval', 0)}")

        scores = last_s.get('move_scores', [-1e9] * 4)
        best = last_s.get('best_move', 0)
        action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        best_score = scores[best] if best < len(scores) else 0.0
        self.best_move_label.set_text(f"best: {action_names[best]} ({best_score:.2f})")

        max_score = max(scores) if max(scores) > -1e8 else 1.0
        for i, (bar, score_label) in enumerate(zip(self.score_bars, self.score_labels)):
            s = scores[i] if i < len(scores) else -1e9
            if s > -1e8:
                pct = max(0.0, min(100.0, (s / max_score) * 100.0))
                bar.set_current_progress(pct)
                score_label.set_text(f"{s:.2f}")
            else:
                bar.set_current_progress(0.0)
                score_label.set_text("--")

    def _draw_game_over(self, score, max_tile):
        """Draw game over overlay."""
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

    def run(self):
        obs, _ = self.env.reset()
        self.terminated = False
        self.paused = False
        self._current_board_for_search = self.env.game.board.copy()
        self._start_search_if_idle()

        running = True
        while running:
            time_delta = self.clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == self.SEARCH_COMPLETE:
                    self._on_search_complete()
                elif event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.new_game_button:
                        self._reset_game()
                    elif event.ui_element == self.pause_button:
                        self._toggle_pause()
                self.manager.process_events(event)

            if not self.paused and not self.terminated:
                if self._current_result is not None and not self._searching:
                    action = self._current_result['best_move']
                    self._current_result = None
                    self._execute_action(action)

                if not self._searching and not self.terminated:
                    self._start_search_if_idle()

            self._draw_board(self.env.game.board, offset_x=0)

            c = self.cumulative
            n = c['total_moves']
            tt_rate = (c['total_tt_hits'] / c['total_tt_lookups'] * 100) if c['total_tt_lookups'] > 0 else 0
            self.cum_label.set_text(
                f"Moves: {n} | {c['total_time_ms']:.0f}ms | {c['total_nodes']:,} nodes | {tt_rate:.0f}% tt | Score: {self.env.game.score}"
            )

            self._update_stats_labels()
            self._update_history_display()

            self.manager.update(time_delta)
            self.manager.draw_ui(self.screen)
            pygame.display.flip()

        self._worker_running = False
        self._search_event.set()
        if self.search_thread and self.search_thread.is_alive():
            self.search_thread.join(timeout=1.0)
        pygame.quit()