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
    """A Pygame-based visualizer for watching a trained PPO agent play 2048.

    This class provides a graphical interface to observe the performance of a
    trained agent. It can run the agent in two modes:
    1.  **Raw PPO Mode**: The agent makes decisions based directly on the
        policy network's output.
    2.  **PPO + Expectimax**: The agent's decisions are enhanced by a C++
        Expectimax searcher, which uses the PPO's value function (critic)
        to evaluate future states.

    Attributes:
        use_expectimax (bool): Flag to determine if the Expectimax searcher is used.
        search_depth (int): The depth for the Expectimax search.
        screen (pygame.Surface): The main display surface.
        clock (pygame.time.Clock): Pygame clock for controlling the frame rate.
        fonts (dict): A dictionary of pre-loaded fonts for rendering text.
        env (Game2048Env): The game environment.
        model (MaskablePPO): The loaded Stable Baselines 3 PPO model.
        searcher (ExpectimaxSearcher or None): The C++ Expectimax searcher instance.
    """

    def __init__(self, model_path: str, use_expectimax: bool = True, search_depth: int = 3):
        """Initializes the Visualizer.

        Args:
            model_path (str): The file path to the saved PPO model (.zip).
            use_expectimax (bool, optional): Whether to use the Expectimax
                searcher. Defaults to True.
            search_depth (int, optional): The search depth for the Expectimax
                algorithm. Defaults to 3.

        Raises:
            FileNotFoundError: If the model file does not exist at the given path.
        """
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
        """Evaluates a batch of boards using the PPO critic.

        This method serves as a callback for the C++ Expectimax searcher. The
        searcher generates potential future board states and passes them to this
        function to get a value estimate from the trained PPO model's critic.

        Args:
            boards_list (List[np.ndarray]): A list of 4x4 game boards to evaluate.

        Returns:
            List[float]: A list of value estimates corresponding to each board.
        """
        if not boards_list:
            return []
        batch_tensor = board_to_tensor(np.array(boards_list))
        with torch.no_grad():
            values = self.model.policy.predict_values(
                torch.as_tensor(batch_tensor).to(self.model.device)
            )
        return values.cpu().numpy().flatten().tolist()

    def _get_font_for_tile(self, value: int) -> pygame.font.Font:
        """Selects an appropriate pre-loaded font based on the tile's value.

        Larger numbers get smaller fonts to fit within the tile.

        Args:
            value (int): The numerical value of the tile.

        Returns:
            pygame.font.Font: The pre-loaded Pygame font object.
        """
        if value < 100: return self.fonts["xlarge"]
        if value < 1000: return self.fonts["large"]
        return self.fonts["medium"]

    def _draw_board(self, board: np.ndarray):
        """Draws the 4x4 game grid and all the tiles onto the screen.

        Each tile is colored according to the theme, and its value is rendered
        on top.

        Args:
            board (np.ndarray): The 4x4 game board with log2-encoded tile values.
        """
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
        """Draws the statistics panel at the bottom of the screen.

        This panel displays the current score, max tile, step count, and the
        last action taken.

        Args:
            score (int): The current game score.
            max_tile_exp (int): The log2 value of the highest tile on the board.
            step (int): The current step count in the episode.
            action (int): The last action taken by the agent.
        """
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
        """Determines the next action using either raw PPO or PPO + Expectimax.

        If the Expectimax searcher is enabled, it calls the C++ extension to
        find the best move. Otherwise, it uses the standard prediction method
        of the PPO model.

        Args:
            obs (np.ndarray): The current observation from the environment.

        Returns:
            int: The integer representing the chosen action.
        """
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

    def _draw_game_over(self, score: int, max_tile: int):
        """Displays a "Game Over" screen with the final score and max tile.

        Args:
            score (int): The final score of the game.
            max_tile (int): The log2 value of the highest tile achieved.
        """
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
        """Runs the main visualization game loop.

        This loop handles Pygame events, gets actions from the agent, steps the
        environment, and orchestrates the drawing of all visual components.
        The loop continues until the user closes the window.
        """
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