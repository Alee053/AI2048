import os

import pygame
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env

from src.Game2048Env import Game2048Env

TILE_COLORS = {
    0: (205, 193, 180), 2: (238, 228, 218), 4: (237, 224, 200),
    8: (242, 177, 121), 16: (245, 149, 99), 32: (246, 124, 95),
    64: (246, 94, 59), 128: (237, 207, 114), 256: (237, 204, 97),
    512: (237, 200, 80), 1024: (237, 197, 63), 2048: (237, 194, 46),
    4096: (60, 58, 50), 8192: (60, 58, 50),
}
TEXT_COLOR_DARK = (119, 110, 101)
TEXT_COLOR_LIGHT = (249, 246, 242)
BG_COLOR = (187, 173, 160)

class Visualizer:
    def __init__(self):
        self.model_path = ""

    def load_model(self, model_path):
        if not model_path:
            raise ValueError("Model path cannot be empty.")
        if os.path.exists(model_path):
            self.model_path = model_path
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")

    def draw_board(self, board,screen,font):
        screen.fill(BG_COLOR)

        width, height = screen.get_size()

        for r in range(4):
            for c in range(4):
                tile_value = 2 ** board[r][c] if board[r][c] != 0 else 0

                tile_color = TILE_COLORS.get(tile_value, (60, 58, 50))  # Default to dark color for high tiles

                pygame.draw.rect(screen, tile_color, (c * width/4 + width/40, r * width/4 + width/40, width/5, width/5), border_radius=int(width/80))

                if tile_value != 0:
                    text_color = TEXT_COLOR_DARK if tile_value < 8 else TEXT_COLOR_LIGHT
                    text_surface = font.render(str(tile_value), True, text_color)
                    text_rect = text_surface.get_rect(center=(c * width/4 + width/8, r * width/4 + width/8))
                    screen.blit(text_surface, text_rect)

    def draw_stats(self, score, action, screen):
            width, height = screen.get_size()
            small_font = pygame.font.Font(None, int(width / 18))
            text = f"Score: {score}  Action: {action==0 and 'Up' or action==1 and 'Right' or action==2 and 'Down' or 'Left'}"
            stats_surface = small_font.render(text, True, (255, 255, 255))
            bottom_band = 100
            stats_rect = stats_surface.get_rect(center=(width // 2, int(height - bottom_band / 2)))
            bg_rect = stats_rect.inflate(20, 10)
            pygame.draw.rect(screen, (50, 50, 50), bg_rect, border_radius=8)
            screen.blit(stats_surface, stats_rect)

    def test_agent(self):
        if not self.model_path:
            raise ValueError("Model path not set. Please set the model_path attribute before testing the agent.")

        pygame.init()
        pygame.display.set_caption("2048 PPO Agent")
        vec_env = make_vec_env(Game2048Env, n_envs=16)


        screen = pygame.display.set_mode((800, 900))
        font = pygame.font.Font(None, int(screen.get_size()[0]/9))

        env = Game2048Env()
        model = MaskablePPO.load(self.model_path,vec_env=vec_env)
        obs, info = env.reset()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            print(action,reward,env.game.score)

            self.draw_board(env.game.board,screen,font)
            self.draw_stats(env.game.score,action,screen)

            pygame.display.flip()

            if terminated or truncated:
                print(f"Game Over! Final Score: {env.game.score}, Max Tile: {2 ** env.game.max_tile}")
                pygame.time.wait(3000)
                running = False
            pygame.time.wait(200)

        pygame.quit()