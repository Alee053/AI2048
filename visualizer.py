import pygame
from stable_baselines3 import PPO

from game.Game2048Env import Game2048Env
import pygame

MODEL_PATH = "models/ppo_run_1/final_model.zip"

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


def draw_board(screen, board, font):
    screen.fill(BG_COLOR)

    for r in range(4):
        for c in range(4):
            tile_value = 2 ** board[r][c] if board[r][c] != 0 else 0

            tile_color = TILE_COLORS.get(tile_value, (60, 58, 50))  # Default to dark color for high tiles

            pygame.draw.rect(screen, tile_color, (c * 100 + 10, r * 100 + 10, 80, 80), border_radius=5)

            if tile_value != 0:
                text_color = TEXT_COLOR_DARK if tile_value < 8 else TEXT_COLOR_LIGHT
                text_surface = font.render(str(tile_value), True, text_color)
                text_rect = text_surface.get_rect(center=(c * 100 + 50, r * 100 + 50))
                screen.blit(text_surface, text_rect)


def test_agent(model_path):
    pygame.init()
    screen = pygame.display.set_mode((400, 400))
    pygame.display.set_caption("2048 PPO Agent")
    font = pygame.font.Font(None, 48)

    env = Game2048Env()

    model = PPO.load(model_path)

    obs, info = env.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        draw_board(screen, env.game.board, font)

        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {env.game.score}, Max Tile: {2 ** env.game.max_tile}")
            pygame.time.wait(3000)
            running = False
        pygame.time.wait(300)

    pygame.quit()

if __name__ == '__main__':
    test_agent(MODEL_PATH)