import numpy as np
from twenty_forty_eight_ai.env.game import Fast2048

game = Fast2048()
game.board = np.array([
    [3, 4, 6, 0],
    [2, 8, 3, 1],
    [2, 0, 0, 0],
    [1, 0, 0, 0]
], dtype=np.int32)

print("UP valid:", game.is_move_valid(0))
print("RIGHT valid:", game.is_move_valid(1))
print("DOWN valid:", game.is_move_valid(2))
print("LEFT valid:", game.is_move_valid(3))

score, done, moved = game.move(0)
print(f"Executed UP: moved={moved}, board=\n{game.board}")
