import numpy as np

from .utility import row_to_number, stack_row, merge_row


class Fast2048:
    move_row_LUT = []
    move_reward_LUT = []

    def __init__(self):
        if not Fast2048.move_row_LUT:
            self.init_LUT()
        self.board = None
        self.max_tile = None
        self.empty_cells = None
        self.sum_tiles = None
        self.score = None
        self.done = None
        self.total_reward = None
        self.useless_move_count = 0
        self.reset()

    def init_LUT(self):
        for i in range(65536):
            row = [(i >> 0) & 0xf, (i >> 4) & 0xf, (i >> 8) & 0xf, (i >> 12) & 0xf]
            row = stack_row(row)
            row, reward = merge_row(row)
            row = stack_row(row)
            Fast2048.move_row_LUT.append(row)
            Fast2048.move_reward_LUT.append(reward)

    def reset(self):
        self.board = np.array([[0 for _ in range(4)]for _ in range(4)])
        self.max_tile = 0
        self.empty_cells = 16
        self.sum_tiles = 0
        self.score = 0
        self.done = False
        self.total_reward=0
        self.generate_random()
        self.generate_random()
        self.update_values()

    def update_values(self):
        self.empty_cells = 0
        self.sum_tiles = 0

        for row in self.board:
            for cell in row:
                self.max_tile = max(self.max_tile, cell)
                self.sum_tiles += cell
                if cell==0:
                    self.empty_cells+=1

    def generate_random(self):
        num = 1 if np.random.random() < 0.9 else 2
        empty_cells = np.argwhere(self.board == 0)

        if empty_cells.size==0:
            return

        chosen_position = empty_cells[np.random.choice(len(empty_cells))]

        self.board[chosen_position[0], chosen_position[1]] = num

    def check_done(self):
        if self.empty_cells==0:
            for i in range(4):
                for j in range(4):
                    if i+1<4 and self.board[i][j]==self.board[i+1][j]:
                        return False
                    if j+1<4 and self.board[i][j]==self.board[i][j+1]:
                        return False
            return True
        return False

    def move(self, direction):
        merge_score=0
        prev=self.board.copy()

        if direction==3: # left
            for i in range(4):
                index= row_to_number(self.board[i])
                merge_score+=self.move_reward_LUT[index]
                self.board[i] = self.move_row_LUT[index]
        elif direction==1: # right
            for i in range(4):
                index= row_to_number(self.board[i][::-1])
                merge_score+=self.move_reward_LUT[index]
                self.board[i] = self.move_row_LUT[index][::-1]
        elif direction==0: # up
            for i in range(4):
                index= row_to_number(self.board[:,i])
                merge_score+=self.move_reward_LUT[index]
                self.board[:,i] = self.move_row_LUT[index]
        elif direction==2: # down
            for i in range(4):
                index= row_to_number(self.board[:,i][::-1])
                merge_score+=self.move_reward_LUT[index]
                self.board[:,i] = self.move_row_LUT[index][::-1]


        self.score+=merge_score


        moved=not np.array_equal(prev, self.board)
        if moved:
            self.generate_random()
            reward=merge_score
            #self.useless_move_count=0
        else:
            #if self.useless_move_count>=1:
                #self.done=True
            #self.useless_move_count += 1
            reward=-2


        self.update_values()
        self.done=self.check_done()

        return reward, self.done