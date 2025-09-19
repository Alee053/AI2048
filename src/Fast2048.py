import numpy as np
from reward_function import ROW_GRADIENT, COL_GRADIENT

class Fast2048:
    move_row_LUT = []
    move_reward_LUT = []
    move_valid_LUT = []

    def __init__(self):
        if not Fast2048.move_row_LUT:
            self.init_LUT()
        self.board = None
        self.max_tile = None
        self.score = None
        self.done = None

        # Curriculum learning parameters
        self.prob_4 = 0.1
        self.p_helpful = 0.0

        self.reset()

    def init_LUT(self):
        for i in range(65536):
            original_row = [(i >> 0) & 0xf, (i >> 4) & 0xf, (i >> 8) & 0xf, (i >> 12) & 0xf]
            row = original_row.copy()

            row = stack_row(row)
            row, reward = merge_row(row)
            row = stack_row(row)

            Fast2048.move_row_LUT.append(row)
            Fast2048.move_reward_LUT.append(reward)

            Fast2048.move_valid_LUT.append(original_row != row)

    def reset(self):
        self.board = np.array([[0 for _ in range(4)]for _ in range(4)])
        self.max_tile = 0
        self.score = 0
        self.done = False
        self.generate_random()
        self.generate_random()
        self.update_values()

    def update_values(self):
        for row in self.board:
            for cell in row:
                self.max_tile = max(self.max_tile, cell)

    def is_move_valid(self, action):
        if action == 3:  # left
            for i in range(4):
                if self.move_valid_LUT[row_to_number(self.board[i])]: return True
        elif action == 1:  # right
            for i in range(4):
                if self.move_valid_LUT[row_to_number(self.board[i][::-1])]: return True
        elif action == 0:  # up
            for i in range(4):
                if self.move_valid_LUT[row_to_number(self.board[:, i])]: return True
        elif action == 2:  # down
            for i in range(4):
                if self.move_valid_LUT[row_to_number(self.board[:, i][::-1])]: return True
        return False

    def generate_random(self):
        # The logic for choosing between a 2 or 4 remains the same
        num = 1 if np.random.random() > self.prob_4 else 2

        empty_cells = np.argwhere(self.board == 0)

        if empty_cells.size == 0:
            return

        # --- REFINED CURRICULUM SPAWN LOGIC ---

        # 1. Rank all empty cells from best to worst
        log_board = np.log2(self.board, out=np.zeros_like(self.board, dtype=np.float32), where=(self.board != 0))
        s1 = np.sum(log_board * ROW_GRADIENT)
        s2 = np.sum(log_board * COL_GRADIENT)
        gradient_to_use = ROW_GRADIENT if s1 >= s2 else COL_GRADIENT

        # Create a list of (gradient_value, position) for each empty cell
        ranked_cells = sorted(
            [(gradient_to_use[r, c], (r, c)) for r, c in empty_cells],
            key=lambda x: x[0]
        )

        # 2. Determine the size of the "good" candidate pool based on p_helpful
        # The pool size shrinks as p_helpful decays from 1.0 to 0.0
        # The formula is len * (1.0 - p_helpful), ensuring at least 1 candidate
        num_candidates = int(np.ceil(len(ranked_cells) * (1.0 - self.p_helpful)))
        num_candidates = max(1, num_candidates)

        # 3. Select the pool of best candidates
        candidate_pool = ranked_cells[:num_candidates]

        # 4. Choose a random position from within that top-tier pool
        chosen_index = np.random.choice(len(candidate_pool))
        chosen_position = candidate_pool[chosen_index][1]

        self.board[chosen_position[0], chosen_position[1]] = num

    def check_done(self):
        res=False
        for i in range(4):
            res|=self.is_move_valid(i)
            if res:
                break
        return not res


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


        self.update_values()
        self.done=self.check_done()

        return merge_score, self.done, moved

def row_to_number(row):
    return row[0] | row[1]<<4 | row[2]<<8 | row[3]<<12
def stack_row(row):
    for k in range(4):
        for i in range(1, 4):
            if row[i]!=0 and row[i - 1]==0:
                row[i-1]=row[i]
                row[i]=0
    return row
def merge_row(row):
    reward=0
    for i in range(1,4):
        if row[i-1]==row[i] and row[i]!=0:
            row[i-1]+=1
            row[i]=0
            reward+=2**row[i-1]
    return [row,reward]