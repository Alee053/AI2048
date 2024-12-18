import numpy as np

move_row_LUT=[]
move_reward_LUT=[]

def init_LUT():
    for i in range(65536):
        row = [(i >> 0) & 0xf, (i >> 4) & 0xf, (i >> 8) & 0xf, (i >> 12) & 0xf]
        row = stack_row(row)
        row, reward = merge_row(row)
        row = stack_row(row)
        move_row_LUT.append(row)
        move_reward_LUT.append(reward)

def row_to_number(row):
    return row[0] | row[1]<<4 | row[2]<<8 | row[3]<<12

def stack_row(row):
    for k in range(4):
        for i in range(1, 4):
            if(row[i]!=0 and row[i-1]==0):
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


invalid_penalty=5
weight_empty=2


class Fast2048:
    def __init__(self):
        init_LUT()
        self.board = None
        self.max_tile = None
        self.empty_cells = None
        self.sum_tiles = None
        self.score = None
        self.done = None
        self.total_reward = None
        self.reset()

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

    def move(self, dir):
        reward=0
        prev=[row_to_number(self.board[0]), row_to_number(self.board[1]), row_to_number(self.board[2]), row_to_number(self.board[3])]

        if(dir==3):
            for i in range(4):
                id= row_to_number(self.board[i])
                reward=move_reward_LUT[id]
                self.board[i] = move_row_LUT[id]
        elif(dir==1):
            for i in range(4):
                id= row_to_number(self.board[i][::-1])
                reward=move_reward_LUT[id]
                self.board[i] = move_row_LUT[id][::-1]
        elif(dir==0):
            for i in range(4):
                id= row_to_number(self.board[:,i])
                reward=move_reward_LUT[id]
                self.board[:,i] = move_row_LUT[id]
        elif(dir==2):
            for i in range(4):
                id= row_to_number(self.board[:,i][::-1])
                reward=move_reward_LUT[id]
                self.board[:,i] = move_row_LUT[id][::-1]

        after=[row_to_number(self.board[0]), row_to_number(self.board[1]), row_to_number(self.board[2]), row_to_number(self.board[3])]

        self.score+=reward


        moved=(prev!=after)
        if moved:
            self.generate_random()
        else:
            reward-=invalid_penalty


        self.update_values()

        reward+=self.empty_cells*weight_empty

        self.total_reward+=reward

        if self.total_reward<=-50:
            self.done=True
        self.done|=self.check_done()

        return reward, self.done

    def show_board(self):
        for row in self.board:
            for cell in row:
                print(2**cell if cell!=0 else 0, end=' ')
            print()
        print("Score: ", self.score)