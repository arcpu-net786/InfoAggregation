import torch
import random
import numpy as np
from collections import deque
from game import CarGameAI
from model import Linear_QNet, QTrainer


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.0001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.gamma = 0.85 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.outputs = 100
        self.model = Linear_QNet(225, 256, 256, self.outputs)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):

        #state of the environment, used for features
        state = [
            # front-center coordinates
            int(game.car.front[0]),
            int(game.car.front[1]),

            #front-left coordinates
            int(game.car.x_fl),
            int(game.car.y_fl),

            #front-right coordinates
            int(game.car.x_fr),
            int(game.car.y_fr),

            #back-right coordinates
            int(game.car.x_br),
            int(game.car.y_br),

            #back-left coordinates
            int(game.car.x_bl),
            int(game.car.y_bl),

            # Car direction/bearing
            game.car.deg,
            
            #number of iterations so far
            game.frame_iteration,

            #relative position of food
            int(game.goal[0] > game.car.front[0]),
            int(game.goal[1] > game.car.front[1]),
            
            # Manhattan distance from front-center to food
            game.goal[0] - game.car.front[0],  # x_dist
            game.goal[1] - game.car.front[1]   # y_dist

            ] + [game.view_ahead_pt(dist, ang) for dist in range(20, 121, 10) for ang in range(-90, 91, 10)]

        return np.array(state, dtype=int)


    def remember(self, state, action, reward, next_state, done): #add "sarsa" to memory
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones) #train model on batch of "sarsa"s


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done) #train model on single "sarsa"


    def get_action(self, state):
        final_move = [0] * self.outputs

        #explore random action if the number of games played < 250
        if self.n_games < 250:
            move = random.randint(0, self.outputs-1)
            final_move[move] = 1

        #use model to predict best action if number of games >= 250
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move