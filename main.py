import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Deep_QNet, QTrainer
from helper import plot

MAX_MEMORY = 200_000
BATCH_SIZE = 1000
LR = 0.003
EPSILON_START = 0.1  # Initial exploration rate
EPSILON_DECAY = 0.99995  # Exploration rate decay factor
MIN_EPSILON = 0.001  # Minimum exploration rate

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = EPSILON_START  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Deep_QNet(15, 256, 128, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        dist_to_obstacle_right = self.get_distance_to_obstacle(game, head, Direction.RIGHT)
        dist_to_obstacle_left = self.get_distance_to_obstacle(game, head, Direction.LEFT)
        dist_to_obstacle_up = self.get_distance_to_obstacle(game, head, Direction.UP)
        dist_to_obstacle_down = self.get_distance_to_obstacle(game, head, Direction.DOWN)

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            dist_to_obstacle_right,
            dist_to_obstacle_left,
            dist_to_obstacle_up,
            dist_to_obstacle_down,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def get_distance_to_obstacle(self, game, point, direction):
        x, y = point.x, point.y
        dx, dy = 0, 0

        if direction == Direction.RIGHT:
            dx = 20
        elif direction == Direction.LEFT:
            dx = -20
        elif direction == Direction.UP:
            dy = -20
        elif direction == Direction.DOWN:
            dy = 20

        distance = 0
        while not game.is_collision(Point(x + dx, y + dy)):
            distance += 1
            x += dx
            y += dy

        return distance

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Increase exploration (epsilon-greedy strategy)
        self.epsilon = max(self.epsilon * EPSILON_DECAY, MIN_EPSILON)

        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
