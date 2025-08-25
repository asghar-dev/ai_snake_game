import pygame
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

# Constants
WIDTH, HEIGHT = 750, 750
GRID_SIZE = 20
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")

font = pygame.font.Font(None, 50)
small_font = pygame.font.Font(None, 30)
clock = pygame.time.Clock()

# Global variables for the AI agent and its state
snake_speed = 5
agent = None
record = 0
total_score = 0
game_number = 0

def draw_button(text, rect, color, text_color):
    """A helper function to draw a button on the screen."""
    pygame.draw.rect(screen, color, rect)
    text_surface = font.render(text, True, text_color)
    text_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, text_rect)

def get_random_position():
    """Generates a random position on the grid."""
    x = random.randrange(0, WIDTH, GRID_SIZE)
    y = random.randrange(0, HEIGHT, GRID_SIZE)
    return (x, y)

def is_collision(head, snake_body, boundaries):
    """Checks for collision with walls or body."""
    if head[0] < 0 or head[0] >= boundaries[0] or \
       head[1] < 0 or head[1] >= boundaries[1]:
        return True
    if head in snake_body:
        return True
    return False

def get_state(head, direction_vec, snake_body, food_pos):
    """Generates a structured 'state' array for the AI."""
    point_straight = (head[0] + direction_vec[0] * GRID_SIZE, head[1] + direction_vec[1] * GRID_SIZE)
    point_right = (head[0] + direction_vec[1] * GRID_SIZE, head[1] + direction_vec[0] * -1 * GRID_SIZE)
    point_left = (head[0] + direction_vec[1] * GRID_SIZE * -1, head[1] + direction_vec[0] * GRID_SIZE)

    danger_straight = is_collision(point_straight, snake_body, (WIDTH, HEIGHT))
    danger_right = is_collision(point_right, snake_body, (WIDTH, HEIGHT))
    danger_left = is_collision(point_left, snake_body, (WIDTH, HEIGHT))

    food_is_up = food_pos[1] < head[1]
    food_is_down = food_pos[1] > head[1]
    food_is_left = food_pos[0] < head[0]
    food_is_right = food_pos[0] > head[0]
    
    state = [
        danger_straight,
        danger_right,
        danger_left,
        
        direction_vec == (0, -1), # UP
        direction_vec == (0, 1),  # DOWN
        direction_vec == (-1, 0), # LEFT
        direction_vec == (1, 0),  # RIGHT
        
        food_is_up,
        food_is_down,
        food_is_left,
        food_is_right
    ]
    return np.array(state, dtype=int)

class DQNAgent:
    """A simple Deep Q-Network agent."""
    def __init__(self):
        self.gamma = 0.9 # Discount rate
        self.epsilon = 0.5 # A high exploration rate at the beginning
        self.memory = deque(maxlen=10000) # Long-term memory
        self.batch_size = 1000

        self.model = nn.Sequential(
            nn.Linear(11, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        """Stores a new experience in the memory."""
        self.memory.append((state, action, reward, next_state, done))
        
    def get_action(self, state):
        """Decides the next action."""
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 2)
        
        state_tensor = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state_tensor)
        return torch.argmax(prediction).item()
        
    def train_long_memory(self):
        """Trains the model on a batch of experiences from memory."""
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        dones = torch.tensor(np.array(dones), dtype=torch.bool)
        
        pred = self.model(states)
        target = pred.clone()

        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                Q_new = rewards[idx] + self.gamma * torch.max(self.model(next_states[idx]))
            target[idx][actions[idx].item()] = Q_new
            
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

    def save_model(self, file_path="snake_model.pth"):
        """Saves the model's state to a file."""
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path="snake_model.pth"):
        """Loads the model's state from a file."""
        try:
            self.model.load_state_dict(torch.load(file_path))
            self.model.eval()
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("Model file not found. A new model will be created.")

def train_ai(n_games=1000):
    """Automates the training of the AI agent over many games."""
    global agent, record, total_score, game_number
    if agent is None:
        agent = DQNAgent()

    epsilon_decay = 0.500
    min_epsilon = 0.05
    
    for i in range(n_games):
        score = game_loop(ai_mode=True)
        game_number += 1
        total_score += score
        
        # Epsilon decay
        if agent.epsilon > min_epsilon:
            agent.epsilon *= epsilon_decay
        
        if score > record:
            record = score
            print(f"New Record! Score: {record}")
            agent.save_model() # Auto-save the best model

        avg_score = total_score / game_number if game_number > 0 else 0
        print(f"Game: {game_number}, Score: {score}, Record: {record}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.4f}")

def game_loop(ai_mode=False):
    """The main game loop."""
    global agent
    if ai_mode and agent is None:
        agent = DQNAgent()
    
    snake_pos = get_random_position()
    snake_body = [snake_pos]
    food_pos = get_random_position()
    direction_vec = (1, 0)
    score = 0
    game_over = False
    
    # Mapping for actions
    action_map = {0: 'STRAIGHT', 1: 'RIGHT', 2: 'LEFT'}
    
    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Save the model automatically on exit
                if agent:
                    agent.save_model()
                pygame.quit()
                sys.exit()
            if not ai_mode and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and direction_vec != (0, 1):
                    direction_vec = (0, -1)
                if event.key == pygame.K_DOWN and direction_vec != (0, -1):
                    direction_vec = (0, 1)
                if event.key == pygame.K_LEFT and direction_vec != (1, 0):
                    direction_vec = (-1, 0)
                if event.key == pygame.K_RIGHT and direction_vec != (-1, 0):
                    direction_vec = (1, 0)
        
        if ai_mode:
            state_old = get_state(snake_pos, direction_vec, snake_body, food_pos)
            action_index = agent.get_action(state_old)
            ai_choice = action_map[action_index]
            
            direction_vec_new = direction_vec
            if ai_choice == 'RIGHT':
                direction_vec_new = (direction_vec[1], -direction_vec[0])
            elif ai_choice == 'LEFT':
                direction_vec_new = (-direction_vec[1], direction_vec[0])
            
            direction_vec = direction_vec_new

        snake_pos = (snake_pos[0] + direction_vec[0] * GRID_SIZE, snake_pos[1] + direction_vec[1] * GRID_SIZE)
        
        game_over = is_collision(snake_pos, snake_body, (WIDTH, HEIGHT))
        
        reward = 0
        if game_over:
            reward = -10
        elif snake_pos == food_pos:
            reward = 1
            score += 1
            food_pos = get_random_position()
        else:
            snake_body.pop()

        snake_body.insert(0, snake_pos)
        
        if ai_mode:
            state_new = get_state(snake_pos, direction_vec, snake_body, food_pos)
            agent.remember(state_old, action_index, reward, state_new, game_over)
            agent.train_long_memory()

        screen.fill(WHITE)
        for pos in snake_body:
            pygame.draw.rect(screen, GREEN, pygame.Rect(pos[0], pos[1], GRID_SIZE, GRID_SIZE))
        
        pygame.draw.rect(screen, RED, pygame.Rect(food_pos[0], food_pos[1], GRID_SIZE, GRID_SIZE))
        
        score_text = font.render(f"Score: {score}", True, BLACK)
        score_rect = score_text.get_rect(topleft=(10, 10))
        screen.blit(score_text, score_rect)

        # Display AI training stats
        if ai_mode and game_number > 0:
            avg_score = total_score / game_number
            stats_text = small_font.render(f"Games: {game_number} | Record: {record} | Avg: {avg_score:.2f}", True, BLACK)
            stats_rect = stats_text.get_rect(topright=(WIDTH - 10, 10))
            screen.blit(stats_text, stats_rect)
        elif ai_mode:
            stats_text = small_font.render(f"Games: {game_number} | Record: {record} | Avg: 0.00", True, BLACK)
            stats_rect = stats_text.get_rect(topright=(WIDTH - 10, 10))
            screen.blit(stats_text, stats_rect)

        pygame.display.flip()
        clock.tick(snake_speed)
    
    return score

def settings_menu():
    """Settings menu to adjust the game speed."""
    global snake_speed
    
    while True:
        screen.fill(WHITE)
        
        speed_text = font.render(f"Speed: {snake_speed}", True, BLACK)
        speed_rect = speed_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 100))
        screen.blit(speed_text, speed_rect)
        
        increase_rect = pygame.Rect(WIDTH // 2 + 50, HEIGHT // 2 - 50, 100, 50)
        decrease_rect = pygame.Rect(WIDTH // 2 - 150, HEIGHT // 2 - 50, 100, 50)
        back_rect = pygame.Rect(WIDTH // 2 - 150, HEIGHT // 2 + 50, 300, 80)
        
        draw_button("+", increase_rect, GRAY, BLACK)
        draw_button("-", decrease_rect, GRAY, BLACK)
        draw_button("Back", back_rect, GRAY, BLACK)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Save the model automatically on exit
                if agent:
                    agent.save_model()
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if increase_rect.collidepoint(event.pos):
                    snake_speed += 1
                if decrease_rect.collidepoint(event.pos):
                    if snake_speed > 1:
                        snake_speed -= 1
                if back_rect.collidepoint(event.pos):
                    return

        pygame.display.flip()

def main_menu():
    """Main menu screen with 'Start' and 'Settings' buttons."""
    global agent
    if agent is None:
        agent = DQNAgent()
        # Auto-load the model on startup
        agent.load_model()
        
    while True:
        screen.fill(WHITE)

        start_button_rect = pygame.Rect(WIDTH // 2 - 150, HEIGHT // 2 - 150, 300, 60)
        train_ai_rect = pygame.Rect(WIDTH // 2 - 150, HEIGHT // 2 - 70, 300, 60)
        start_ai_rect = pygame.Rect(WIDTH // 2 - 150, HEIGHT // 2 + 10, 300, 60)
        save_model_rect = pygame.Rect(WIDTH // 2 - 150, HEIGHT // 2 + 90, 300, 60)
        load_model_rect = pygame.Rect(WIDTH // 2 - 150, HEIGHT // 2 + 170, 300, 60)
        settings_button_rect = pygame.Rect(WIDTH // 2 - 150, HEIGHT // 2 + 250, 300, 60)

        draw_button("Play Manually", start_button_rect, GRAY, BLACK)
        draw_button("Train AI", train_ai_rect, GRAY, BLACK)
        draw_button("Start AI", start_ai_rect, GRAY, BLACK)
        draw_button("Save Model", save_model_rect, GRAY, BLACK)
        draw_button("Load Model", load_model_rect, GRAY, BLACK)
        draw_button("Settings", settings_button_rect, GRAY, BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Save the model automatically on exit
                if agent:
                    agent.save_model()
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if start_button_rect.collidepoint(event.pos):
                    game_loop(ai_mode=False)
                if train_ai_rect.collidepoint(event.pos):
                    train_ai()
                if start_ai_rect.collidepoint(event.pos):
                    game_loop(ai_mode=True)
                if save_model_rect.collidepoint(event.pos):
                    agent.save_model()
                if load_model_rect.collidepoint(event.pos):
                    agent.load_model()
                if settings_button_rect.collidepoint(event.pos):
                    settings_menu()

        pygame.display.flip()

if __name__ == "__main__":
    main_menu()
    