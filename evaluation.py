import cv2
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F

# Load Environment 
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Load Trained SARSA/Q-Learning Q-Network 
class QNetwork(nn.Module):
    def __init__(self, stateDim, actionDim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(stateDim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, actionDim)
        )

    def forward(self, x):
        return self.model(x)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_net = QNetwork(state_dim, action_dim)
# q_net.load_state_dict(torch.load("SarsaQNet.pt", map_location=torch.device("cpu")))
q_net.load_state_dict(torch.load("QLearningQNet.pt", map_location=torch.device("cpu")))
q_net.eval()

# Policy Function 
def policy(state, epsilon=0.0):
    """Epsilon-greedy policy for evaluation."""
    if np.random.rand() < epsilon:
        return np.random.randint(action_dim)
    with torch.no_grad():
        q_values = q_net(torch.tensor(state, dtype=torch.float32))
    return torch.argmax(q_values).item()

# Evaluation Loop 
for episode in range(5):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        frame = env.render()
        cv2.imshow("CartPole", frame)
        cv2.waitKey(50)  # Reduce for faster playback

        action = policy(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward
        state = next_state

    print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

env.close()
cv2.destroyAllWindows()
