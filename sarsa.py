'''Sarsa Implementation using NN'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

# Environment
cartPoleEnv = gym.make("CartPole-v1", render_mode="rgb_array")

# Q Network
class QNetwork(nn.Module):
    def __init__(self, stateDim, actionDim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(stateDim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, actionDim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)

stateDim = cartPoleEnv.observation_space.shape[0]
actionDim = cartPoleEnv.action_space.n
qNetwork = QNetwork(stateDim, actionDim)

# Parameters
ALPHA = 0.001
EPSILON = 1.0
EPSILONDECAY = 1.005
GAMMA = 0.99
NUMEPISODES = 500

# Policy function
def policy(state, explore=0.0):
    with torch.no_grad():
        qValues = qNetwork(state)
        action = torch.argmax(qValues[0]).item()
    if torch.rand(1).item() <= explore:
        action = torch.randint(0, actionDim, (1,)).item()
    return action


for episode in range(NUMEPISODES):
    state, _ = cartPoleEnv.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    done = False
    totalReward = 0
    episodeLength = 0

    action = policy(state, EPSILON)

    while not done:
        nextState, reward, terminated, truncated, _ = cartPoleEnv.step(action)
        done = terminated or truncated
        nextState = torch.tensor(nextState, dtype=torch.float32).unsqueeze(0)
        nextAction = policy(nextState, EPSILON)

        # Compute target
        with torch.no_grad():
            target = reward + (0 if done else GAMMA * qNetwork(nextState)[0][nextAction])

        # Compute prediction and loss
        qValues = qNetwork(state)
        currentQ = qValues[0][action]
        loss = (target - currentQ) ** 2 / 2

        # Manual gradient update
        qNetwork.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in qNetwork.parameters():
                param += ALPHA * param.grad  # gradient ascent on Q (since delta = target - current)

        state = nextState
        action = nextAction
        totalReward += reward
        episodeLength += 1

    print(f"Episode: {episode+1:4d} | Length: {episodeLength:4d} | Reward: {totalReward:6.3f} | Epsilon: {EPSILON:6.3f}")
    EPSILON /= EPSILONDECAY

# Save model
torch.save(qNetwork.state_dict(), "SarsaQNet.pt")
cartPoleEnv.close()
