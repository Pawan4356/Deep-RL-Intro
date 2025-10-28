'''Q-Learning Implementation using NN'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

# Environment
cartPoleEnv = gym.make("CartPole-v1", render_mode="rgb_array")

# Q Network
# class QNetwork(nn.Module):
#     def __init__(self, stateDim, actionDim):
#         super(QNetwork, self).__init__()
#         self.fc1 = nn.Linear(stateDim, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.output = nn.Linear(32, actionDim)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.output(x)
# ====================== OR ======================
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

stateDim = cartPoleEnv.observation_space.shape[0]
actionDim = cartPoleEnv.action_space.n
qNetwork = QNetwork(stateDim, actionDim)

# Parameters
ALPHA = 0.001
EPSILON = 1.0
EPSILONDECAY = 1.005
GAMMA = 0.99
NUMEPISODES = 500

# Policy function (epsilon-greedy)
def policy(state, explore=0.0):
    with torch.no_grad():
        qValues = qNetwork(state)
        # Choose the best action (exploit)
        action = torch.argmax(qValues[0]).item()
        
    # Check for exploration
    if torch.rand(1).item() <= explore:
        # Choose a random action (explore)
        action = torch.randint(0, actionDim, (1,)).item()
    
    return action


for episode in range(NUMEPISODES):
    state, _ = cartPoleEnv.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    done = False
    totalReward = 0
    episodeLength = 0

    while not done:
        # 1. Choose action (A) from state (S) using policy
        action = policy(state, EPSILON)
        
        # 2. Take action, get reward (R) and next state (S')
        nextState, reward, terminated, truncated, _ = cartPoleEnv.step(action)
        done = terminated or truncated
        nextState = torch.tensor(nextState, dtype=torch.float32).unsqueeze(0)

        # 3. COMPUTE Q-LEARNING TARGET
        with torch.no_grad():
            # Find the max Q-value for the next state (S')
            # This is the "max_a' Q(S', a')" part
            max_next_q = torch.max(qNetwork(nextState)[0])
            
            # The target is R + gamma * max_Q(S', a')
            target = reward + (0 if done else GAMMA * max_next_q)

        # 4. Compute prediction and loss
        # Prediction is the Q-value for the original state (S) and action (A)
        qValues = qNetwork(state)
        currentQ = qValues[0][action]
        loss = (target - currentQ) ** 2 / 2

        # 5. Manual gradient update (FIXED)
        qNetwork.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in qNetwork.parameters():
                # Use gradient DESCENT (minus) to MINIMIZE the loss
                param -= ALPHA * param.grad

        # 6. Update state
        state = nextState
        
        # Note: We do NOT set 'action = nextAction' like in Sarsa
        
        totalReward += reward
        episodeLength += 1

    print(f"Episode: {episode+1:4d} | Length: {episodeLength:4d} | Reward: {totalReward:6.3f} | Epsilon: {EPSILON:6.3f}")
    
    # Epsilon decay
    if EPSILON > 0.01: # Set a minimum epsilon
        EPSILON /= EPSILONDECAY

# Save model
torch.save(qNetwork.state_dict(), "QLearningQNet.pt")
cartPoleEnv.close()