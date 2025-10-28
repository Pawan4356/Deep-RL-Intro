'''Sarsa Implementation using NN'''

import torch
import torch.nn as nn
import gymnasium as gym

# Environment
cartPoleEnv = gym.make("CartPole-v1", render_mode="rgb_array")

# Q Network
class QNetwork(nn.Module):
    def __init__(self, stateDim, actionDim):
        super().__init__()
        # Define the neural network architecture
        self.model = nn.Sequential(
            nn.Linear(stateDim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, actionDim) # Outputs Q-values for each action
        )

    def forward(self, x):
        return self.model(x)

# Get state and action dimensions from the environment
stateDim = cartPoleEnv.observation_space.shape[0]
actionDim = cartPoleEnv.action_space.n
# Create an instance of the Q-Network
qNetwork = QNetwork(stateDim, actionDim)

# Parameters
ALPHA = 0.005       # Learning rate
EPSILON = 1.0       # Initial exploration rate
EPSILONDECAY = 1.005 # Rate at which exploration decreases
GAMMA = 0.99        # Discount factor for future rewards
NUMEPISODES = 500   # Total number of episodes to train

# Policy function (epsilon-greedy)
def policy(state, explore=0.0):
    with torch.no_grad():
        # Get Q-values from the network for the given state
        qValues = qNetwork(state)
        # Choose the action with the highest Q-value (exploit)
        action = torch.argmax(qValues[0]).item()
        # print(f"Q-Value: {qValues} | Action: {action}") # For Visualization purpose.
    
    # Exploration logic
    if torch.rand(1).item() <= explore:
        # Choose a random action (explore)
        action = torch.randint(0, actionDim, (1,)).item()
    return action


# --- Main Training Loop ---
for episode in range(NUMEPISODES):
    # Reset environment for a new episode
    state, _ = cartPoleEnv.reset()
    # Convert initial state (S) to a PyTorch tensor with a batch dimension
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0) 
    done = False
    totalReward = 0
    episodeLength = 0
    
    # 1. Choose first action (A) based on initial state (S) using the policy
    action = policy(state, EPSILON)

    while not done:
        # 2. Take action (A), get Reward (R) and Next State (S')
        nextState, reward, terminated, truncated, _ = cartPoleEnv.step(action)
        done = terminated or truncated
        nextState = torch.tensor(nextState, dtype=torch.float32).unsqueeze(0)
        
        # 3. Choose Next Action (A') from Next State (S') using the policy
        # This is the key step for Sarsa (on-policy)
        nextAction = policy(nextState, EPSILON)

        # 4. Compute the Sarsa Target value: R + gamma * Q(S', A')
        with torch.no_grad():
            # Get the Q-value of the *next* state and *next* action
            nextQ = qNetwork(nextState)[0][nextAction]
            target = reward + (0 if done else GAMMA * nextQ)

        # 5. Compute prediction and loss
        # Prediction is the Q-value for the *current* state (S) and action (A)
        qValues = qNetwork(state)
        currentQ = qValues[0][action]
        # Loss is the difference between the target and the prediction
        loss = (target - currentQ) ** 2 / 2

        # 6. Manual gradient update
        qNetwork.zero_grad() # Clear old gradients
        loss.backward()      # Calculate new gradients
        with torch.no_grad():
            for param in qNetwork.parameters():
                # Update parameters using gradient DESCENT
                param -= ALPHA * param.grad 

        # 7. Transition: S becomes S', A becomes A'
        state = nextState
        action = nextAction
        
        totalReward += reward
        episodeLength += 1

    # Print episode stats
    print(f"Episode: {episode+1:4d} | Length: {episodeLength:4d} | Reward: {totalReward:6.3f} | Epsilon: {EPSILON:6.3f}")
    
    # Decay epsilon for the next episode (reduce exploration over time)
    if EPSILON > 0.01: # Set a minimum epsilon
        EPSILON /= EPSILONDECAY

# Save the trained model's weights
torch.save(qNetwork.state_dict(), "SarsaQNet.pt")
cartPoleEnv.close()