# Reinforcement Learning with Function Approximation

### *SARSA & Q-Learning on CartPole-v1 using Deep RL*

### **SARSA (On-Policy)**

SARSA (State–Action–Reward–State–Action) is an **on-policy** reinforcement learning algorithm, meaning it learns from the actions actually taken under the current policy. The update rule uses the next action chosen by the same policy, making learning more **stable** but sometimes slower to converge. When combined with a neural network as a function approximator, SARSA estimates Q-values for state–action pairs and updates its parameters using the semi-gradient TD(0) method. This results in smoother learning but can still be affected by correlated samples and high variance in returns.

### **Q-Learning (Off-Policy)**

Q-Learning is an **off-policy** algorithm that learns from the **greedy** action (the best predicted action), even if a different one was taken in the environment. This approach allows it to converge toward the optimal policy more quickly, often achieving higher peak performance than SARSA. When implemented with a neural network, Q-Learning uses the maximum Q-value of the next state as its target for updates. While more aggressive, this can make training less stable—especially without techniques like experience replay or target networks that help stabilize learning.

### CartPole Results



## References

* **Sutton & Barto (2020)** – *Reinforcement Learning: An Introduction*
* OpenAI Gymnasium Documentation



***Author - Pawankumar Navinchandra***
