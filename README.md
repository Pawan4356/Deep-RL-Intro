# Reinforcement Learning with Function Approximation

### *SARSA & Q-Learning on CartPole-v1 using Deep RL*



## Overview

This project demonstrates **Reinforcement Learning (RL)** using **Function Approximation** with **Neural Networks (NN)** instead of traditional tabular methods.

The agent learns to balance a pole on a moving cart using **SARSA** and **Q-Learning**, two popular **Temporal Difference (TD)** algorithms.



## Core Concepts Covered

### 1. **From Tabular to Function Approximation**

* **Tabular methods** (like Q-tables) are not feasible for large or continuous state spaces.
* **Neural Networks** approximate the **Q-value function**:
  $Q(s, a; \theta) \approx \text{True Q}(s, a)$


Benefits:

* Handles **large/continuous** state spaces
* **Lower memory** usage
* **Generalizes** across similar states



### 2. **Semi-Gradient Methods**

Used when targets depend on the same parameters being updated (bootstrapping).
The gradient is computed **partially**—hence *semi-gradient*.

Two key forms:

* **Monte Carlo Evaluation** – Uses complete episode returns.
* **TD(0) Evaluation** – Updates online using one-step returns.



### 3. **Stochastic Gradient Descent (SGD)**

* The network parameters are updated incrementally:
  
  $\theta \leftarrow \theta + \alpha \cdot (target - prediction) \cdot \nabla_\theta Q(s, a)
  $
* Targets are **noisy** (from experience samples), leading to *stochastic* gradients.



## $ Implemented Algorithms

### **1. SARSA (On-Policy)**

Learns from actions **actually taken** following the current policy:
$Q(s, a) \leftarrow Q(s, a) + \alpha \cdot (r + \gamma Q(s', a')) - Q(s, a)$

File: `sarsa.py`



### **2. Q-Learning (Off-Policy)**

Learns from the **greedy action**, even if not taken:

$Q(s, a) \leftarrow Q(s, a) + \alpha \cdot [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$

File: `qlearning.py`



## Problems in the code with Current Approach

| Issue| Description|
| - | - |
| **1. Non-linear Function Approximator** | Convergence is **not guaranteed** when using NNs.                            |
| **2. High Correlation in Samples**      | Consecutive states are highly correlated → violates i.i.d. assumption.       |
| **3. One Experience per Update**        | Using **single samples** slows learning; **mini-batches** improve stability. |
| **4. Sample Inefficiency**              | Experiences are **used once** and discarded (no replay buffer).              |

 *These issues are later addressed by techniques like Experience Replay and Target Networks in Deep Q-Learning (DQN).*



## Code Summary

### `randomagent.py`

A simple baseline that takes random actions to test the environment rendering.

### `sarsa.py`

Implements SARSA using a **3-layer fully-connected neural network** for Q-function approximation.
Manual gradient ascent is used for parameter updates.

### `qlearning.py`

Almost identical to `sarsa.py`, but updates Q-values using the **maximum** predicted value from the next state.

### `evaluation.py`

Loads a trained model (`.pt` file) and visualizes agent performance using OpenCV.



## How to Run


### Create a Virtual Environment

Make sure Python **3.10.9** is installed. Then, open your terminal inside the project folder and run:

```bash
# Create virtual environment
python3.10 -m venv .venv
```

### Activate it

```bash
venv\Scripts\activate
```

You should now see `(venv)` in your terminal prompt.

Once your venv is activated:

```bash
pip install -r requirements.txt
```

### Train an Agent

```bash
# SARSA
python sarsa.py

# or Q-Learning
python qlearning.py
```

### Evaluate a Trained Model

```bash
python evaluation.py
```

A window will open showing the **CartPole** environment in action.



## Results

* The agent learns to keep the pole upright for **increasingly longer durations**.
* SARSA tends to be **more stable** (on-policy).
* Q-Learning can achieve **higher peak performance** (off-policy).



## Key Learnings

* Function approximation enables RL on **continuous** or **high-dimensional** problems.
* Neural networks introduce **instability** and **non-deterministic behavior**.
* Modern RL algorithms (like **DQN**, **DDPG**, **PPO**) extend these foundations.



## References

* **Sutton & Barto (2020)** – *Reinforcement Learning: An Introduction*
* OpenAI Gymnasium Documentation



***Author - Pawankumar Navinchandra***
