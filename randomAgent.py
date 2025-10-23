import cv2
import torch
import gymnasium as gym

cartPoleEnv = gym.make("CartPole-v1", render_mode="rgb_array")

for episode in range(5):
    done = False
    state, _ = cartPoleEnv.reset()
    while not done:
        frame = cartPoleEnv.render()
        cv2.imshow("Cart Pole Simulation", frame)
        cv2.waitKey(100)
        action = torch.randint(0, 2, (1,)).item()
        state, reward, done, _, info = cartPoleEnv.step(action)
    print(f"Episode: {episode+1:3d}")

cartPoleEnv.close()