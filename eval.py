import gym
import numpy as np
from collections import deque
import torch

def eval_agent(agent, env):
    #env = gym.make('LunarLanderContinuous-v2')
    eval_env = env
    eval_env.seed(100)
    last_scores = deque(maxlen=100)
    for _ in range(100):
        state = eval_env.reset()
        total_reward = 0
        while True:
            action = agent.act(state, 0, mode='test')
            next_state, reward, done, _ = eval_env.step(action)
            state = next_state
            total_reward += reward
            if done:
                break

        last_scores.append(total_reward)
    
    mean_score = np.mean(last_scores)
    eval_env.close()
    return mean_score

def eval_agent_hardcore(agent, env, seed, eval_episodes):
    eval_env = env
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = agent.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    
    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward