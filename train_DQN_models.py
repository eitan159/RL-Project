import gym
import numpy as np
from collections import deque
import DQN, DDQN, dueling_DQN, rainbow_DQN
import torch
import sys
from eval import eval_agent
from utils import plot

env = gym.make('LunarLanderContinuous-v2')
env.seed(0)
n_actions = env.action_space.shape[0]

BATCH_SIZE = 64
MAX_EPISODES = 10000
MAX_REWARD = 200
MAX_STEPS = 2000        #env._max_episode_steps
BUFFER_SIZE = int(1e5)  # replay buffer size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-3               # learning rate
UPDATE_EVERY = 4        # how often to update the network
MEAN_EVERY = 100
eps = 0.99
EPSILON_DECAY = 0.001
EPSILON_MIN = 0.001
ATOM_SIZE = 51
EVAL_EVERY = 50

model_name = str(sys.argv[1])
n_state_params = env.observation_space.shape[0]
device = torch.device("cuda:" + str(sys.argv[2]) if torch.cuda.is_available() else "cpu")

if model_name == "DQN":
    agent = DQN.Agent(n_state_params, n_actions, 0, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY, device)

elif model_name == "DDQN":
    agent = DDQN.Agent(n_state_params, n_actions, 0, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY, device)

elif model_name == "Dueling_DQN":
    agent = dueling_DQN.Agent(n_state_params, n_actions, 0, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY, device)

elif model_name == "Rainbow_DQN":
    #LR = 9e-4
    agent = rainbow_DQN.Agent(n_state_params, n_actions, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, UPDATE_EVERY, ATOM_SIZE, device)
else:
    raise Exception("Model type did not chose!!!")
    
losses_mean_episode = []
mean_scores = []
for ep in range(1, MAX_EPISODES + 1):
    state = env.reset()
    losses = []
    while True: #for t in range(MAX_STEPS)
        action = agent.act(state, eps)
        next_state, reward, done, _ = env.step(action)
        loss = agent.step(state, action, reward, next_state, done)
        if loss is not None:
            losses.append(loss)
        state = next_state
        if done:
            break
    eps = max(EPSILON_MIN, EPSILON_DECAY * eps)
    if len(losses) >= 1:
        mean_loss = np.mean(losses)
        losses_mean_episode.append(mean_loss)
    
    if ((ep % EVAL_EVERY) == 0):
        mean_score = eval_agent(agent, env)
        mean_scores.append(mean_score)
        print('\nEnvironment got average Score: {:.2f} \tin {:d} episodes.'.format(mean_score, ep))
        if mean_score >= 200:
            torch.save(agent.qnetwork_local.state_dict(), "/home/dsi/davidc/RL/saved_models/" + model_name + "/local_best.pth")
            torch.save(agent.qnetwork_target.state_dict(), "/home/dsi/davidc/RL/saved_models/" + model_name + "/target_best.pth")
            print("Solved !!!")
            break

env.close()
plot(losses_mean_episode, "Episode", "Agent Loss", "Agent Loss on LunarLanderContinuous-v2", "/home/dsi/davidc/RL/saved_models/" + model_name + "/episodes_loss.jpg")
plot(mean_scores, "Episode", "Agent mean reward", "Agent mean reward every 100 episodes on LunarLanderContinuous-v2", "/home/dsi/davidc/RL/saved_models/" + model_name + "/episodes_mean_score.jpg")