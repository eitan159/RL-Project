import sys
import DQN, DDQN, dueling_DQN, rainbow_DQN, TD3_FORK
import gym
import torch
import pathlib
from eval import eval_agent, eval_agent_hardcore

model_name = str(sys.argv[1])
device_num = str(sys.argv[2])
device = torch.device("cuda:" + device_num if torch.cuda.is_available() else "cpu")

if model_name != "TD3_FORK":
    env = gym.make('LunarLanderContinuous-v2')
    n_state_params = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
else:
    env = gym.make('BipedalWalkerHardcore-v3')
    upper_bound = env.action_space.high[0] 
    lower_bound = env.action_space.low[0]
    obs_upper_bound = env.observation_space.high[0]
    obs_lower_bound = env.observation_space.low[0]
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

env.seed(0)    

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

PATH = str(pathlib.Path(__file__).parent.resolve()) + '/saved_models'

if model_name == "DQN":
    agent = DQN.Agent(n_state_params, n_actions, 0, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY, device)

elif model_name == "DDQN":
    agent = DDQN.Agent(n_state_params, n_actions, 0, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY, device)

elif model_name == "Dueling_DQN":
    agent = dueling_DQN.Agent(n_state_params, n_actions, 0, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY, device)

elif model_name == "Rainbow_DQN":
    LR = 1e-4
    agent = rainbow_DQN.Agent(n_state_params, n_actions, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, UPDATE_EVERY, ATOM_SIZE, device)

elif model_name == "TD3_FORK":
    agent = TD3_FORK.TD3_FORK_model(state_size, action_size, upper_bound, lower_bound
                        ,obs_upper_bound, obs_lower_bound, cuda=device_num)
else:
    raise Exception("Model type did not chose!!!")

if model_name == "TD3_FORK":
    agent.load_weight(PATH + "/TD3_FORK")
    mean_score = eval_agent_hardcore(agent, env, 88, 100)
else:
    agent.qnetwork_local.load_state_dict(torch.load(PATH + "/" + model_name + "/local_best.pth"))
    agent.qnetwork_target.load_state_dict(torch.load(PATH + "/" + model_name + "/target_best.pth"))
    mean_score = eval_agent(agent, env)

    print("---------------------------------------")
    print(f"Evaluation over 100 episodes: {mean_score:.3f}")
    print("---------------------------------------")