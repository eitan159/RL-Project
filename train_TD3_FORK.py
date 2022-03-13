from TD3_FORK import TD3_FORK_model
import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from eval import eval_agent_hardcore
import sys

max_steps = 3000
falling_down = 0
env = gym.make('BipedalWalkerHardcore-v3')
upper_bound = env.action_space.high[0] 
lower_bound = env.action_space.low[0]
obs_upper_bound = env.observation_space.high[0]
obs_lower_bound = env.observation_space.low[0]
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

gpu_num = int(sys.argv[1])

agent = TD3_FORK_model(state_size, action_size, upper_bound, lower_bound
                        ,obs_upper_bound, obs_lower_bound, cuda=gpu_num)
total_episodes = 5000
start_timestep = 0           
time_start = time.time()      
ep_reward_list = []
avg_reward_list = []
total_timesteps = 0
sys_loss = 0
expcount = 0
totrain = 0
directory = "/home/lab/eitanshaar/RL/saved_models/TD3"

for ep in range(total_episodes):
    state = env.reset()
    episodic_reward = 0
    timestep = 0
    temp_replay_buffer = []
    for st in range(max_steps):
        # Select action randomly or according to policy
        if total_timesteps < start_timestep:
            action = env.action_space.sample()
        else:
            action = agent.policy(state)

        # Recieve state and reward from environment.
        next_state, reward, done, info = env.step(action)
        #change original reward from -100 to -5 and 5*reward for other values
        episodic_reward += reward
        if reward == -100:
            add_reward = -1
            reward = -5
            falling_down += 1
            expcount += 1
        else:
            add_reward = 0
            reward = 5 * reward

        temp_replay_buffer.append((state, action, reward, next_state, done))
        
        # End this episode when `done` is True
        if done:
            if add_reward == -1 or episodic_reward < 250:            
                totrain = 1
                for temp in temp_replay_buffer: 
                    agent.add_to_replay_memory(temp, agent.replay_memory_buffer)
            elif expcount > 0 and np.random.rand() > 0.5:
                totrain = 1
                expcount -= 10
                for temp in temp_replay_buffer: 
                    agent.add_to_replay_memory(temp, agent.replay_memory_buffer)
            break
        state = next_state
        timestep += 1     
        total_timesteps += 1

    ep_reward_list.append(episodic_reward)
    # Mean of last 100 episodes
    avg_reward = np.mean(ep_reward_list[-100:])
    avg_reward_list.append(avg_reward)

    if avg_reward > 294:
        test_reward = eval_agent_hardcore(agent, env, seed=88, eval_episodes=10)
        if test_reward > 300:
            final_test_reward = eval_agent_hardcore(agent, env, seed=88, eval_episodes=100)
            if final_test_reward > 300:
                agent.save(directory)
                print("===========================")
                print('Task Solved')
                print("===========================")
                break
                    
    s = (int)(time.time() - time_start)

    #Training agent only when new experiences are added to the replay buffer
    weight =  1 - np.clip(np.mean(ep_reward_list[-100:])/300, 0, 1)
    if totrain == 1:
        sys_loss = agent.learn_and_update_weights_by_replay(timestep, weight, totrain)
    else: 
        sys_loss = agent.learn_and_update_weights_by_replay(100, weight, totrain)
    totrain = 0

    print('Ep. {}, Timestep {},  Ep.Timesteps {}, Episode Reward: {:.2f}, Moving Avg.Reward: {:.2f}, Time: {:02}:{:02}:{:02} , Falling down: {}, Weight: {}'
            .format(ep, total_timesteps, timestep,
                    episodic_reward, avg_reward, s//3600, s%3600//60, s%60, falling_down, weight)) 
    
# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.title("Agent mean reward every episode on BipedalWalkerHardcore-v3")
plt.savefig(directory + "/plot.png")
env.close()