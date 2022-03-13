import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from collections import namedtuple, deque
import torch.optim as optim
from utils import ReplayBuffer

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc_units=400, fc1_units=300):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, fc1_units)
        self.fc3 = nn.Linear(fc1_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Q1 architecture
        self.l1 = nn.Linear(state_size + action_size, fc1_units)
        self.l2 = nn.Linear(fc1_units, fc2_units)
        self.l3 = nn.Linear(fc2_units, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_size + action_size, fc1_units)
        self.l5 = nn.Linear(fc1_units, fc2_units)
        self.l6 = nn.Linear(fc2_units, 1)

    def forward(self, state, action):
        xa = torch.cat([state, action], 1)

        x1 = F.relu(self.l1(xa))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xa))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)

        return x1, x2

class SysModel(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=400, fc2_units=300):
        super(SysModel, self).__init__()
        self.l1 = nn.Linear(state_size + action_size, fc1_units)
        self.l2 = nn.Linear(fc1_units, fc2_units)
        self.l3 = nn.Linear(fc2_units, state_size)


    def forward(self, state, action):
        xa = torch.cat([state, action], 1)

        x1 = F.relu(self.l1(xa))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        return x1

class TD3_FORK_model:
    def __init__(
        self,state_size, action_size, upper_bound, lower_bound, obs_upper_bound, obs_lower_bound,
        load = False,
        gamma = 0.99, #discount factor
        lr_actor = 3e-4,
        lr_critic = 3e-4,
        lr_sysmodel = 3e-4,
        batch_size = 100,
        buffer_capacity = 1000000,
        tau = 0.02,  #target network update factor
        cuda = 0,
        policy_noise=0.2, 
        std_noise = 0.1,
        noise_clip=0.5,
        policy_freq=2 #target network update period
    ):
        self.device = torch.device("cuda:" + str(cuda) if torch.cuda.is_available() else "cpu")
        #self.env = env
        self.action_size = action_size
        
        self.actor = Actor(state_size, action_size, 88).to(self.device)
        self.actor_target = Actor(state_size, action_size, 88).to(self.device)
        self.critic = Critic(state_size, action_size, 88).to(self.device)
        self.critic_target = Critic(state_size, action_size, 88).to(self.device)
        self.sysmodel = SysModel(state_size, action_size).to(self.device)
        
        self.act_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.crt_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.sys_opt = optim.Adam(self.sysmodel.parameters(), lr=lr_sysmodel)
        
        self.set_weights()
        self.replay_memory_buffer = ReplayBuffer(max_size = buffer_capacity)#deque(maxlen = buffer_capacity)
        self.batch_size = batch_size
        self.tau = tau
        self.policy_freq = policy_freq
        self.gamma = gamma
        self.upper_bound = upper_bound #action space upper bound
        self.lower_bound = lower_bound  #action space lower bound
        self.obs_upper_bound = obs_upper_bound #state space upper bound
        self.obs_lower_bound = obs_lower_bound #state space lower bound
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.std_noise = std_noise   
 
    def set_weights(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def load_weight(self, directory):
        self.actor.load_state_dict(torch.load(directory + '/actor.pth', map_location=self.device))
        self.critic.load_state_dict(torch.load(directory + '/critic.pth', map_location=self.device))
        self.actor_target.load_state_dict(torch.load(directory + '/actor_t.pth', map_location=self.device))
        self.critic_target.load_state_dict(torch.load(directory + '/critic_t.pth', map_location=self.device))
        self.sysmodel.load_state_dict(torch.load(directory + '/sysmodel.pth', map_location=self.device))

    def save(self, directory):
        torch.save(self.actor.state_dict(), directory + '/actor.pth')
        torch.save(self.critic.state_dict(), directory + '/critic.pth')
        torch.save(self.actor_target.state_dict(), directory + '/actor_t.pth')
        torch.save(self.critic_target.state_dict(), directory + '/critic_t.pth')
        torch.save(self.sysmodel.state_dict(), directory + '/sysmodel.pth')

    def add_to_replay_memory(self, transition, buffername):
        #add samples to replay memory
        buffername.append(transition)

    # def get_random_sample_from_replay_mem(self, buffername):
    #     #random samples from replay memory
    #     random_sample = random.sample(buffername, self.batch_size)
    #     return random_sample

    def learn_and_update_weights_by_replay(self,training_iterations, weight, totrain):
        if len(self.replay_memory_buffer) < 1e4:
            return 1
        for it in range(training_iterations):
            # mini_batch = self.get_random_sample_from_replay_mem(self.replay_memory_buffer)
            # state_batch = torch.from_numpy(np.vstack([i[0] for i in mini_batch])).float().to(self.device)
            # action_batch = torch.from_numpy(np.vstack([i[1] for i in mini_batch])).float().to(self.device)
            # reward_batch = torch.from_numpy(np.vstack([i[2] for i in mini_batch])).float().to(self.device)
            # add_reward_batch = torch.from_numpy(np.vstack([i[3] for i in mini_batch])).float().to(self.device)
            # next_state_batch = torch.from_numpy(np.vstack([i[4] for i in mini_batch])).float().to(self.device)
            # done_list = torch.from_numpy(np.vstack([i[5] for i in mini_batch]).astype(np.uint8)).float().to(self.device)
            state_batch, action_batch, reward_batch, next_state_batch, done_list = self.replay_memory_buffer.sample(self.batch_size)         
            state_batch = torch.from_numpy(state_batch).float().to(self.device)
            action_batch = torch.from_numpy(action_batch).float().to(self.device)
            reward_batch = torch.from_numpy(reward_batch).unsqueeze(1).float().to(self.device)
            next_state_batch = torch.from_numpy(next_state_batch).float().to(self.device)
            done_list = torch.from_numpy(done_list.astype(np.uint8)).unsqueeze(1).float().to(self.device)

            # Training and updating Actor & Critic networks.
            
            #Train Critic
            target_actions = self.actor_target(next_state_batch)
            offset_noises = torch.FloatTensor(action_batch.shape).data.normal_(0, self.policy_noise).to(self.device)

            #clip noise
            offset_noises = offset_noises.clamp(-self.noise_clip, self.noise_clip)
            target_actions = (target_actions + offset_noises).clamp(self.lower_bound, self.upper_bound)

            #Compute the target Q value
            Q_targets1, Q_targets2 = self.critic_target(next_state_batch, target_actions)
            Q_targets = torch.min(Q_targets1, Q_targets2)
            Q_targets = reward_batch + self.gamma * Q_targets * (1 - done_list)

            #Compute current Q estimates
            current_Q1, current_Q2 = self.critic(state_batch, action_batch)
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, Q_targets.detach()) + F.mse_loss(current_Q2, Q_targets.detach())
            # Optimize the critic
            self.crt_opt.zero_grad()
            critic_loss.backward()
            self.crt_opt.step()

            self.soft_update_target(self.critic, self.critic_target)

            #Train_sysmodel
            predict_next_state = self.sysmodel(state_batch, action_batch) * (1-done_list)
            next_state_batch = next_state_batch * (1 -done_list)
            sysmodel_loss = F.mse_loss(predict_next_state, next_state_batch.detach())
            self.sys_opt.zero_grad()
            sysmodel_loss.backward()
            self.sys_opt.step()
        
            s_flag = 1 if sysmodel_loss.item() < 0.020  else 0

            #Train Actor
            # Delayed policy updates
            if it % self.policy_freq == 0 and totrain == 1:
                actions = self.actor(state_batch)
                actor_loss1,_ = self.critic_target(state_batch, actions)
                actor_loss1 =  actor_loss1.mean()
                actor_loss =  - actor_loss1 

                if s_flag == 1:
                    p_actions = self.actor(state_batch)
                    p_next_state = self.sysmodel(state_batch, p_actions).clamp(self.obs_lower_bound,self.obs_upper_bound)

                    p_actions2 = self.actor(p_next_state.detach()) * self.upper_bound
                    actor_loss2,_ = self.critic_target(p_next_state.detach(), p_actions2)
                    actor_loss2 = actor_loss2.mean() 

                    p_next_state2= self.sysmodel(p_next_state.detach(), p_actions2).clamp(self.obs_lower_bound,self.obs_upper_bound)
                    p_actions3 = self.actor(p_next_state2.detach()) * self.upper_bound
                    actor_loss3,_ = self.critic_target(p_next_state2.detach(), p_actions3)
                    actor_loss3 = actor_loss3.mean() 

                    actor_loss_final =  actor_loss - weight * (actor_loss2) - 0.5 *  weight * actor_loss3
                else:
                    actor_loss_final =  actor_loss

                self.act_opt.zero_grad()
                actor_loss_final.backward()
                self.act_opt.step()
               
                self.soft_update_target(self.actor, self.actor_target)
                
        return sysmodel_loss.item()

    def soft_update_target(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def policy(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(state).cpu().data.numpy()
        self.actor.train()
        # Adding noise to action
        shift_action = np.random.normal(0, self.std_noise, size=self.action_size)
        sampled_actions = (actions + shift_action)
        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions,self.lower_bound,self.upper_bound)
        return np.squeeze(legal_action)

    def select_action(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            actions = self.actor_target(state).cpu().data.numpy()
        return np.squeeze(actions)
