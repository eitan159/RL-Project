import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import ReplayBuffer

class DuelingDQN(nn.Module):

    def __init__(self, state_size, action_size, num_opt, seed):
        super(DuelingDQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 256)
        self.v_fc1 = nn.Linear(256, 256)
        self.v_fc2 = nn.Linear(256, 1)

        self.a_fc1 = nn.Linear(256, 256)
        self.a_fc2 = nn.Linear(256, num_opt ** action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        
        val = F.relu(self.v_fc1(x))
        val = self.v_fc2(val)
        
        adv = F.relu(self.a_fc1(x))
        adv = self.a_fc2(adv)

        # Q = V + A(a) - 1/|A| * sum A(a')
        output = val + adv - adv.mean(dim=1, keepdim=True)
        return output

class Agent():

    def __init__(self, state_size, action_size, seed,
                 buffer_size, batch_size, gamma,
                 tau, learning_rate, update_every, device):
        self.num_opt = 9
        self.batch_size = batch_size
        self.gamma = gamma
        self.state_size = state_size
        self.n_actions = self.num_opt ** action_size
        self.tau = tau
        self.update_every = update_every
        self.seed = random.seed(seed)
        self.device = device

        self.qnetwork_local = DuelingDQN(state_size, action_size, self.num_opt, seed).to(device)
        self.qnetwork_target = DuelingDQN(state_size, action_size, self.num_opt, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        self.memory = ReplayBuffer(max_size=buffer_size)
        self.t_step = 0
        self.action2idx = self.action_to_idx()

    def get_action(self, action_idx):
        action = []
        # 1
        output = int(action_idx / self.num_opt)
        rest = action_idx - self.num_opt * int(action_idx / self.num_opt)
        action.append(-1 + output * (2 / (self.num_opt - 1)))
        # 2
        action.append(-1 + (rest) * (2 / (self.num_opt - 1)))

        return action
    
    def action_to_idx(self):
        action2idx = {}
        for i in range(self.n_actions):
            action2idx[tuple(self.get_action(i))] = i
        return action2idx

    def act(self, state, eps, mode='train'):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        if mode == 'train':
            self.qnetwork_local.train()
            if random.random() > eps:
                action_idx = np.argmax(action_values.cpu().data.numpy())
            else:
                action_idx = int(random.randrange(self.n_actions))
        else:
            action_idx = np.argmax(action_values.cpu().data.numpy())
        return self.get_action(action_idx)

    def step(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                return self.learn(experiences, self.gamma)
        return None

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        next_states = torch.as_tensor(next_states).to(self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32).to(self.device)
        states = torch.as_tensor(states).to(self.device)
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).max(1)[0]
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * (1 - dones) * Q_targets_next)

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states)
        actions_idxes = torch.as_tensor([self.action2idx[tuple(action.tolist())] for action in actions]).unsqueeze(1).to(self.device)
        Q_expected = torch.gather(Q_expected , 1, actions_idxes).squeeze()

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        return loss.cpu().data.numpy()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)