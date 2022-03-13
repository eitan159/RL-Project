import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from utils import ReplayBuffer
import random
import torch.optim as optim

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init = 0.5,):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

class RainbowDQN(nn.Module):
    def __init__(self, in_dim, out_dim, atom_size, support):
        super(RainbowDQN, self).__init__()
        
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        self.fc1 = nn.Linear(in_dim, 512)
        
        self.advantage1 = NoisyLinear(512, 512)
        self.advantage2 = NoisyLinear(512, out_dim * atom_size)

        self.value1 =  NoisyLinear(512, 512)
        self.value2 = NoisyLinear(512, atom_size)

    def forward(self, x):
        dist = self.calc_dist(x)
        q = torch.sum(dist * self.support, dim=2)

        return q

    def calc_dist(self, x):
        x = F.relu(self.fc1(x))
        adv = F.relu(self.advantage1(x))
        adv = self.advantage2(adv).view(-1, self.out_dim, self.atom_size)
        value = F.relu(self.value1(x))
        value = self.value2(value).view(-1, 1, self.atom_size)

        # Q(s) = V(s) + A(s,a) - 1/|A| sum(A(s, a'))
        q_atoms = value + adv - adv.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)

        return dist
    
    def reset_noise(self):        
        self.advantage1.reset_noise()
        self.advantage2.reset_noise()

        self.value1.reset_noise()
        self.value2.reset_noise()


class Agent():
    def __init__(self, state_size, action_size, 
                 buffer_size, batch_size, gamma,
                 learning_rate, update_every, atom_size, device, v_min=-100, v_max=100):

        self.num_opt = 9
        self.batch_size = batch_size
        self.gamma = gamma
        self.state_size = state_size
        self.n_actions = self.num_opt ** action_size
        self.update_every = update_every
        self.device = device

        self.memory = ReplayBuffer(max_size=buffer_size)

        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support  = torch.linspace(self.v_min, self.v_max, self.atom_size).to(device)

        self.qnetwork_local = RainbowDQN(state_size, self.n_actions, self.atom_size, self.support).to(device)
        self.qnetwork_target = RainbowDQN(state_size, self.n_actions, self.atom_size, self.support).to(device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval()
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        self.action2idx = self.action_to_idx()
        self.t_step = 0


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
            self.target_hard_update()
       
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
        actions = torch.as_tensor(actions).to(self.device)

        # Get max predicted Q values (for next states) from target model
        
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.qnetwork_local(next_states).argmax(1)
            next_dist = self.qnetwork_target.calc_dist(next_states)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = rewards.reshape(-1,1) + (1 - dones.reshape(-1,1)) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.qnetwork_local.calc_dist(states)
        actions_idxes = torch.as_tensor([self.action2idx[tuple(action.tolist())] for action in actions]).to(self.device)
        log_p = torch.log(dist[range(self.batch_size), actions_idxes])
        loss = -(proj_dist * log_p).sum(1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.qnetwork_local.reset_noise()
        self.qnetwork_target.reset_noise()
        return loss.cpu().data.numpy()
        
    def target_hard_update(self):
        """Hard update: target <- local."""
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        