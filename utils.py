from collections import deque
import numpy as np
import matplotlib.pyplot as plt

def plot(data, label_x, label_y, title, save_path):
    fig = plt.figure()
    x = np.arange(1, len(data) + 1)
    y = data
    if save_path.split("/")[-1] == 'episodes_mean_score.jpg':
        x *= 50
    plt.plot(x, y)
    plt.ylabel(label_y)
    plt.xlabel(label_x)
    plt.title(title)
    fig.savefig(save_path)


class ReplayBuffer:
    def __init__(self, max_size=5e6):
        self.buffer = deque(maxlen = max_size)
        self.max_size = int(max_size)
        self.size = 0
    
    def add(self, transition):
        # transiton is tuple of (state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []
        
        for i in indexes:
            s, a, r, s_, d = self.buffer[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))
        
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    
    def __len__(self):
        return len(self.buffer)