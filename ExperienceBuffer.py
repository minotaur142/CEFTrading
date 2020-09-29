import collections
import numpy as np
import pandas as pd

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states = np.array([buffer.buffer[ind][0] for ind in indices])
        actions = np.array([buffer.buffer[ind][1] for ind in indices])
        rewards = np.array([buffer.buffer[ind][2] for ind in indices])
        dones = np.array([buffer.buffer[ind][3] for ind in indices])
        next_states = np.array([buffer.buffer[ind][4] for ind in indices])
        return states, actions, rewards, dones, next_states