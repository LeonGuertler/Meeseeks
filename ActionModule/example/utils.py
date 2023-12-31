from collections import namedtuple 
import random, torch 
import numpy as np 


Transition = namedtuple('Transion', 
                        ('state', 'action', 'reward', 'next_state', 'done', 'task_desc'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        # Converts batch of transitions to transitions of batches
        batch = Transition(*zip(*batch))
        return batch

    def __len__(self):
        return len(self.memory)
    
def process_state(obs):
    state = np.array(obs)
    state = torch.from_numpy(state)
    return state.unsqueeze(0)