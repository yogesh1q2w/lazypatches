import random
import torch

class RandomSampler:
    def __init__(self, **kwargs):
        self.drop_probability = kwargs.get("drop_probability", 0.5)
        
    def sample(self, hidden_states):
        if random.random() < self.drop_probability:
            return None
        return hidden_states
    
    def __call__(self, hidden_states):
        return self.sample(hidden_states)
            
