import numpy as np
import torch
import abc

class Sampler(abc.ABC):
    def __init__(self, sampler_name):
        self.sampler_name = sampler_name

    @abc.abstractmethod
    def sample(self, hidden_states):
        pass

    def __call__(self, hidden_states, video_mask):
        return self.sample(hidden_states, video_mask)
    

class UniformSampler(Sampler):
    def __init__(self, config):
        super().__init__("uniform")
        self.retain_proportion = config.retain_proportion
    
    
    def sample(self, hidden_states, video_mask):
        batch_size, seq_len, embed_dim = hidden_states.shape

        # Initialize sampling_mask with the same dimensions as video_mask
        sampling_mask = torch.ones_like(video_mask, dtype=torch.bool)  # Shape: (batch_size, seq_len, embed_dim)

        # Iterate over the batch to randomly drop (1 - retain_proportion) values where video_mask is True
        for b in range(batch_size):
            # Get the indices where video_mask is True for the current batch
            true_indices = torch.nonzero(video_mask[b].flatten(), as_tuple=True)[0]  # Shape: (num_valid_positions,)
            num_to_drop = int((1 - self.retain_proportion) * true_indices.size(0))  # Number of values to drop

            # Randomly select indices to drop
            drop_indices = true_indices[torch.randperm(true_indices.size(0))[:num_to_drop]]

            # Flatten the sampling_mask for the current batch, set selected indices to False
            flat_sampling_mask = sampling_mask[b].flatten()
            flat_sampling_mask[drop_indices] = False

            # Reshape back to original shape
            sampling_mask[b] = flat_sampling_mask.view(video_mask[b].shape)

        # Apply the sampling mask to hidden_states directly
        hidden_states = hidden_states * sampling_mask.float()
        video_mask = video_mask & sampling_mask

        return hidden_states, video_mask, sampling_mask