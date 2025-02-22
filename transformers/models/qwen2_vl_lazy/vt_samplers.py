import numpy as np
import torch
import abc

class Sampler(abc.ABC):
    def __init__(self, sampler_name):
        self.sampler_name = sampler_name

    @abc.abstractmethod
    def sample(self, hidden_states):
        pass

    def __call__(self, hidden_states, video_mask, position_ids):
        return self.sample(hidden_states, video_mask, position_ids)
    

class UniformSampler(Sampler):
    def __init__(self, config):
        super().__init__("uniform")
        self.retain_proportion = config.retain_proportion
    
    
    def sample(self, hidden_states, video_mask, position_ids):
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
    
class TemporalHeuristicSampler(Sampler):
    def __init__(self, config):
        super().__init__("temporal_heuristic")
        self.retain_proportion = config.retain_proportion
        self.drop_start = config.drop_start  # Proportion of early frames to drop
        self.drop_end = config.drop_end  # Proportion of late frames to drop
    
    def sample(self, hidden_states, video_mask, position_ids):
        batch_size, seq_len, embed_dim = hidden_states.shape
        sampling_mask = torch.ones_like(video_mask, dtype=torch.bool)
        
        temporal_positions = position_ids[0]  # Extract temporal indices
        max_time = temporal_positions.max().item()
        
        for b in range(batch_size):
            valid_indices = torch.nonzero(video_mask[b].flatten(), as_tuple=True)[0]
            valid_indices_2D = torch.nonzero(video_mask[b], as_tuple=True)  # Returns (row_indices, col_indices)
            valid_temporal_indices = valid_indices_2D[0]  # Extract only the row indices (temporal positions)
            temporal_vals = temporal_positions[0, valid_temporal_indices]

            drop_start_idx = int(self.drop_start * max_time)
            drop_end_idx = int((1 - self.drop_end) * max_time)
            
            drop_mask = (temporal_vals <= drop_start_idx) | (temporal_vals >= drop_end_idx)
            drop_indices = valid_indices[drop_mask.nonzero(as_tuple=True)[0]]
            
            flat_sampling_mask = sampling_mask[b].flatten()
            flat_sampling_mask[drop_indices] = False
            sampling_mask[b] = flat_sampling_mask.view(video_mask[b].shape)
        
        hidden_states = hidden_states * sampling_mask.float()
        video_mask = video_mask & sampling_mask
        return hidden_states, video_mask, sampling_mask

class SpatialHeuristicSampler(Sampler):
    def __init__(self, config):
        super().__init__("spatial_heuristic")
        self.retain_proportion = config.retain_proportion
        self.periphery_ratio = config.periphery_ratio  # Proportion of peripheral patches to drop
    
    def sample(self, hidden_states, video_mask, position_ids):
        batch_size, seq_len, embed_dim = hidden_states.shape
        sampling_mask = torch.ones_like(video_mask, dtype=torch.bool)
        
        height_positions = position_ids[1]
        width_positions = position_ids[2]
        
        max_h, max_w = height_positions.max().item(), width_positions.max().item()
        periphery_h = int(self.periphery_ratio * max_h)
        periphery_w = int(self.periphery_ratio * max_w)
        
        for b in range(batch_size):
            valid_indices = torch.nonzero(video_mask[b].flatten(), as_tuple=True)[0]
            valid_indices_2D = torch.nonzero(video_mask[b], as_tuple=True)
            valid_hw_indices = valid_indices_2D[0]
            
            h_vals = height_positions[0, valid_hw_indices]
            w_vals = width_positions[0, valid_hw_indices]
            
            drop_mask = (h_vals <= periphery_h) | (h_vals >= max_h - periphery_h) | \
                        (w_vals <= periphery_w) | (w_vals >= max_w - periphery_w)
            drop_indices = valid_indices[drop_mask]
            
            flat_sampling_mask = sampling_mask[b].flatten()
            flat_sampling_mask[drop_indices] = False
            sampling_mask[b] = flat_sampling_mask.view(video_mask[b].shape)
        
        hidden_states = hidden_states * sampling_mask.float()
        video_mask = video_mask & sampling_mask
        return hidden_states, video_mask, sampling_mask
