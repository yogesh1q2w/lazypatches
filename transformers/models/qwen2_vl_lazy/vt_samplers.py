import numpy as np
import torch
import abc
import random

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
        #NEW METHOD:
        # batch_size, seq_len, embed_dim = hidden_states.shape

        # # Initialize the sampling mask with the same dimensions as video_mask
        # sampling_mask = torch.ones_like(video_mask, dtype=torch.bool)  # Shape: (batch_size, seq_len)

        # # Iterate over the batch to randomly drop (1 - retain_proportion) values where video_mask is True
        # for b in range(batch_size):
        #     # Get the indices where video_mask is True for the current batch
        #     true_indices = torch.nonzero(video_mask[b], as_tuple=True)[0]  # Shape: (num_valid_positions,)
        #     num_to_drop = int((1 - self.retain_proportion) * true_indices.size(0))  # Number of values to drop

        #     # Randomly select indices to drop
        #     drop_indices = true_indices[torch.randperm(true_indices.size(0))[:num_to_drop]]

        #     # Set the selected indices in the sampling mask to False
        #     sampling_mask[b, drop_indices] = False
            
        #     # import pdb; pdb.set_trace()

        # # # Expand sampling_mask to match the embedding dimension of hidden_states
        # # sampling_mask_expanded = sampling_mask.unsqueeze(-1).expand(-1, -1, embed_dim)  # Shape: (batch_size, seq_len, embed_dim)

        # # Apply the sampling mask to hidden_states
        # hidden_states = hidden_states * sampling_mask.float()

        # # Update the video_mask to reflect the dropped tokens
        # video_mask = video_mask & sampling_mask

        # return hidden_states, video_mask, sampling_mask

        # OLD METHOD: 
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
        print(f'SAMPLING RATE FOR EXPERIMENT IS {(1-self.retain_proportion)*100}%')
        return hidden_states, video_mask, sampling_mask

        #NEWER METHOD::::::
        # batch_size, seq_len, embed_dim = hidden_states.shape
        # sampling_mask = torch.ones_like(hidden_states, dtype=torch.bool)
        # video_states = hidden_states * video_mask.float()
        # non_video_states = hidden_states * torch.logical_not(video_mask)
        
        # random_mask = [0 ,1]
        # random_wt = [1 - self.retain_proportion, self.retain_proportion]
        # # import pdb; pdb.set_trace()
        # for b in range(batch_size):
        #     for s in range(seq_len):
        #         for e in range(embed_dim):
        #             if video_states[b][s][e] != 0:
        #                 mask_index = random.choices(random_mask, random_wt)[0]
        #                 video_states[b][s][e] = video_states[b][s][e] * mask_index
        #                 video_mask[b][s][e] = video_mask[b][s][e] * mask_index
        #                 sampling_mask[b][s][e] = False

        # updated_hidden_states = video_states + non_video_states

        # return updated_hidden_states, video_mask, sampling_mask

class SpatioTemporalHeuristicSampler(Sampler):
    def __init__(self, config):
        super().__init__("st_gaussian")
        self.retain_proportion = config.retain_proportion
        self.temporal_variance = config.temporal_variance
        self.spatial_variance = config.spatial_variance
    
    def sample(self, hidden_states, video_mask, position_ids):
        batch_size, seq_len, d = hidden_states.shape
        video_positions = position_ids[:, video_mask.sum(axis=-1)>0]
        
        T_min, H_min, W_min = video_positions.min(dim=1).values.tolist()
        T_max, H_max, W_max = video_positions.max(dim=1).values.tolist()
        sampling_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=video_mask.device)
        
        for b in range(batch_size):
            
            t_indices = position_ids[0, b]
            h_indices = position_ids[1, b]
            w_indices = position_ids[2, b]
            
            video_tokens = video_mask[b].sum(axis=-1)>0
            t_tokens, h_tokens, w_tokens = t_indices[video_tokens], h_indices[video_tokens], w_indices[video_tokens]
            
            t_center = (T_min + T_max) / 2
            h_center = (H_min + H_max) / 2
            w_center = (W_min + W_max) / 2

            t_dist_sq = (t_tokens - t_center) ** 2 / (2 * self.temporal_variance ** 2)
            hw_dist_sq = ((h_tokens - h_center) ** 2 + (w_tokens - w_center) ** 2) / (2 * self.spatial_variance ** 2)
            prob = torch.exp(-(t_dist_sq + hw_dist_sq))
            prob /= prob.sum()

            sampled_indices = torch.multinomial(prob, int((1-self.retain_proportion)*prob.numel()), replacement=False)


            drop_mask = torch.ones_like(video_tokens, dtype=torch.bool)
            drop_mask[video_tokens.nonzero(as_tuple=True)[0][sampled_indices]] = 0

            sampling_mask[b] &= drop_mask
            
        
        hidden_states = hidden_states * sampling_mask.unsqueeze(-1)
        video_mask = video_mask & sampling_mask.unsqueeze(-1)
        return hidden_states, video_mask, sampling_mask
