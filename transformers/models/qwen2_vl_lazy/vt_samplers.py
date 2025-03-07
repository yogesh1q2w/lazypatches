import numpy as np
import torch
import torch.nn.functional as F
import abc
import random


class Sampler(abc.ABC):
    def __init__(self, sampler_name):
        self.sampler_name = sampler_name

    @abc.abstractmethod
    def sample(self, hidden_states):
        pass

    def __call__(self, hidden_states, video_mask, position_ids, input_ids):
        return self.sample(hidden_states, video_mask, position_ids, input_ids)


class UniformSampler(Sampler):
    def __init__(self, config):
        super().__init__("uniform")
        self.retain_proportion = config.retain_proportion

    def sample(self, hidden_states, video_mask, position_ids, input_ids):
        batch_size, seq_len, embed_dim = hidden_states.shape

        sampling_mask = torch.ones_like(video_mask, dtype=torch.bool)

        for b in range(batch_size):
            true_indices = torch.nonzero(video_mask[b].flatten(), as_tuple=True)[0]
            num_to_drop = int((1 - self.retain_proportion) * true_indices.size(0))

            drop_indices = true_indices[torch.randperm(true_indices.size(0))[:num_to_drop]]

            flat_sampling_mask = sampling_mask[b].flatten()
            flat_sampling_mask[drop_indices] = False

            sampling_mask[b] = flat_sampling_mask.view(video_mask[b].shape)

        hidden_states = hidden_states * sampling_mask.float()
        video_mask = video_mask & sampling_mask

        print(f"SAMPLING RATE FOR UNIFORM IS {(1-self.retain_proportion)*100}%")

        return hidden_states, video_mask, sampling_mask


class SpatioTemporalHeuristicSampler(Sampler):
    def __init__(self, config):
        super().__init__("st_gaussian")
        self.retain_proportion = config.retain_proportion
        self.temporal_variance = config.temporal_variance
        self.spatial_variance = config.spatial_variance

    def sample(self, hidden_states, video_mask, position_ids, input_ids):
        batch_size, seq_len, d = hidden_states.shape
        video_positions = position_ids[:, video_mask.sum(axis=-1) > 0]

        T_min, H_min, W_min = video_positions.min(dim=1).values.tolist()
        T_max, H_max, W_max = video_positions.max(dim=1).values.tolist()
        sampling_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=video_mask.device)

        for b in range(batch_size):
            t_indices = position_ids[0, b]
            h_indices = position_ids[1, b]
            w_indices = position_ids[2, b]

            video_tokens = video_mask[b].sum(axis=-1) > 0
            t_tokens, h_tokens, w_tokens = t_indices[video_tokens], h_indices[video_tokens], w_indices[video_tokens]

            t_center = (T_min + T_max) / 2
            h_center = (H_min + H_max) / 2
            w_center = (W_min + W_max) / 2

            t_dist_sq = (t_tokens - t_center) ** 2 / (2 * self.temporal_variance**2)
            hw_dist_sq = ((h_tokens - h_center) ** 2 + (w_tokens - w_center) ** 2) / (2 * self.spatial_variance**2)
            prob = torch.exp(-(t_dist_sq + hw_dist_sq))
            prob /= prob.sum()

            sampled_indices = torch.multinomial(
                prob, int((1 - self.retain_proportion) * prob.numel()), replacement=False
            )

            drop_mask = torch.ones_like(video_tokens, dtype=torch.bool)
            drop_mask[video_tokens.nonzero(as_tuple=True)[0][sampled_indices]] = 0

            sampling_mask[b] &= drop_mask

        hidden_states = hidden_states * sampling_mask.unsqueeze(-1)
        video_mask = video_mask & sampling_mask.unsqueeze(-1)

        print(f"SAMPLING RATE FOR GAUSSIAN(sigma_t={self.temporal_variance}, sigma_s={self.spatial_variance}) IS {(1-self.retain_proportion)*100}%")

        return hidden_states, video_mask, sampling_mask


class KMclosestTokenSampler(Sampler):
    def __init__(self, config):
        super().__init__("km_closest")
        self.k = config.k_farthest
        self.retain_proportion = config.retain_proportion

    def sample(self, hidden_states, video_mask, position_ids, input_ids):
        batch_size, seq_len, d = hidden_states.shape
        video_positions = position_ids[:, video_mask.sum(axis=-1) > 0]
        sampling_mask = torch.ones_like(video_mask, dtype=torch.bool)
        video_seq_len = len(video_positions[0])

        for b in range(batch_size):
            state_normalized = F.normalize(hidden_states[b])
            cosine_sim = torch.mm(state_normalized, state_normalized.T)  # should return seq_len, seq_len

            assert torch.allclose(cosine_sim, cosine_sim.T, atol=1e-6), "Matrix is not symmetric"
            assert torch.allclose(
                torch.diag(cosine_sim), torch.ones(seq_len), atol=1e-6
            ), "Diagonal values are not all 1"

            cosine_dist = -1.0 * cosine_sim

            text_indices = (video_mask[b] == 0).nonzero(as_tuple=True)[0].unique()
            video_indices = (video_mask[b] == 1).nonzero(as_tuple=True)[0].unique()
            lower_text_bound = (input_ids[0][text_indices] == 151653).nonzero(as_tuple=True)[
                0
            ].item() + 1  # User input text starts after |vision_end| token whose token id is 151653.
            upper_text_bound = (
                lower_text_bound
                + (input_ids[0][text_indices][lower_text_bound:] == 151645).nonzero(as_tuple=True)[0].item()
            )  # User input ends before the first |im_end| tag after |vision_end| is encountered. Input ID of |im_end| is 151645

            relevant_text_indices = text_indices[lower_text_bound:upper_text_bound]

            num_text_tokens = int(self.k * len(relevant_text_indices))
            m = (self.retain_proportion * video_seq_len) // num_text_tokens

            selected_text = []
            cosine_dist_text = cosine_dist.clone()
            if len(relevant_text_indices) == num_text_tokens:
                selected_text = relevant_text_indices

            elif num_text_tokens < len(relevant_text_indices):
                selected_text = [relevant_text_indices[torch.randint(0, len(relevant_text_indices), (1,)).item()]]

                for _ in range(num_text_tokens - 1):
                    last_text_token = selected_text[-1]
                    distances = cosine_dist_text[last_text_token, relevant_text_indices]
                    farthest_idx = torch.argmax(distances).item()
                    next_text_token = relevant_text_indices[farthest_idx]
                    # Avoid selecting already picked tokens
                    cosine_dist_text[last_text_token, :] = -1e9
                    cosine_dist_text[:, last_text_token] = -1e9

                    selected_text.append(next_text_token)

            selected_video = set()
            remaining_video_indices = video_indices.clone()
            for text_token in selected_text:
                if len(remaining_video_indices) == 0:
                    break
                video_dists = cosine_dist[text_token, remaining_video_indices]
                num_to_select = int(m)
                if num_to_select > len(remaining_video_indices):
                    num_to_select = len(remaining_video_indices)

                closest_video_indices = remaining_video_indices[video_dists.topk(num_to_select, largest=False).indices]
                for idx in closest_video_indices:
                    # Avoid selecting already picked tokens
                    cosine_dist[:, idx] = 1e12
                    cosine_dist[idx, :] = 1e12
                selected_video.update(closest_video_indices.tolist())
                mask = torch.tensor(
                    [idx not in selected_video for idx in remaining_video_indices.tolist()],
                    device=hidden_states.device,
                    dtype=torch.bool,
                )
                remaining_video_indices = remaining_video_indices[mask]

            selected_video = torch.tensor(list(selected_video), device=hidden_states.device)
            sampling_mask[b, video_indices] = False
            sampling_mask[b, selected_video] = True

        hidden_states = hidden_states * sampling_mask.float()
        video_mask = video_mask & sampling_mask

        print(f"SAMPLING RATE FOR KM-CLOSEST(k={self.k}) IS {(1-self.retain_proportion)*100}%")

        return hidden_states, video_mask, sampling_mask
