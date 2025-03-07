import torch

from inference.utils.vision_process import FPS, smart_nframes


def fps_reduction(video_embeds, input_ids, inputs_embeds, attention_mask, video_grid_thw, llm_fps, video_token_id):
    # import pdb
    # pdb.set_trace()
    assert video_grid_thw.shape[0] == 1  # currently only handles batch size 1
    D = video_embeds.shape[-1]
    nframes, _, _ = video_grid_thw[0]
    nframes = int(nframes)
    reduced_nframes = smart_nframes({"fps": llm_fps}, nframes, FPS)
    idx = torch.linspace(0, nframes - 1, reduced_nframes).round().long()
    video_embeds = video_embeds.reshape(nframes, -1, D)[idx].reshape(-1, D)
    video_token_pos = torch.nonzero(input_ids[0] == video_token_id, as_tuple=True)[0]
    n_reduced_video_tokens = video_embeds.shape[0]
    input_ids = torch.cat(
        [
            input_ids[:, : video_token_pos[0]],
            input_ids[:, video_token_pos[0] : video_token_pos[0] + n_reduced_video_tokens],
            input_ids[:, video_token_pos[-1] + 1 :],
        ],
        dim=1,
    )
    inputs_embeds = torch.cat(
        [
            inputs_embeds[:, : video_token_pos[0], :],
            inputs_embeds[:, video_token_pos[0] : video_token_pos[0] + n_reduced_video_tokens, :],
            inputs_embeds[:, video_token_pos[-1] + 1 :, :],
        ],
        dim=1,
    )
    attention_mask = torch.cat(
        [
            attention_mask[:, : video_token_pos[0]],
            attention_mask[:, video_token_pos[0] : video_token_pos[0] + n_reduced_video_tokens],
            attention_mask[:, video_token_pos[-1] + 1 :],
        ],
        dim=1,
    )
    video_grid_thw[0, 0] = reduced_nframes

    print(f'The effective FPS into LLM for current experiment is {llm_fps}.')

    return video_embeds, input_ids, inputs_embeds, attention_mask, video_grid_thw
