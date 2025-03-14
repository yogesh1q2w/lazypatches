from inference.utils.vision_process import fetch_video

import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path


def generate_video(frames_tensor, save_path):
    frames_np = frames_tensor.permute(0, 3, 1, 2).numpy()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 1  
    width = frames_np.shape[3]
    height = frames_np.shape[2]

    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    for i in range(frames_np.shape[0]):
        frame = frames_np[i].transpose(1, 2, 0) * 255  
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = np.uint8(frame) 
        out.write(frame)

    out.release()
    
def generate_video_with_mask(video_path, mask, save=False):
    show_frames_with_mask = True
    video_name = video_path.split('/')[-1]
    save_folder = Path('/home/atuin/g102ea/shared/group_10/video_with_mask')
    save_folder.mkdir(parents=True, exist_ok=True)
    save_path = str(save_folder/video_name)
    print(save_path)
    print(video_path)
    extracted_video = fetch_video(video_path).permute(0, 2, 3, 1)[0::2, :, :, :]
    extracted_video = (extracted_video - extracted_video.min()) / (extracted_video.max() - extracted_video.min())
    print(extracted_video.shape)
    
    assert extracted_video.shape[0] == mask.shape[0]
    
    patch_size = 14
    alpha = 0.2
    frames_with_mask = extracted_video.clone()
    for Ti in range(extracted_video.shape[0]):
        for Hi,h in enumerate(range(0, extracted_video.shape[1], patch_size)):
            for Wi,w in enumerate(range(0, extracted_video.shape[2], patch_size)):
                patch_img = extracted_video[Ti, h:h+patch_size, w:w+patch_size, :]
                if not mask[Ti, Hi, Wi]:
                    frames_with_mask[Ti, h:h+patch_size, w:w+patch_size, :] = (alpha * patch_img + (1 - alpha) * 1.0)
    if save:
        generate_video(frames_with_mask, save_path)
    
    if show_frames_with_mask:
        for i in range(extracted_video.shape[0]):
            plt.figure()
            plt.imshow(frames_with_mask[i].numpy())
            plt.axis('off')
            plt.show()
    
    return frames_with_mask    

if __name__ == "__main__":
    video_path = '/home/atuin/g102ea/shared/group_10/datasets/charades/videos/Charades_v1/0A8CF.mp4'
    mask = torch.randint(0, 2, (15, 26, 20), dtype=torch.float32)
    # print(mask)
    fames_with_mask = generate_video_with_mask(video_path, mask)