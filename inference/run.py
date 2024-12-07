
from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from torch.utils.data import DataLoader

import sys
import os
dataset = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(dataset)
from dataset.charadesdescription import Charades_decription

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = "/home/atuin/g102ea/shared/group_10/model_checkpoints/qwen2vl-7b-instruct"  

# Load the model in half-precision on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(checkpoint_path, device_map="auto", torch_dtype="auto")
processor = AutoProcessor.from_pretrained(checkpoint_path, max_pixels=202500)

print("Loading model complete")
# Image
# url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
# image = Image.open(requests.get(url, stream=True).raw)

# conversation = [
#     {
#         "role":"user",
#         "content":[
#             {
#                 "type":"image",
#             },
#             {
#                 "type":"text",
#                 "text":"Describe this image."
#             }
#         ]
#     }
# ]


# Preprocess the inputs
# text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
# Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

# inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
# inputs = inputs.to('cuda')

# Inference: Generation of the output
# output_ids = model.generate(**inputs, max_new_tokens=128)
# generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
# output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
# print(output_text)

# Video
# def fetch_video(ele: Dict, nframe_factor=2):
#     if isinstance(ele['video'], str):
#         def round_by_factor(number: int, factor: int) -> int:
#             return round(number / factor) * factor

#         video = ele["video"]
#         if video.startswith("file://"):
#             video = video[7:]

#         video, _, info = io.read_video(
#             video,
#             start_pts=ele.get("video_start", 0.0),
#             end_pts=ele.get("video_end", None),
#             pts_unit="sec",
#             output_format="TCHW",
#         )
#         assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
#         if "nframes" in ele:
#             nframes = round_by_factor(ele["nframes"], nframe_factor)
#         else:
#             fps = ele.get("fps", 1.0)
#             nframes = round_by_factor(video.size(0) / info["video_fps"] * fps, nframe_factor)
#         idx = torch.linspace(0, video.size(0) - 1, nframes, dtype=torch.int64)
#         return video[idx]

# video_info = {"type": "video", "video": "/path/to/video.mp4", "fps": 1.0}
# video = fetch_video(video_info)
# conversation = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "video"},
#             {"type": "text", "text": "What happened in the video?"},
#         ],
#     }
# ]

charades_dataset = Charades_decription(root="/home/atuin/g102ea/g102ea12/dataset/charades/videos/Charades_v1",
                                split="val_video",
                                labelpath="/home/atuin/g102ea/g102ea12/dataset/charades/anotations/Charades/Charades_v1_test.csv",
                                cachedir="/home/atuin/g102ea/g102ea12/dataset/charades/cache"
                                )
batch_size = 1
shuffle = True
num_workers = 0
data_loader = DataLoader(dataset=charades_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
data_loader_iter = iter(data_loader)
video, text, target = next(data_loader_iter)

print("before handle:", video.shape)
video = video.squeeze(0)
print("input:", video.shape)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video"},
            {"type": "text", "text": text[0]},
        ],
    }
]
print(conversation)

# Preprocess the inputs
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
# Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>What happened in the video?<|im_end|>\n<|im_start|>assistant\n'

inputs = processor(text=[text_prompt], videos=[video], padding=True, return_tensors="pt", do_resize=True, size=140)
inputs = inputs.to(DEVICE)

# Inference: Generation of the output
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)
