
from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers.models.qwen2_vl_lazy import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from inference.utils.vision_process import fetch_video

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


checkpoint_path = "/home/atuin/g102ea/shared/group_10/model_checkpoints/qwen2vl-7b-instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(checkpoint_path, device_map="auto", torch_dtype="auto")
processor = Qwen2VLProcessor.from_pretrained(checkpoint_path)

print("Loading model complete")

# Video
video_info = {"type": "video", "video": "./test_files/monkey.gif", "fps": 1.0}
video = fetch_video(video_info)
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video"},
            {"type": "text", "text": "Describe the video."},
        ],
    }
]

# Preprocess the inputs
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
# Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>What happened in the video?<|im_end|>\n<|im_start|>assistant\n'

inputs = processor(text=[text_prompt], videos=[video], padding=True, return_tensors="pt")
inputs = inputs.to(DEVICE)

# Inference: Generation of the output
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)
