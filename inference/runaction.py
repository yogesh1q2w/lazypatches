
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
from dataset.charadesaction import Charades_action

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = "/home/atuin/g102ea/shared/group_10/model_checkpoints/qwen2vl-7b-instruct"  

# Load the model in half-precision on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(checkpoint_path, device_map="auto", torch_dtype="auto")
processor = AutoProcessor.from_pretrained(checkpoint_path, max_pixels=202500)

print("Loading model complete")
charades_dataset = Charades_action(root="/home/atuin/g102ea/g102ea12/dataset/charades/videos/Charades_v1",
                                labelpath="/home/atuin/g102ea/g102ea12/dataset/charades/anotations/Charades/Charades_v1_test.csv",
                                classespath="/home/atuin/g102ea/g102ea12/dataset/charades/anotations/Charades/Charades_v1_classes.txt"
                                )
batch_size = 1
shuffle = True
num_workers = 0
data_loader = DataLoader(dataset=charades_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
data_loader_iter = iter(data_loader)
for i in range(3):
    video, question, answer = next(data_loader_iter)

    video = video.squeeze(0)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": question[0]},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], videos=[video], padding=True, return_tensors="pt", do_resize=True, size=140)
    inputs = inputs.to(DEVICE)

    output_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    torch.cuda.empty_cache()
    print(output_text)
