import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from torch.utils.data import DataLoader
from dataset.charades_action import CharadesActionMCQ

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CHECKPOINT_PATH = "/home/atuin/g102ea/shared/group_10/model_checkpoints/qwen2vl-7b-instruct"

ROOT_PATH = "/home/atuin/g102ea/g102ea12/dataset"
DATASET_PATH = os.path.join(ROOT_PATH, "charades")

if os.path.exists(os.path.join(DATASET_PATH, "charades_mcq.json")):
    charades_dataset = CharadesActionMCQ(dataset_path=os.path.exists(os.path.join(DATASET_PATH, "charades_mcq.json")), reload=True)
else:
    charades_dataset = CharadesActionMCQ(dataset_path=os.path.exists(os.path.join(DATASET_PATH, "charades_mcq.json")),
                                    videos_path=os.path.join(DATASET_PATH, "/videos/Charades_v1"),
                                    labels_path=os.path.join(DATASET_PATH, "anotations/Charades/Charades_v1_test.csv"),
                                    classes_path=os.path.join(DATASET_PATH, "anotations/Charades/Charades_v1_classes.txt"),
                                    n_wrong_options=4,
                                    reload=False
                                    )


# Load the model in half-precision on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT_PATH, device_map="auto", torch_dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_CHECKPOINT_PATH)

print("Loading model complete")

data_loader = DataLoader(dataset=charades_dataset, batch_size=1, shuffle=True)
data_loader_iter = iter(data_loader)
for i in range(5):
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
    inputs = processor(text=[text_prompt], videos=[video], padding=True, return_tensors="pt")
    inputs = inputs.to(DEVICE)

    output_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(output_text)
    torch.cuda.empty_cache()
    