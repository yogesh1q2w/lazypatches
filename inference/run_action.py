import os
import sys
import torch
import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from torch.utils.data import DataLoader
from dataset.perceptiontest_mcq import PerceptiontestMCQ

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_EVERY = 100

MODEL_CHECKPOINT_PATH = "/home/atuin/g102ea/shared/group_10/model_checkpoints/qwen2vl-7b-instruct"

ROOT_PATH = "/home/atuin/g102ea/shared/group_10/datasets"
DATASET_PATH = os.path.join(ROOT_PATH, "perceptiontest")


# RELOAD=True
# if RELOAD:
#     charades_dataset = CharadesActionMCQ(dataset_path="/home/atuin/g102ea/shared/datasets/charades/charades_mcq.json", reload=RELOAD)
# else:
#     charades_dataset = CharadesActionMCQ(dataset_path="/home/atuin/g102ea/shared/datasets/charades/charades_mcq.json",
#                                     videos_path=os.path.join(DATASET_PATH, "videos/Charades_v1"),
#                                     labels_path=os.path.join(DATASET_PATH, "anotations/Charades/Charades_v1_test.csv"),
#                                     classes_path=os.path.join(DATASET_PATH, "anotations/Charades/Charades_v1_classes.txt"),
#                                     n_wrong_options=4,
#                                     reload=RELOAD
#                                     )

RELOAD=False
if RELOAD:
    perceptiontest_dataset = PerceptiontestMCQ(dataset_path="/home/atuin/g102ea/shared/group_10/datasets/perceptiontest/perceptiontest_mcq.json", reload=RELOAD)
else:
    perceptiontest_dataset = PerceptiontestMCQ(dataset_path="/home/atuin/g102ea/shared/group_10/datasets/perceptiontest/perceptiontest_mcq.json",
                                    videos_path=os.path.join(DATASET_PATH, "valid/videos"),
                                    labels_path=os.path.join(DATASET_PATH, "valid/all_valid.json"),
                                    # classes_path=os.path.join(DATASET_PATH, "anotations/Charades/Charades_v1_classes.txt"),
                                    # n_wrong_options=4,
                                    reload=RELOAD
                                    )

# Load the model in half-precision on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT_PATH, device_map="auto", torch_dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_CHECKPOINT_PATH)

print("Loading model complete", flush=True)

data_loader = DataLoader(dataset=perceptiontest_dataset, batch_size=1, shuffle=False)
print("Length of dataset: ", len(perceptiontest_dataset), flush=True)
results = []
failed_indices = []

for step, data in enumerate(data_loader):
    idx, video, question, answer = data

    idx = idx[0]
    video = video.squeeze(0)
    question = question[0]
    answer = answer[0]

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": question},
            ],
        }
    ]

    try:
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], videos=[video], padding=True, return_tensors="pt")
        inputs = inputs.to(DEVICE)

        output_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(int(idx), output_text[0], flush=True)
        print(answer, flush=True)
        print("-------------------", flush=True)
        results.append({"idx": int(idx), "answer": answer, "output": output_text[0]})
        torch.cuda.empty_cache()
    except:
        failed_indices.append(int(idx))
    
    if step % SAVE_EVERY == 0:
        json.dump(results, open("results.json", "w"))
        json.dump(failed_indices, open("failed_indices.json", "w"))
        print(f"saved till step {step}", file=sys.stderr)
    
json.dump(results, open("results.json", "w"))
json.dump(failed_indices, open("failed_indices.json", "w"))