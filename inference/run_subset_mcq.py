import os
import sys
import torch
import json
import logging
import re
from torch.utils.data import DataLoader

from transformers.models.qwen2_vl_lazy import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from eval.accuracy_mcq import IncrementalMCQAcc
from dataset.charades_mcq import CharadesActionMCQ
from dataset.perceptiontest_mcq import PerceptiontestMCQ
from inference.arg_idx import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_EVERY = 100

MODEL_CHECKPOINT_PATH = "/home/atuin/g102ea/shared/group_10/model_checkpoints/qwen2vl-7b-instruct"

# argument list by order: [LLM_FPS] [RETENTION_RATE] [SAMPLER_TYPE] [DATASET] [HYPERPARAM] [DROPPING_POSITION]


RETENTION_RATE = float(sys.argv[RETENTION_RATE_ARG_IDX])
SAMPLER_TYPE = sys.argv[SAMPLER_TYPE_ARG_IDX]
DATASET = sys.argv[DATASET_ARG_IDX].lower()
HYPERPARAM = float(sys.argv[HYPERPARAM_ARG_IDX])
DROPPING_POSITION = int(sys.argv[DROPPING_POSITION_ARG_IDX])
TARGET_PATH = sys.argv[TARGET_PATH_ARG_IDX]

ROOT_PATH = "/home/atuin/g102ea/shared/group_10/datasets"
DATASET_PATH = os.path.join(ROOT_PATH, DATASET)

# os.makedirs(TARGET_PATH, exist_ok=True) # for debugging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(TARGET_PATH, "evaluation.log")),  # Log to a file
        logging.StreamHandler(sys.stdout),  # Log to console
    ],
)
logger = logging.getLogger(__name__)

RELOAD = True
if DATASET == "charades":
    if RELOAD:
        dataset = CharadesActionMCQ(dataset_path=os.path.join(DATASET_PATH, "subset_charades_mcq.json"), reload=RELOAD)
    else:
        dataset = CharadesActionMCQ(
            dataset_path=os.path.join(DATASET_PATH, "subset_charades_mcq.json"),
            videos_path=os.path.join(DATASET_PATH, "videos/Charades_v1"),
            labels_path=os.path.join(DATASET_PATH, "anotations/Charades/Charades_v1_test.csv"),
            classes_path=os.path.join(DATASET_PATH, "anotations/Charades/Charades_v1_classes.txt"),
            n_wrong_options=4,
            reload=RELOAD,
        )
elif DATASET == "perceptiontest":
    if RELOAD:
        dataset = PerceptiontestMCQ(
            dataset_path=os.path.join(DATASET_PATH, "sub_perceptiontest_mcq.json"), reload=RELOAD
        )
    else:
        dataset = PerceptiontestMCQ(
            dataset_path=os.path.join(DATASET_PATH, "sub_perceptiontest_mcq.json"),
            videos_path=os.path.join(DATASET_PATH, "valid/videos"),
            labels_path=os.path.join(DATASET_PATH, "valid/all_valid.json"),
            reload=RELOAD,
        )

model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT_PATH, device_map="auto", torch_dtype="auto")
processor = Qwen2VLProcessor.from_pretrained(MODEL_CHECKPOINT_PATH)

print("Loading model complete", flush=True)

metrics = IncrementalMCQAcc(DATASET)

data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
print("Length of dataset: ", len(dataset), flush=True)
results = []
failed_indices = []

for step, data in enumerate(data_loader):

    # if step == 4:   # for debugging
    #     break

    if DATASET == "charades":
        idx, video, question, answer = data
        question = question[0]
        answer = answer[0]
        answer = next(f"{i}" for i, t in re.findall(r"\((\d+)\) (.+)", question) if t.strip() == answer.strip())
        area, tag = None, None

    elif DATASET == "perceptiontest":
        idx, video, question, answer, area, tag = data
        question = question[0]
        answer = answer[0]
        area = area[0]
        tag = [i[0] for i in tag]

    idx = idx[0]
    video = video.squeeze(0)

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
        generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        is_correct = metrics.eval_results(answer, output_text[0], area, tag)

        print(f"{idx} {question}", flush=True)
        print(f"Correct answer = {answer}", flush=True)
        print(f"Output answer = {output_text[0]}", flush=True)
        print(f"Is correct = {is_correct}", flush=True)

        results.append(
            {
                "idx": int(idx),
                "question": question,
                "answer": answer,
                "prediction": output_text,
                "is_correct": is_correct,
            }
        )
        torch.cuda.empty_cache()

    except Exception as e:
        logger.info(f"Exception thrown is {e}")
        failed_indices.append(int(idx))
        torch.cuda.empty_cache()

    if step % SAVE_EVERY == 0:
        json.dump(results, open(os.path.join(TARGET_PATH, "results.json"), "w"))
        json.dump(failed_indices, open(os.path.join(TARGET_PATH, "failed_indices.json"), "w"))
        logger.info(f"---Saved till step {step}----")

        logger.info(
            f"Processed {step}/{len(dataset)} - Current Accuracy: {metrics.get_total_accuracy():.4f} ({metrics.total['correct']}/{metrics.total['answered']}), Failed = {len(failed_indices)}"
        )

    torch.cuda.empty_cache()

json.dump(results, open(os.path.join(TARGET_PATH, "results.json"), "w"))
json.dump(failed_indices, open(os.path.join(TARGET_PATH, "failed_indices.json"), "w"))


print("-------------------", flush=True)
correct = metrics.total["correct"]
total_attempts = metrics.total["answered"] + len(failed_indices)
failure_rate = len(failed_indices) / total_attempts if total_attempts > 0 else 0
logger.info("\nFinal Evaluation Results:")
logger.info(f"Processed Samples: {total_attempts}")
logger.info(f"Failed Samples: {len(failed_indices)}")
logger.info(f"Accuracy: {metrics.get_total_accuracy():.4f} ({metrics.total['correct']}/{metrics.total['answered']})")
logger.info(f"Failure Rate: {failure_rate:.4f}")
if DATASET == "perceptiontest":
    area_accuracy, tag_accuracy = metrics.get_area_and_tag_accuracy()
    logger.info(f"Area accuracy = {area_accuracy}")
    logger.info(f"Tag accuracy = {tag_accuracy}")
