import os
import sys
import torch
import json
import logging
import time
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from torch.utils.data import DataLoader
from dataset.charades_action import CharadesActionMCQ
from dataset.sub_charades_action import Sub_CharadesActionMCQ

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_EVERY = 5

MODEL_CHECKPOINT_PATH = "/home/atuin/g102ea/shared/group_10/model_checkpoints/qwen2vl-7b-instruct"

ROOT_PATH = "/home/atuin/g102ea/g102ea12/dataset"
DATASET_PATH = os.path.join(ROOT_PATH, "charades")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("evaluation.log"),  # Log to a file
        logging.StreamHandler(sys.stdout),      # Log to console
    ]
)
logger = logging.getLogger(__name__)

RELOAD=True
if RELOAD:
    charades_dataset = Sub_CharadesActionMCQ(dataset_path="/home/atuin/g102ea/shared/group_10/datasets/charades/subset_charades_mcq.json", reload=RELOAD)
else:
    charades_dataset = Sub_CharadesActionMCQ(dataset_path="/home/atuin/g102ea/shared/group_10/datasets/charades/subset_charades_mcq.json",
                                    videos_path=os.path.join(DATASET_PATH, "videos/Charades_v1"),
                                    labels_path=os.path.join(DATASET_PATH, "anotations/Charades/Charades_v1_test.csv"),
                                    classes_path=os.path.join(DATASET_PATH, "anotations/Charades/Charades_v1_classes.txt"),
                                    n_wrong_options=4,
                                    reload=RELOAD
                                    )


# Load the model in half-precision on the available devicea(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT_PATH, device_map="auto", torch_dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_CHECKPOINT_PATH)

print("Loading model complete", flush=True)

def normalize_text(text):
    """Normalize text for comparison"""
    return text.strip().lower() if isinstance(text, str) else ""

data_loader = DataLoader(dataset=charades_dataset, batch_size=1, shuffle=False)
print("Length of dataset: ", len(charades_dataset), flush=True)
results = []
failed_indices = []

for step, data in enumerate(data_loader):
    if step >= 5:
        break
    start_time = time.time()  # <-- Start timer
    
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
        end_time = time.time()  # <-- End timer
        elapsed_time = end_time - start_time  # <-- Calculate time
        print(int(idx), output_text[0], flush=True)
        print(answer, flush=True)
        print("-------------------", flush=True)

        # Evaluate correctness
        pred = normalize_text(output_text)
        gt = normalize_text(answer)
        is_correct = any([
            gt in pred,       # Check if answer is substring of prediction
            pred == gt,       # Exact match
            pred.startswith(gt.split()[0])  # Handle partial matches
        ])

        # Store results
        results.append({
            "idx": int(idx),
            "question": question,
            "answer": answer,
            "prediction": output_text,
            "is_correct": is_correct,
            "processing_time": elapsed_time  # <-- Store processing time
        })
        torch.cuda.empty_cache()
    # except:
    #     failed_indices.append(int(idx))
    #     torch.cuda.empty_cache()
    except KeyError as e:
        print(f"KeyError at index {idx}: {e}", exc_info=True)
        logger.error(f"KeyError at index {idx}: {e}", exc_info=True)
        failed_indices.append(int(idx))

    except ValueError as e:
        print(f"ValueError at index {idx}: {e}", exc_info=True)
        logger.error(f"ValueError at index {idx}: {e}", exc_info=True)
        failed_indices.append(int(idx))

    except RuntimeError as e:
        print(f"RuntimeError (Possible CUDA issue) at index {idx}: {e}", exc_info=True)
        logger.error(f"RuntimeError (Possible CUDA issue) at index {idx}: {e}", exc_info=True)
        failed_indices.append(int(idx))

    except Exception as e:  # Catch-all for unexpected errors
        print(f"Unexpected error at index {idx}: {e}", exc_info=True)
        logger.error(f"Unexpected error at index {idx}: {e}", exc_info=True)
        failed_indices.append(int(idx))
    finally:
        torch.cuda.empty_cache()    
    
    # if step % 10 == 0:
            # logger.info(f"Processed {step}/{len(charades_dataset)} - Current ACC: {sum(r['is_correct'] for r in results)/len(results):.2f}")

    if step % SAVE_EVERY == 0:
        json.dump(results, open("results.json", "w"))
        json.dump(failed_indices, open("failed_indices.json", "w"))
        print(f"saved till step {step}", file=sys.stderr)
    torch.cuda.empty_cache()
    
json.dump(results, open("results.json", "w"))
json.dump(failed_indices, open("failed_indices.json", "w"))


# Calculate final metrics
correct = sum(r["is_correct"] for r in results)
total = len(results)
accuracy = correct / total if total > 0 else 0

logger.info("\nFinal Evaluation Results:")
logger.info(f"Processed Samples: {total}")
logger.info(f"Failed Samples: {len(failed_indices)}")
logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
logger.info(f"Error Rate: {len(failed_indices)/len(charades_dataset):.4f}")