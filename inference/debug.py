# run_action.py
import os
import sys
import torch
import json
import logging
import pdb  # Python Debugger
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from torch.utils.data import DataLoader
from dataset.charades_action import CharadesActionMCQ

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("evaluation.log"),  # Log to a file
        logging.StreamHandler(sys.stdout),      # Log to console
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_EVERY = 100

# Configuration paths
MODEL_CHECKPOINT_PATH = "/home/atuin/g102ea/shared/group_10/model_checkpoints/qwen2vl-7b-instruct"
DATASET_PATH = "/home/atuin/g102ea/shared/datasets/charades/charades_mcq.json"

# Initialize dataset
logger.info("Loading dataset...")
charades_dataset = CharadesActionMCQ(dataset_path=DATASET_PATH, reload=True)

# Load model and processor
logger.info("Loading model and processor...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_CHECKPOINT_PATH, 
    device_map="auto", 
    torch_dtype="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_CHECKPOINT_PATH)
logger.info("Model and processor loaded successfully.")

# Create data loader
data_loader = DataLoader(dataset=charades_dataset, batch_size=1, shuffle=False)
logger.info(f"Dataset contains {len(charades_dataset)} samples.")

results = []
failed_indices = []

def normalize_text(text):
    """Normalize text for comparison"""
    return text.strip().lower()

for step, data in enumerate(data_loader):
    idx, video, question, answer = data
    idx, video, question, answer = idx[0], video.squeeze(0), question[0], answer[0]

    try:
        # Debugging: Log current sample info
        logger.info(f"\nProcessing sample {idx}...")
        logger.info(f"Question: {question}")
        logger.info(f"Answer: {answer}")
        logger.info(f"Video shape: {video.shape}")

        # Prepare input
        conversation = [{
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": question},
            ]
        }]
        
        # Debugging: Log conversation template
        logger.info("Conversation template:")
        logger.info(conversation)

        # Process inputs
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        logger.info("Text prompt after applying template:")
        logger.info(text_prompt)

        inputs = processor(
            text=[text_prompt], 
            videos=[video], 
            padding=True, 
            return_tensors="pt"
        ).to(DEVICE)

        # Debugging: Log input tensors
        logger.info("Input tensors:")
        logger.info(f"Input IDs shape: {inputs.input_ids.shape}")
        logger.info(f"Video tensor shape: {inputs.pixel_values.shape}")

        # Generate output
        output_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        output_text = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )[0]

        # Debugging: Log model output
        logger.info("Model output:")
        logger.info(output_text)

        # Evaluate correctness
        pred = normalize_text(output_text)
        gt = normalize_text(answer)
        is_correct = any([
            gt in pred,       # Check if answer is substring of prediction
            pred == gt,       # Exact match
            pred.startswith(gt.split()[0])  # Handle partial matches
        ])
        
        # Debugging: Log evaluation result
        logger.info(f"Prediction: {pred}")
        logger.info(f"Ground Truth: {gt}")
        logger.info(f"Correct: {is_correct}")

        # Store results
        results.append({
            "idx": int(idx),
            "question": question,
            "answer": answer,
            "prediction": output_text,
            "is_correct": is_correct
        })

        # Progress logging
        if step % 10 == 0:
            logger.info(f"Processed {step}/{len(charades_dataset)} - Current ACC: {sum(r['is_correct'] for r in results)/len(results):.2f}")

        # Periodic saving
        if step % SAVE_EVERY == 0:
            json.dump(results, open("results.json", "w"))
            json.dump(failed_indices, open("failed_indices.json", "w"))

    except Exception as e:
        # Detailed error logging
        logger.error(f"Error processing sample {idx}:", exc_info=True)
        failed_indices.append(int(idx))

        # Interactive debugging with pdb
        logger.error("Entering pdb for interactive debugging...")
        pdb.post_mortem()  # Drop into pdb at the point of failure
    
    torch.cuda.empty_cache()

# Final save
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