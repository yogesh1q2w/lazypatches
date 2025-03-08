import os
import sys
import json
import re
import logging
from accuracy_mcq import IncrementalMCQAcc

ROOT_PATH = "/home/atuin/g102ea/shared/group_10/results/charades"
EXPERIMENT_FOLDER_NAME = [
    "charades_uniform_1.0_0_10%_0",
    "charades_uniform_1.0_0_30%_0",
    "charades_uniform_1.0_0_50%_0",
    "charades_uniform_1.0_0_70%_0",
    "charades_uniform_1.0_0_90%_0",
    "charades_uniform_1.0_12_10%_0",
    "charades_uniform_1.0_12_30%_0",
    "charades_uniform_1.0_12_50%_0",
    "charades_uniform_1.0_12_70%_0",
    "charades_uniform_1.0_12_90%_0",
    "charades_uniform_1.0_24_10%_0",
    "charades_uniform_1.0_24_30%_0",
    "charades_uniform_1.0_24_50%_0",
    "charades_uniform_1.0_24_70%_0",
    "charades_uniform_1.0_24_90%_0",
]

for experiment_folder in EXPERIMENT_FOLDER_NAME:
    results = json.load(open(os.path.join(ROOT_PATH, experiment_folder, "results.json"), "r"))
    failed_indices = json.load(open(os.path.join(ROOT_PATH, experiment_folder, "failed_indices.json"), "r"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join("./dummy", experiment_folder + ".log")),  # Log to a file
            logging.StreamHandler(sys.stdout),  # Log to console
        ],
    )
    logger = logging.getLogger()

    metrics = IncrementalMCQAcc("charades")

    new_results = []
    for result in results:
        question, answer, prediction = result["question"], result["answer"], result["prediction"]
        answer = next(f"{i}" for i, t in re.findall(r"\((\d+)\) (.+)", question) if t.strip() == answer.strip())
        result["is_correct"] = metrics.eval_results(answer, prediction[0])

        new_results.append(results)

    # json.dump(results, open(os.path.join(ROOT_PATH, experiment_folder, "results_new.json"), "w"))
    print("-------------------", flush=True)
    correct = metrics.total["correct"]
    total_attempts = metrics.total["answered"] + len(failed_indices)
    failure_rate = len(failed_indices) / total_attempts if total_attempts > 0 else 0
    logger.info("\nFinal Evaluation Results:")
    logger.info(f"Processed Samples: {total_attempts}")
    logger.info(f"Failed Samples: {len(failed_indices)}")
    logger.info(
        f"Accuracy: {metrics.get_total_accuracy():.4f} ({metrics.total['correct']}/{metrics.total['answered']})"
    )
    logger.info(f"Failure Rate: {failure_rate:.4f}")

    while logger.hasHandlers():
        handler = logger.handlers[0]
        handler.close()
        logger.removeHandler(handler)
