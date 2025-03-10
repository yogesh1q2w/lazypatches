import os
import json
import torch
from tqdm import tqdm
import torch.utils.data as data
from inference.utils.vision_process import fetch_video


Prompts_pre = f"Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, or C) of the correct option."
Prompts_suf = f"The best answer is:"

num_mapping = {0: "A", 1: "B", 2: "C"}


def parse_video_and_mcq_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    parsed_data = {}

    for video_id, video_info in data.items():
        parsed_data[video_id] = {"metadata": video_info["metadata"], "mcq_data": []}

        for mcq in video_info["mc_question"]:
            parsed_data[video_id]["mcq_data"].append(
                {
                    "id": mcq["id"],
                    "question": mcq["question"],
                    "options": mcq["options"],
                    "answer_id": mcq["answer_id"],
                    "area": mcq["area"],
                    "reasoning": mcq["reasoning"],
                    "tag": mcq["tag"],
                }
            )

    return parsed_data


class PerceptiontestMCQ(data.Dataset):
    def __init__(
        self, dataset_path=None, videos_path=None, labels_path=None, reload=True, use_subset=True, target_fps=2.0
    ):
        # give either already exisiting dataset or corresponding paths to build one
        if reload:
            assert videos_path is None and labels_path is None
            self.data = json.load(open(dataset_path, "r"))
        else:
            assert dataset_path is not None and videos_path is not None and labels_path is not None

            video_mcqs_info = parse_video_and_mcq_json(labels_path)

            if use_subset:
                with open(
                    "/home/atuin/g102ea/shared/group_10/datasets/perceptiontest/subset_perceptiontest.json", "r"
                ) as f:
                    subset_data = json.load(f)

                mcq_data = self.prepare(video_mcqs_info, videos_path, subset_data)
            else:
                mcq_data = self.prepare(video_mcqs_info, videos_path)

            self.data = {
                "videos_path": videos_path,
                "labels_path": labels_path,
                "mcq_data": mcq_data,
                "n_samples": len(mcq_data["mcqs"]),
            }
            json.dump(self.data, open(dataset_path, "w"))
        self.target_fps = target_fps

    def prepare(self, video_mcqs_info, videos_path, subset_data=None):
        video_paths = []
        video_ids = []
        mcqs = []
        mcq_labels = []

        mcq_areas = []
        mcq_tags = []

        for video_id, video_info in tqdm(video_mcqs_info.items()):
            video_path = f"{videos_path}/{video_id}.mp4"

            # get all questions in a video
            mcq_list = video_info["mcq_data"]

            # count the number of questions in a video
            if subset_data == None:
                num_questions = len(mcq_list)

                # ensure the number of video_paths and video_ids aligning with questions
                video_paths.extend([video_path] * num_questions)
                video_ids.extend([video_id] * num_questions)

                # extract question and answer
                for mcq in mcq_list:
                    mcq_text = (
                        Prompts_pre
                        + "\n"
                        + mcq["question"]
                        + "\n"
                        + "\n".join([f"{num_mapping[idx]}. {option}" for idx, option in enumerate(mcq["options"])])
                        + "\n"
                        + Prompts_suf
                    )
                    mcqs.append(mcq_text)

                    # (index) option_text"
                    answer_index = mcq["answer_id"]
                    mcq_labels.append(f"{num_mapping[answer_index]}")

                    mcq_areas.append(mcq["area"])
                    mcq_tags.append(mcq["tag"])
            else:
                if video_id not in subset_data:
                    continue
                # Filter the mcq_list to only include questions in the subset
                for mcq in mcq_list:
                    if mcq["id"] in subset_data[video_id]:
                        mcq_text = (
                            Prompts_pre
                            + "\n"
                            + mcq["question"]
                            + "\n"
                            + "\n".join([f"{num_mapping[idx]}. {option}" for idx, option in enumerate(mcq["options"])])
                            + "\n"
                            + Prompts_suf
                        )
                        mcqs.append(mcq_text)

                        answer_index = mcq["answer_id"]
                        mcq_labels.append(f"{num_mapping[answer_index]}")

                        mcq_areas.append(mcq["area"])
                        mcq_tags.append(mcq["tag"])

                        video_paths.append(video_path)
                        video_ids.append(video_id)

        return {
            "video_paths": video_paths,
            "video_ids": video_ids,
            "mcqs": mcqs,
            "mcq_labels": mcq_labels,
            "mcq_areas": mcq_areas,
            "mcq_tags": mcq_tags,
        }

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (frames, conversation, target_description).
        """
        path = self.data["mcq_data"]["video_paths"][index]
        answer = self.data["mcq_data"]["mcq_labels"][index]

        video_info = {"type": "video", "video": path, "fps": self.target_fps}
        video = fetch_video(video_info["video"])
        question = self.data["mcq_data"]["mcqs"][index]

        area = self.data["mcq_data"]["mcq_areas"][index]
        tag = self.data["mcq_data"]["mcq_tags"][index]

        return index, video, question, answer, area, tag

    def __len__(self):
        return self.data["n_samples"]
