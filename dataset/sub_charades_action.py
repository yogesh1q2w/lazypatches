import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import json

from torchvision import io
from typing import Dict
from tqdm import tqdm
from inference.utils.vision_process import fetch_video


def parse_video_action_csv(filename):
    def extract_details(annot):
        actions = [action_desc.split(" ") for action_desc in str(annot["actions"]).split(";")]
        actions = [(item[0], float(item[1]), float(item[2])) for item in actions if item[0] != "nan"]
        descriptions = annot["descriptions"].split(";")
        objects = [] if str(annot["objects"]) == "nan" else annot["objects"].split(";")
        retval = {
            "id": annot["id"],
            "actions": actions,
            "descriptions": descriptions,
            "scene": annot["scene"],
            "objects": objects,
            "length": float(annot["length"]),
        }
        return retval

    df = pd.read_csv(filename)
    labels = list(df.apply(lambda annot: extract_details(annot), axis=1))
    return labels


def extract_subset_data(filename, sampled_data):
    df = pd.read_csv(filename)

    subset_data = {}
    num = 0
    for _, row in df.iterrows():
        video_id = row["id"]
        for action, info in sampled_data.items():
            if video_id in info["videos"]:  # Only keep selected actions
                if video_id not in subset_data:
                    subset_data[video_id] = {
                        "id": video_id,
                        "actions": [],
                        "descriptions": row["descriptions"].split(";"),
                        "scene": row["scene"],
                        "objects": [] if str(row["objects"]) == "nan" else row["objects"].split(";"),
                        "length": float(row["length"]),
                    }

                raw_actions = [action_desc.split(" ") for action_desc in str(row["actions"]).split(";")]

                seen = set()  # record existed action_id
                actions = []
                for item in raw_actions:
                    action_id = item[0]
                    if action_id == action and action_id not in seen:
                        actions.append((action_id, float(item[1]), float(item[2])))
                        seen.add(action_id)

                if actions:
                    subset_data[video_id]["actions"].extend(actions)

    return list(subset_data.values())  # Convert dictionary back to list


def parse_action_id_mapping(filename):
    action_id_mapping = {}
    with open(filename) as f:
        for line in f:
            # Split on whitespace, assuming first word is class_id of length 4
            if len(line) > 0:
                action_id_mapping[line[:4]] = line[5:-1]

    return action_id_mapping


class Sub_CharadesActionMCQ(data.Dataset):
    def __init__(
        self,
        dataset_path=None,
        videos_path=None,
        labels_path=None,
        classes_path=None,
        n_wrong_options=None,
        reload=True,
        target_fps=2.0,
    ):
        # give either already exisiting dataset or corresponding paths to build one
        if reload:
            assert videos_path is None and labels_path is None and classes_path is None and n_wrong_options is None
            self.data = json.load(open(dataset_path, "r"))
        else:
            assert (
                dataset_path is not None
                and videos_path is not None
                and labels_path is not None
                and classes_path is not None
                and n_wrong_options is not None
            )
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils/subset_charades.json"), "r") as f:
                subset_data = json.load(f)

            subset_labels = extract_subset_data(labels_path, subset_data)
            labels = parse_video_action_csv(labels_path)
            action_id_mapping = parse_action_id_mapping(classes_path)

            mcq_data = self.prepare(labels, subset_labels, action_id_mapping, n_wrong_options, videos_path)

            self.data = {
                "videos_path": videos_path,
                "labels_path": labels_path,
                "classes_path": classes_path,
                "n_wrong_options": n_wrong_options,
                "mcq_data": mcq_data,
                "n_samples": len(mcq_data["mcqs"]),
            }
            json.dump(self.data, open(dataset_path, "w"))
        self.target_fps = target_fps

    def prepare(self, labels, subset_labels, action_id_mapping, n_wrong_options, videos_path):
        question_prompt = open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils/action_mcq.txt"),
            "r",
        ).read()

        def generate_questions(true_actions_ids, subset_true_action_ids):
            false_action_ids = list(set(action_id_mapping.keys()) - set(true_actions_ids))
            questions = []
            answers = []
            for true_option_id in subset_true_action_ids:
                false_options_ids = np.random.choice(false_action_ids, size=n_wrong_options, replace=False)
                all_options = np.append([true_option_id], false_options_ids)
                np.random.shuffle(all_options)
                question = (
                    question_prompt
                    + "\n"
                    + "\n".join([f"({i+1}) {action_id_mapping[option]}" for i, option in enumerate(all_options)])
                )
                questions.append(question)
                answers.append(action_id_mapping[true_option_id])
            return questions, answers

        video_paths = []
        video_ids = []
        mcqs = []
        mcq_labels = []

        labels_dict = {video_info["id"]: video_info for video_info in labels}

        for subset_video_info in tqdm(subset_labels):
            video_id = subset_video_info["id"]

            if video_id in labels_dict:  # ensure video_id also in labels
                video_info = labels_dict[video_id]  # get corresponding video_id
                video_path = f"{videos_path}/{video_id}.mp4"

                # 生成 MCQs，仅使用 subset_labels 里的动作
                video_questions, video_answers = generate_questions(
                    [action[0] for action in video_info["actions"]],  # all true actions in video
                    [action[0] for action in subset_video_info["actions"]],  # subset true actions in video
                )

            video_paths.extend([video_path] * len(video_answers))
            video_ids.extend([video_info["id"]] * len(video_answers))
            mcqs.extend(video_questions)
            mcq_labels.extend(video_answers)

        return {
            "video_paths": video_paths,
            "video_ids": video_ids,
            "mcqs": mcqs,
            "mcq_labels": mcq_labels,
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
        video = fetch_video(video_info)
        question = self.data["mcq_data"]["mcqs"][index]

        return index, video, question, answer

    def __len__(self):
        return self.data["n_samples"]
