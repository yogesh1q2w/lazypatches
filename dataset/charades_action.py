""" Dataset loader for the Charades dataset """

import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import json

from torchvision import io
from typing import Dict


def parse_video_action_csv(filename):
    def extract_details(annot):
        actions = [
            action_desc.split(" ") for action_desc in str(annot["actions"]).split(";")
        ]
        actions = [
            (item[0], float(item[1]), float(item[2]))
            for item in actions
            if item[0] != "nan"
        ]
        descriptions = annot["descriptions"].split(";")
        objects = annot["objects"].split(";")
        retval = {
            annot["id"]: {
                "actions": actions,
                "descriptions": descriptions,
                "scene": annot["scene"],
                "objects": objects,
                "length": float(annot["length"]),
            }
        }
        return retval

    df = pd.read_csv(filename)
    labels = df.apply(lambda annot: extract_details(annot), axis=1)
    return labels


def parse_action_id_mapping(filename):
    action_id_mapping = {}
    with open(filename) as f:
        for line in f:
            # Split on whitespace, assuming first word is class_id of length 4
            if len(line) > 0:
                action_id_mapping[line[:4]] = line[5:-1]

    return action_id_mapping


def fetch_video(ele: Dict, nframe_factor=2):
    if isinstance(ele["video"], str):

        def round_by_factor(number: int, factor: int) -> int:
            # a rounding function to have appropriate linspace
            return round(number / factor) * factor

    video = ele["video"]
    if video.startswith("file://"):
        video = video[7:]

    video, _, info = io.read_video(
        video,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    assert not (
        "fps" in ele and "nframes" in ele
    ), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        # if total frames are specified for processed video
        nframes = round_by_factor(ele["nframes"], nframe_factor)
    else:
        # if target fps is specified for processed video
        fps = ele.get("fps", 2.0)
        nframes = round_by_factor(
            video.size(0) / info["video_fps"] * fps, nframe_factor
        )

    idx = torch.linspace(0, video.size(0) - 1, nframes, dtype=torch.int64)
    return video[idx]


class CharadesActionMCQ(data.Dataset):
    def __init__(
        self,
        dataset_path=None,
        videos_path=None,
        labels_path=None,
        classes_path=None,
        n_wrong_options=None,
        reload=True,
        target_fps=2.0
    ):
        # give either already exisiting dataset or corresponding paths to build one
        if reload:
            assert videos_path is None and labels_path is None and classes_path is None and n_wrong_options is None
            self.data = json.load(dataset_path)
        else:
            assert dataset_path is not None and videos_path is not None and labels_path is not None and classes_path is not None and n_wrong_options is not None
        
            self.labels = parse_video_action_csv(labels_path)
            self.action_id_mapping = parse_action_id_mapping(classes_path)
            self.num_action_classes = len(self.action_id_mapping)
            self.videos_path = videos_path
            self.n_wrong_options = n_wrong_options
            
            mcq_data = self.prepare()

            self.data = {"videos_path": self.videos_path,
                         "labels_path": labels_path,
                         "classes_path": classes_path,
                         "n_wrong_options": self.n_wrong_options,
                         "mcq_data": mcq_data,
                         "n_samples": len(mcq_data["mcqs"])
                         }
            json.dump(self.data, dataset_path)
        self.target_fps = target_fps
        
    def prepare(self):
        question_prompt = open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "../utils/action_mcq.txt"
            ),
            "r",
        ).read()

        def generate_questions(true_actions_ids):
            false_action_ids = list(
                set(self.action_id_mapping.keys()) - set(true_actions_ids)
            )
            questions = []
            answers = []
            for true_option_id in true_actions_ids:
                false_options_ids = np.random.choice(
                    false_action_ids, size=self.num_wrong_options, replace=False
                )
                all_options = [true_option_id] + false_options_ids
                np.random.shuffle(all_options)
                question = (
                    question_prompt
                    + "\n"
                    + "\n".join(
                        [
                            f"({i+1}) {self.action_id_mapping[option]}"
                            for i, option in enumerate(all_options)
                        ]
                    )
                )
                questions.append(question)
                answers.append(self.action_id_mapping[true_option_id])
            return questions, answers

        video_paths = []
        video_ids = []
        mcqs = []
        mcq_labels = []

        for video_id, annot in self.labels.items():

            video_path = "{}/{}.mp4".format(self.videos_path, video_id)
            video_questions, video_answers = generate_questions(annot["actions"])

            video_paths.extend([video_path] * len(video_answers))
            video_ids.extend([video_id] * len(video_answers))
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
        quetion = self.data["mcq_data"]["mcqs"][index]

        return video, quetion, answer

    def __len__(self):
        return len(self.data["data"]["n_samples"])
