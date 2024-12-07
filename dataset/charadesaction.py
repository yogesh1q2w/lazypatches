""" Dataset loader for the Charades dataset """
import torch
import torch.utils.data as data
from glob import glob
import csv
import random

from torchvision import io
from typing import Dict


def parse_charades_csv(filename):
    labels = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row['id']
            actions = row['actions']
            if actions == '':
                actions = []
            else:
                actions = [a.split(' ') for a in actions.split(';')]
                actions = [{'class': x} for x, y, z in actions]
            labels[vid] = actions
    return labels

def parse_charades_classes(filename):
    classes = {}
    with open(filename) as f:
        for line in f:
            # Split on whitespace, assuming first word is class_id
            parts = line.split(maxsplit=1)
            if len(parts) == 2:  # Ensure the line has a class_id and description
                class_id, action = parts
                classes[class_id] = action.strip()
            else:
                print(f"Skipping malformed line: {line}")
    return classes

def fetch_video(ele: Dict, nframe_factor=2):
    if isinstance(ele['video'], str):
        def round_by_factor(number: int, factor: int) -> int:
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
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], nframe_factor)
    else:
        fps = ele.get("fps", 1.0)
        nframes = round_by_factor(video.size(0) / info["video_fps"] * fps, nframe_factor)
    print(video.size(0), info["video_fps"], nframes)
    idx = torch.linspace(0, video.size(0) - 1, nframes, dtype=torch.int64)
    return video[idx]

def genQ(T_actions, F_actions):
    Qs = []
    As = []
    Q = 'Which of these action occur in the video? Please respond with the correct option only.'
    num_F = len(F_actions)
    num_FC = 3
    for T in T_actions:
        Cs = []
        Cs.append(T)
        FC = random.sample(range(num_F), num_FC)
        Cs.extend([F_actions[i] for i in FC])
        random.shuffle(Cs)
        tc_index = Cs.index(T)
        Cs_with_numbers = [f"{i + 1}. {option}" for i, option in enumerate(Cs)]
        Qs.append('\n'.join([Q] + Cs_with_numbers))
        As.append(Cs_with_numbers[tc_index])
    return Qs, As

class Charades_action(data.Dataset):
    def __init__(self, root, labelpath, classespath, transform=None, target_transform=None):
        self.num_classes = 157
        self.transform = transform
        self.target_transform = target_transform
        self.labels = parse_charades_csv(labelpath)
        self.classes = parse_charades_classes(classespath)
        self.root = root
        
        self.data = self.prepare(root, self.labels, self.classes)
    
    def prepare(self, path, labels, classes, has_printed, has_printed1):
        datadir = path
        video_paths, correct_actions_list,wrong_actions_list,qa_list,as_list, ids = [], [], [], [], [], []
        for i, (vid, label) in enumerate(labels.items()):
            
            video_path = '{}/{}.mp4'.format(
                            datadir, vid)
            classes_ids = [l["class"] for l in label]

            correct_actions = [classes[class_id] for class_id in classes_ids if class_id in classes]

            wrong_actions = [action for class_id, action in classes.items() if class_id not in classes_ids]

            for action in correct_actions:
                video_paths.append(video_path)
            correct_actions_list.append(correct_actions)
            wrong_actions_list.append(wrong_actions)
            ids.append(vid)
            QA, AS = genQ(correct_actions, wrong_actions)
            qa_list.extend(QA)
            as_list.extend(AS)
                
        return {'video_paths': video_paths, 'questions': qa_list, 'answers': as_list, 'ids': ids}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (frames, conversation, target_description).
        """
        path = self.data['video_paths'][index]
        answer = self.data['answers'][index]
        
        video_info = {"type": "video", "video": path, "fps": 1.0}
        video = fetch_video(video_info)
        quetion =  self.data['questions'][index]
        
        return video, quetion, answer

    def __len__(self):
        return len(self.data['video_paths'])
