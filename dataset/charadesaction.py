""" Dataset loader for the Charades dataset """
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
from glob import glob
import csv
import pickle
import os
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


def cls2int(x):
    return int(x[1:])


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# def accimage_loader(path):
#     import accimage
#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)


# def default_loader(path):
#     from torchvision import get_image_backend
#     if get_image_backend() == 'accimage':
#         return accimage_loader(path)
#     else:
#         return pil_loader(path)


def cache(cachefile):
    """ Creates a decorator that caches the result to cachefile """
    def cachedecorator(fn):
        def newf(*args, **kwargs):
            print('cachefile {}'.format(cachefile))
            if os.path.exists(cachefile):
                with open(cachefile, 'rb') as f:
                    print("Loading cached result from '%s'" % cachefile)
                    return pickle.load(f)
            res = fn(*args, **kwargs)
            with open(cachefile, 'wb') as f:
                print("Saving result to cache '%s'" % cachefile)
                pickle.dump(res, f)
            return res
        return newf
    return cachedecorator

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

    # print(info)
    # print(video.shape)
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
        # print(Cs)
        Qs.append('\n'.join([Q] + Cs_with_numbers))
        As.append(Cs_with_numbers[tc_index])
    return Qs, As

class Charades_action(data.Dataset):
    def __init__(self, root, split, labelpath, classespath, cachedir, transform=None, target_transform=None):
        self.num_classes = 157
        self.transform = transform
        self.target_transform = target_transform
        self.labels = parse_charades_csv(labelpath)
        self.classes = parse_charades_classes(classespath)
        self.root = root
        # cachename = '{}/{}_{}.pkl'.format(cachedir,
                                        #   self.__class__.__name__, split)
        # self.data = cache(cachename)(self.prepare)(root, self.labels)
        self.has_printed = False 
        self.has_printed1 = False 
        
        self.data = self.prepare(root, self.labels, self.classes, self.has_printed, self.has_printed1)
    
    def prepare(self, path, labels, classes, has_printed, has_printed1):
        datadir = path
        video_paths, correct_actions_list,wrong_actions_list,qa_list,as_list, ids = [], [], [], [], [], []
        for i, (vid, label) in enumerate(labels.items()):
            # iddir = datadir + '/' + vid
            # lines = glob(datadir)

            # n = len(lines)
            # print("video_length:",n)
            # if i % 100 == 0:
            #     print("{} {}".format(i, datadir))
            # if n == 0:
            #     continue
            
            video_path = '{}/{}.mp4'.format(
                            datadir, vid)
            # print(label)
            classes_ids = [l["class"] for l in label]
            # print(classes_ids)
            # print(classes)
            # correct_actions = [labels.get(class_id, "Unknown action") for class_id in class_ids]
            correct_actions = [classes[class_id] for class_id in classes_ids if class_id in classes]
            if not has_printed:
                print(correct_actions)
                has_printed = True
            # print(correct_actions)
            wrong_actions = [action for class_id, action in classes.items() if class_id not in classes_ids]
            if not has_printed1:
                print(wrong_actions)
                has_printed1 = True
            # print(wrong_actions)
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
        meta = {}
        path = self.data['video_paths'][index]
        # print(path)
        answer = self.data['answers'][index]

       
        # meta['id'] = self.data['ids'][index]
        
        video_info = {"type": "video", "video": path, "fps": 1.0}
        video = fetch_video(video_info)
        quetion =  self.data['questions'][index]
        print(path)
        
        return video, quetion, answer

    def __len__(self):
        # print(len(self.data['video_paths']))
        return len(self.data['video_paths'])

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

