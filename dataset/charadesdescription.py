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
                actions = [{'class': x, 'start': float(
                    y), 'end': float(z)} for x, y, z in actions]
            labels[vid] = actions
    return labels

def parse_charades_csv_video(filename):
    labels = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row['id']
            description = row['descriptions']
            labels[vid] = description
    return labels


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
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], nframe_factor)
    else:
        fps = ele.get("fps", 1.0)
        nframes = round_by_factor(video.size(0) / info["video_fps"] * fps, nframe_factor)
    print(video.size(0), info["video_fps"], nframes)
    idx = torch.linspace(0, video.size(0) - 1, nframes, dtype=torch.int64)
    return video[idx]


class Charades_decription(data.Dataset):
    def __init__(self, root, split, labelpath, cachedir, transform=None, target_transform=None):
        self.num_classes = 157
        self.transform = transform
        self.target_transform = target_transform
        self.labels = parse_charades_csv_video(labelpath)
        self.root = root
        # cachename = '{}/{}_{}.pkl'.format(cachedir,
                                        #   self.__class__.__name__, split)
        # self.data = cache(cachename)(self.prepare)(root, self.labels)
        self.data = self.prepare(root, self.labels)
    
    def prepare(self, path, labels):
        datadir = path
        video_paths, targets, ids = [], [], []

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
            video_paths.append(video_path)
            targets.append(label)
            ids.append(vid)
                
        return {'video_paths': video_paths, 'targets': targets, 'ids': ids}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (frames, conversation, target_description).
        """
        path = self.data['video_paths'][index]
        target = self.data['targets'][index]
        meta = {}
        meta['id'] = self.data['ids'][index]
        
        video_info = {"type": "video", "video": path, "fps": 1.0}
        video = fetch_video(video_info)
        conversation = [
            {
                "role":"user",
                "content":[
                    {"type":"video"},
                    {"type":"text", "text":"Describe this video."}
                ]
            }
        ]
        print(path)
        
        return video, conversation, target

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

