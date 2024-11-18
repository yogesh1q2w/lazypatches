import os
import json
import random
from typing import Tuple, List, Dict
import zipfile

import torch
from torch.utils.data import Dataset
import cv2
import imageio
import matplotlib.pyplot as plt
import moviepy.editor as mvp
import numpy as np

class PerceptionTestLoader(Dataset):
    def __init__(self, video_ids_file, annotations_path, split="train"):
        self.video_ids = json.load(open(video_ids_file, "rb"))
        self.annotations = json.load(open(annotations_path, "rb"))


    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        pass