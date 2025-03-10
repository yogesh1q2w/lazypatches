import logging
import math

import torch
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode


logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 225 * 28 * 28
VIDEO_TOTAL_PIXELS = 225 * 28 * 28
FRAME_FACTOR = 2
FPS = 2  # THIS REMAINS FIXED, ARGUMENT FPS IS FOR SECOND FETCH_VIDEO
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 30


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(height, width, factor, min_pixels, max_pixels):
    """
    Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def smart_nframes(target_fps, total_frames, original_fps) -> int:
    """Calculate nframes for target_fps from original_fps"""

    min_frames = ceil_by_factor(FPS_MIN_FRAMES, FRAME_FACTOR)
    max_frames = floor_by_factor(min(FPS_MAX_FRAMES, total_frames), FRAME_FACTOR)
    nframes = total_frames / original_fps * target_fps
    nframes = min(max(nframes, min_frames), max_frames)
    nframes = round_by_factor(nframes, FRAME_FACTOR)

    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")

    return nframes


def _read_video_torchvision(video_path):
    video, audio, info = io.read_video(
        video_path,
        start_pts=0.0,
        end_pts=None,
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_original_fps = video.size(0), info["video_fps"]
    logger.info(f"Reading video:  {video_path=}, {total_frames=}, {video_original_fps=}")
    nframes = smart_nframes(target_fps=FPS, total_frames=total_frames, original_fps=video_original_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    video = video[idx]
    return video


def fetch_video(path):
    video = _read_video_torchvision(path)
    nframes, _, height, width = video.shape
    min_pixels = VIDEO_MIN_PIXELS
    max_pixels = max(min(VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS / nframes * FRAME_FACTOR), int(VIDEO_MIN_PIXELS * 1.05))

    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    return video
