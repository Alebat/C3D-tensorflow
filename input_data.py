# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

from src.exp.bboxes import read_images


def read_batches_of_clips(paths, batch_size, num_frames_per_clip=16, stride=8, resize_size=112):
    clips = []
    names = []
    for clip, name in read_clips(paths, num_frames_per_clip, stride, resize_size):
        if len(clip) >= batch_size:
            yield np.array(clips), names
        clips.append(clip)
        names.append(name)


def read_clips(paths, num_frames_per_clip, stride, resize_size):
    assert stride < num_frames_per_clip, "stride < num_frames_per_clip"
    for path in paths:
        vc = cv2.VideoCapture(path)
        clip = []
        first_frame_n = 0
        for frame in read_images(vc):
            if len(clip) >= num_frames_per_clip:
                yield np.array(clip).astype(np.float32), (path, first_frame_n)
                clip = clip[:num_frames_per_clip - stride]
                first_frame_n += stride
            img = np.array(cv2.resize(np.array(frame), (resize_size, resize_size))).astype(np.float32)
            clip.append(img)
        # if piece of clip remains then pad it
        if len(clip) > 0:
            for frame in range(len(clip), num_frames_per_clip):
                clip[frame] = clip[-1]
            yield np.array(clip).astype(np.float32), (path, first_frame_n)
