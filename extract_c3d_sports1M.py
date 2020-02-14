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

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import os
import tensorflow as tf
# Basic model parameters as external flags.
import torch

from C3D_tensorflow import c3d_model
from src.util.p3d_model import P3D199

BATCH_SIZE = 10


def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           c3d_model.NUM_FRAMES_PER_CLIP,
                                                           c3d_model.CROP_SIZE,
                                                           c3d_model.CROP_SIZE,
                                                           c3d_model.CHANNELS))
    labels_placeholder = tf.placeholder(tf.int64, shape=batch_size)
    return images_placeholder, labels_placeholder


def _variable_on_cpu(name, shape, initializer):
    # with tf.device('/cpu:%d' % cpu_id):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var) * wd
        tf.add_to_collection('losses', weight_decay)
    return var


def read_batches_of_clips(videos, batch_size, augment, num_frames_per_clip=16, stride=8, resize_size=112):
    clips = read_clips(videos, num_frames_per_clip, stride, resize_size, augment)
    batch = []
    descriptors = []
    for clip, path, frame in clips:
        if len(batch) >= batch_size:
            yield np.array(batch), descriptors
            batch = []
            descriptors = []
        batch.append(clip)
        descriptors.append((path, frame))


def read_clips(videos, num_frames_per_clip, stride, resize_size, augment):
    assert stride < num_frames_per_clip, "stride < num_frames_per_clip"
    for images, path in videos:
        images = list(images)
        yield from make_clips(images, num_frames_per_clip, path, resize_size, stride, False, False)
        if augment:
            yield from make_clips(images, num_frames_per_clip, path, resize_size, stride, True, False)
            yield from make_clips(images, num_frames_per_clip, path, resize_size, stride, False, True)
            yield from make_clips(images, num_frames_per_clip, path, resize_size, stride, True, True)


# Padding is 'hold'
def make_clips(images, num_frames_per_clip, path, resize_size, stride, mirror, half_fps):
    name = path
    if mirror:
        name = f'{name}.mirror'
    if half_fps:
        name = f'{name}.lfps'
    clip = []
    first_frame_n = 0
    for frame_num, frame in images:
        if len(clip) >= num_frames_per_clip:
            yield np.array(clip).astype(np.float32), name, first_frame_n
            clip = clip[:num_frames_per_clip - stride]
            first_frame_n += stride
        # skip odd frames if required
        if not half_fps or frame_num % 2 == 0:
            frame = cv2.resize(frame, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)
            if mirror:
                frame = cv2.flip(frame, 1)
            img = np.array(frame).astype(np.float32)
            clip.append(img)
    # if piece of clip remains then pad it
    if len(clip) > 0:
        for frame in range(len(clip), num_frames_per_clip):
            clip.append(clip[-1])
        yield np.array(clip).astype(np.float32), name, first_frame_n


def run_test():
    model_name = "./data/conv3d_deepnetA_sport1m_iter_1900000_TF.model"
    path = '6'
    test_videos = list(
        map(
            lambda x: os.path.join(path, x),
            filter(lambda x: x[-4:] == '.mov',
                   os.listdir(path)
                   )
        )
    )
    print("Number of test videos={}".format(len(test_videos)))

    batch_size = BATCH_SIZE

    # Get the sets of images and labels for training, validation, and
    images_placeholder, labels_placeholder = placeholder_inputs(batch_size)
    with tf.variable_scope('var_name') as var_scope:
        weights = {
            'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
            'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
            'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
            'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
            'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
            'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
            'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
            'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.04, 0.005)
        }
        biases = {
            'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
            'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
            'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
            'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
            'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
            'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
            'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
            'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
            'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
            'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0),
            'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.04, 0.0),
        }
    logits = []
    with tf.device('/gpu:1'):
        logit = c3d_model.inference_c3d(
            images_placeholder[0:BATCH_SIZE, :, :, :, :], 0.6,
            BATCH_SIZE, weights, biases)
        logits.append(logit)
    logits = tf.concat(logits, 0)
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)
    # Create a saver for writing training checkpoints.
    saver.restore(sess, model_name)
    # And then after everything is built, start the training loop.
    for clips, names in read_batches_of_clips(test_videos, batch_size):
        prediction = logits.eval(
            session=sess,
            feed_dict={images_placeholder: clips}
        )
        print(prediction.shape)
        print(prediction)
    print("done")


def main(_):
    run_test()


if __name__ == '__main__':
    tf.app.run()


def get_weights_and_biases():
    with tf.variable_scope('var_name'):
        weights = {
            'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
            'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
            'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
            'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
            'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
            'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
            'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
            'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.04, 0.005)
        }
        biases = {
            'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
            'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
            'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
            'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
            'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
            'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
            'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
            'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
            'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
            'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0),
            'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.04, 0.0),
        }
    return biases, weights


def extract_c3d(batch_size, device, model_name, named_videos, augment=False):
    with torch.no_grad():
        images_placeholder = tf.placeholder(
            tf.float32,
            shape=(batch_size,
                   c3d_model.NUM_FRAMES_PER_CLIP,
                   c3d_model.CROP_SIZE,
                   c3d_model.CROP_SIZE,
                   c3d_model.CHANNELS))
        biases, weights = get_weights_and_biases()
        with tf.device(device):
            logit = c3d_model.inference_c3d(
                images_placeholder[0:BATCH_SIZE, :, :, :, :], 0.6,
                BATCH_SIZE, weights, biases)
        saver = tf.train.Saver()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, model_name)
        for clips, descriptors in read_batches_of_clips(named_videos, batch_size, augment=augment):
            predictions = logit.eval(
                session=sess,
                feed_dict={images_placeholder: clips}
            )
            yield from zip(predictions, descriptors)


def extract_p3d(batch_size, named_videos, augment=False):
    pretrained_file = 'data/p3d_rgb_199.checkpoint.pth.tar'
    weights = torch.load(pretrained_file)['state_dict']
    with torch.no_grad():
        model = P3D199(weights).cuda()
        model.eval()

        for clips, descriptors in read_batches_of_clips(named_videos, batch_size, augment=augment, resize_size=160):
            predictions = model(
                torch.FloatTensor(clips).cuda().permute(0, 4, 1, 2, 3)
            )
            yield from zip(predictions, descriptors)
