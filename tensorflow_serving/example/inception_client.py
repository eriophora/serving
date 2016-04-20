# Copyright 2016 Google Inc. All Rights Reserved.
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

#!/usr/grte/v4/bin/python2.7

"""Send JPEG image to inception_inference server for classification.
"""

import os
import sys
import threading

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy
import tensorflow as tf
from tensorflow.python.platform.logging import warn
from PIL import Image

from tensorflow_serving.example import inception_inference_pb2


tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'inception_inference service host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
tf.app.flags.DEFINE_integer('image_size', 299,
              """Needs to provide same value as in training.""")
tf.app.flags.DEFINE_string('prep_method', 'resize',
               '''Defines the method used to get images to image_size:
                - resize: Resize the image (distortion, whole image, no blank 
                  space)
                - crop: Center-crop the image to image_size (no distortion, 
                  partial image, no blank space)
                - padresize: Pads the image to the appropriate aspect ratio 
                then resizes (no distortion, whole image, blank space)''')

FLAGS = tf.app.flags.FLAGS

if FLAGS.prep_method not in ['resize', 'crop', 'padresize']:
  warn('Preprocessing method "%s" is unknown. Defaulting to resize.', 
       FLAGS.prep_method)
  FLAGS.prep_method = 'resize'

NUM_CLASSES = 5
WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
SYNSET_FILE = os.path.join(WORKING_DIR, 'imagenet_lsvrc_2015_synsets.txt')
METADATA_FILE = os.path.join(WORKING_DIR, 'imagenet_metadata.txt')


def prep_image(img, w=FLAGS.image_size, h=FLAGS.image_size):
  '''
  Preprocesses the requested image to the desired size, permitting server-side
  batching of Inception.
  '''
  if FLAGS.prep_method == 'resize':
    return _resize_to(img, w=w, h=h)
  elif FLAGS.prep_method == 'crop':
    # resize to appropriate dimensions
    resized_im = _resize_to_min(img, w=w, h=h)
    # center crop
    return _center_crop_to(resized_im, w=w, h=h)
  elif FLAGS.prep_method == 'padresize':
    des_asp = float(w) / h
    # pad the image
    padded_im = _pad_to_asp(img, des_asp)
    # resize the image
    return _resize_to(padded_im, w=w, h=h)


def _resize_to(img, w=None, h=None):
  '''
  Resizes the image to a disired width and height. If either is undefined,
  it resizes such that the defined argument is satisfied and preserves aspect
  ratio. If both are defined, resizes to satisfy both arguments without
  preserving aspect ratio.  

  Args:
    img: A PIL image.
    w: The desired width.
    h: The desired height.
  '''
  ow, oh = img.size
  asp = float(ow) / oh
  if w is None and h is None:
    # do nothing
    return img
  elif w is None:
    # set the width
    w = int(h * asp)
  elif h is None:
    h = int(w / asp)
  return img.resize((w, h), Image.BILINEAR)


def _resize_to_min(img, w=None, h=None):
  '''
  Resizes an image so that its size in both dimensions is greater than or 
  equal to the provided arguments. If either argument is None, that dimension 
  is ignored. If the image is larger in both dimensions, then the image is
  shrunk. In either case, the aspect ratio is preserved and image size is
  minimized.

  Args:
    img: A PIL image.
    w: The minimum width desired.
    h: The minimum height desired.
  '''
  ow, oh = img.size
  if w is None and h is None:
    return img
  if w is None:
    # resize to the desired height
    return _resize_to(img, h=h)
  elif h is None:
    # resize to the desired width
    return _resize_to(img, w=w)
  if ow == w and oh == h:
    # then you need not do anything
    return img
  hf = h / float(oh)  # height scale factor
  wf = w / float(ow)  # width scale factor
  if min(hf, wf) < 1.0:
    # then some scaling up is necessary. Scale up by as much as needed,
    # leaving one dimension larger than the requested amount if required.
    scale_factor = max(hf, wf)
  else:
    # scale down by the least amount to ensure both dimensions are larger
    scale_factor = min(hf, wf)
  nw = int(ow * scale_factor)
  nh = int(oh * scale_factor)
  return _resize_to(img, w=nw, h=nh)


def _center_crop_to(img, w, h):
  '''
  Center crops image to desired size. If either dimension of the image is
  already smaller than the desired dimensions, the image is not cropped.

  Args:
    img: A PIL image.
    w: The width desired.
    h: The height desired.
  '''
  ow, oh = img.size
  if ow < w or oh < h:
    return img
  upper = (h - oh) / 2
  lower = cy1 + h
  left = (w - ow) / 2
  right = cx1 + w
  return img.crop((left, upper, right, lower))


def _pad_to_asp(img, asp):
  '''
  Symmetrically pads an image to have the desired aspect ratio.

  Args:
    img: A PIL image.
    asp: The aspect ratio, a float, as w / h
  '''
  ow, oh = img.size
  oasp = float(ow) / oh
  if asp > oasp:
    # the image is too narrow. Pad out width.
    nw = int(oh * asp)
    left = (nw - ow) / 2
    upper = 0
    newsize = (nw, oh)
  elif asp < oasp:
    # the image is too short. Pad out height.
    nh = int(ow / asp)
    left = 0
    upper = (nh - oh) / 2
    newsize = (ow, nh)
  nimg = Image.new(img.mode, newsize)
  nimg.paste(img, box=(left, upper))
  return nimg


def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = inception_inference_pb2.beta_create_InceptionService_stub(channel)
  # Create label->synset mapping
  synsets = []
  with open(SYNSET_FILE) as f:
    synsets = f.read().splitlines()
  # Create synset->metadata mapping
  texts = {}
  with open(METADATA_FILE) as f:
    for line in f.read().splitlines():
      parts = line.split('\t')
      assert len(parts) == 2
      texts[parts[0]] = parts[1]
  # Send request
  with open(FLAGS.image, 'rb') as f:
    # See inception_inference.proto for gRPC request/response details.
    data = f.read()
    request = inception_inference_pb2.InceptionRequest()
    request.jpeg_encoded = data
    result = stub.Classify(request, 10.0)  # 10 secs timeout
    for i in range(NUM_CLASSES):
      index = result.classes[i]
      score = result.scores[i]
      print '%f : %s' % (score, texts[synsets[index - 1]])


def checkerboard(w=256, h=256, c0=0, c1=255, blocksize=16):
  '''
  Generates a checkerboard pattern, for testing the preprocessing.

  Returns the pattern as a PIL image.
  '''
  block = numpy.tile([1], (blocksize, blocksize))
  block1 = block * c0
  block2 = block * c1
  sb1 = numpy.hstack([block1, block2])
  sb2 = numpy.hstack([block2, block1])
  sb = numpy.vstack([sb1, sb2])
  d = blocksize * 2
  rep_w = numpy.ceil(float(w) / d)
  rep_h = numpy.ceil(float(h) / d)
  board = numpy.tile(sb, (rep_h, rep_w))[:h, :w]
  return Image.fromarray(board.astype(float))


if __name__ == '__main__':
  tf.app.run()
