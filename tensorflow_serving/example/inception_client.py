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

"""
Send JPEG image to inception_inference server for classification. This
client can handle images with very different aspect ratios from the
original training material by using alternative preprocessing methods to
simple resizing, such as padding (symmetrically adding 0s to the edges
of the image to bring it to a square shape) or centrally cropping. Both
of these methods prevent heavy distortion from being introduced into the
image, but also carry other disadvantages (see options).
"""

import os
import sys
import threading

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy as np
import tensorflow as tf
from tensorflow.python.platform.logging import warn
from PIL import Image

from tensorflow_serving.example import inception_inference_pb2


tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
# TODO: Add support for concurrency requests.
tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'inception_inference service host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
tf.app.flags.DEFINE_string('image_list_file', '', 'path to a text file containing a list of images')
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


def _prep_image(img, w=FLAGS.image_size, h=FLAGS.image_size):
  '''
  Preprocesses the requested image to the desired size, permitting server-side
  batching of Inception. The original preprocessing operation for Inception is
  to central crop by 87.5%, and then simply resize to the appropriate
  dimensions. Here, there are additional options for users who have retrained
  Inception using alternative preprocessing methods. The source image cropping
  of 87.5% is presumed completed; this is intended to be a more general prep 
  function.

  Args:
    img: A PIL image.
    w: The desired width.
    h: The desired height.
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
  minimized. If the target of interest is in the center of the frame, but the
  image has an unusual aspect ratio, center cropping is likely the best option.
  If the image has an unusual aspect ratio but is irregularly framed, padding
  the image will prevent distortion while also including the entirety of the
  original image.

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


def prep_inception_from_file(image_file):
  '''
  Preprocesses an image from a fully-qualified file, in same
  manager as the batchless inception server (including the 
  87.5% source crop) and wraps _prep_image to make the image
  the correct size.
  '''
  # Load the image.
  try:
    image = Image.open(FLAGS.image)
  except IOError, e:
    warn('Could not open %s with PIL. It will be skipped!' % e.filename)
    return None

  # In the original implementation of Inception export, the images are
  # centrally cropped by 87.5 percent before undergoing adjustments to
  # bring them into the correct size, which we replicate here.
  ow, oh = image.size  # obtain the original width and height
  nw = int(ow * 0.875)  # compute the new width
  nh = int(oh * 0.875)  # compute the new height
  image = _center_crop_to(image, w=nw, h=nh)  # center crop to 87.5%

  # preprocess the image to bring it to a square with edge length
  # FLAGS.image_size
  image = prep_image(data)

  # Convert to a numpy array
  image = numpy.array(image).astype(float32)

  # Perform additional preprocessing to mimic the inputs to inception.
  # Scale image pixels. all pixels now reside in [0, 1), as in the
  # tensor representation following tf.image.decode_jpeg.
  image /= 256.

  # Scale the image to the domain [-1, 1) (referred to incorrectly
  # as (-1, 1) in the original documentation).
  image -= 0.5
  image *= 2.0
  return image


def do_inference(hostport, concurrency, listfile):
  '''
  Performs inference over multiple images given a list of images
  as a text file, with one image per line. The image path cannot
  be relative and must be fully-qualified. Prints the results of
  the top N classes.

  Args:
    hostport: Host:port address of the mnist_inference service.
    concurrency: Maximum number of concurrent requests.
    listfile: The path to a text file containing the fully-qualified
      path to a single image per line.

  Returns:
    None.
  '''
  imagefns = []
  with open(listfile, 'r') as f:
    imagefns = f.read().splitlines()
  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = inception_inference_pb2.beta_create_InceptionService_stub(channel)
  cv = threading.Condition()
  # this will store the ouput Inception. We require it to map filenames
  # to their labels in the case of batching.
  inference_results = []
  def done(result_future, filename):
    '''
    Callback for result_future, modifies inference_results to hold the 
    output of Inception.
    '''
    with cv:
      exception = result_future.exception()
      if exception:
        result['error'] += 1
        print exception
      result['done'] += 1
      result['active'] -= 1
      indices = [result.classes[i] for i in range(NUM_CLASSES)]
      scores = [result.scores[i] for i in range(NUM_CLASSES)]
      inf_res = [filename, indices, scores]
      inference_results.append(inf_res)
      cv.notify()

  for imagefn in imagefns:
    image_array = prep_inception_from_file(imagefn)
    if image_array is None:
      continue
    request.image_data = image_array.tobytes()
    with cv:
      while result['active'] == concurrency:
        cv.wait()
      result['active'] += 1
    result_future = stub.Classify.future(request, 10.0)  # 10 second timeout
    result_future.add_done_callback(
        lambda result_future, filename=imagefn: done(result_future))  # pylint: disable=cell-var-from-loop
  with cv:
    while result['done'] != num_tests:
      cv.wait()
  return inference_results


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
  if FLAGS.image:
    # Load and preprocess the image.
    image = prep_inception_from_file(FLAGS.image)
    if image is None:
      return
    # The image is now a numpy nd array with the appropraite size for 
    # Inception, with each element constrained to the domain [-1, 1).

    # Create the request. See inception_inference.proto for gRPC request/
    # response details. Instead of using an encoded jpeg, we send the
    # data as a row-major byte encoding using numpy's tobytes method.
    request.image_data = image.tobytes()
    result = stub.Classify(request, 10.0)  # 10 secs timeout
    for i in range(NUM_CLASSES):
      index = result.classes[i]
      score = result.scores[i]
      print '%f : %s' % (score, texts[synsets[index - 1]])
  elif FLAGS.image_list_file:
    inference_results = do_inference(FLAGS.server, 
                                     FLAGS.concurrency, 
                                     FLAGS.image_list_file)
    for filename, indices, scores in inference_results:
      print '%s Inference:'
      for index, score in zip(indices, scores):
        print '\t%f : %s' % (score, texts[synsets[index - 1]])

if __name__ == '__main__':
  tf.app.run()
