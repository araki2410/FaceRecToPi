#!/usr/bin/python3.4
# -*- coding: utf-8 -*-
"""
画像特徴量を取得する
"""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import facenet
import align.detect_face
import pickle
from scipy import misc

def main(args):
    args_filepaths = args.image_files
    image_size = args.image_size
    margin = args.margin
    gpu_memory_fraction = args.gpu_memory_fraction
    model = args.model

    batch_size = args.batch_size

    embs = []
    extracted_filepaths = []

    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Load the model
            facenet.load_model(model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            for i in range(0, len(args_filepaths), batch_size):
                target_filepaths = args_filepaths[i:i+batch_size]
                print("target_filepaths len:{}".format(len(target_filepaths)))
                images, target_filepaths = load_and_align_data(target_filepaths, image_size, margin, gpu_memory_fraction)
                print("target_filepaths len:{}".format(len(target_filepaths)))
                # Run forward pass to calculate embeddings
                feed_dict = { images_placeholder: images, phase_train_placeholder:False }
                emb = sess.run(embeddings, feed_dict=feed_dict)
                print("emb len:{}".format(len(emb)))

                for j in range(len(target_filepaths)):
                    extracted_filepaths.append(target_filepaths[j])
                    embs.append(emb[j, :])

    save_embs(embs, extracted_filepaths)  

def save_embs(embs, paths):
    # 特徴量の取得
    reps = {}
    for i, (emb, path) in enumerate(zip(embs, paths)):
        #print('%1d: %s' % (i, paths))
        #print(emb)
        try:
            basename = os.path.basename(path)
            reps[basename] = emb
        except:
            print('error %1d: %s' % (i, path) )
    # 特徴量の保存
    with open('img_facenet.pkl', 'wb') as f:
        pickle.dump(reps, f)


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):
    # 処理が正常に行えた画像パス
    extracted_filepaths = []

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    nrof_samples = len(image_paths)
    img_list = [] #[None] * nrof_samples
    for i in range(nrof_samples):
        print('%1d: %s' % (i, image_paths[i]))
        img = misc.imread(os.path.expanduser(image_paths[i]))
        img_size = np.asarray(img.shape)[0:2]
        try:
            bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            det = np.squeeze(bounding_boxes[0,0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            #img_list[i] = prewhitened
            img_list.append(prewhitened)
            extracted_filepaths.append(image_paths[i])
        except:
            print("cannot extract_image_align")

    image = np.stack(img_list)
    return image, extracted_filepaths

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--batch_size', type=int,
        help='Batch size for extraction image emb', default=1000)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
