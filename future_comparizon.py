#!/usr/bin/python3
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
#sys.path.append("/home/yanai-lab/araki-t/Git/facenet/src/")
import os
import argparse
import facenet
import facenets.src.align.detect_face
import pickle
import scipy
from scipy import misc
img_paths_list = [] #{./Face/image1, ./Face/image2,...}
imglist = [] # {image1,image2,image....}
#distance = {} # {[image:distance],[:],...}
#likelist = [] # alike image



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
            try:
                facenet.load_model(model)
            except:
                print("No such models, add auguments: --model [model name]")
                exit()
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            print("\n\n",args_filepaths,"\n\n")
            for i in range(0, len(args_filepaths), batch_size):
                target_filepaths = args_filepaths[i:i+batch_size]
                print("target_filepaths len:{}".format(len(target_filepaths)))
                images, target_filepaths = load_and_align_data(target_filepaths, image_size, margin, gpu_memory_fraction)
#                print("target_filepaths len:{}".format(len(target_filepaths)))

                # Run forward pass to calculate embeddings
                feed_dict = { images_placeholder: images, phase_train_placeholder:False }
                emb = sess.run(embeddings, feed_dict=feed_dict)
                print("emb len:{}".format(len(emb)))

                for j in range(len(target_filepaths)):
                    extracted_filepaths.append(target_filepaths[j])
                    embs.append(emb[j, :])
#                    print(target_filepaths[j])
    save_embs(embs, extracted_filepaths)  


def save_embs(embs, paths):
    # 特徴量の取得
    pkl_path = "img_facenet.pkl"

    # with open(pkl_path, 'rb') as f:
    #     data = pickle.load(f)
    # f.close()
    f = open(pkl_path, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(f)
    elif sys.version_info.major == 3:
        data = pickle.load(f, encoding='latin-1')
    f.close()
                           
    for i in paths:
        imglist.append(i.split('/')[-1])
    reps = {}
    for i, (emb, path) in enumerate(zip(embs, paths)):
        try:
            basename = os.path.basename(path)
            reps[basename] = emb
        except:
            print('error %1d: %s' % (i, path) )
    # 特徴量の保存
    with open(pkl_path, 'wb') as f:
        reps.update(data)
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
            pnet, rnet, onet = facenets.src.align.detect_face.create_mtcnn(sess, None)


    nrof_samples = len(image_paths)
    img_list = [] #[None] * nrof_samples
    for i in range(nrof_samples):
        print('%1d: %s' % (i, image_paths[i]))
        img_paths_list.append(image_paths[i])
        img = misc.imread(os.path.expanduser(image_paths[i]))
        img_size = np.asarray(img.shape)[0:2]
        try:
            # Try Detect to face And Crop face image!
            bounding_boxes, _ = facenets.src.align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            det = np.squeeze(bounding_boxes[0,0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list.append(prewhitened)
            extracted_filepaths.append(image_paths[i])
        except:
            print("cannot extract_image_align")
            exit()
        
    image = np.stack(img_list)
    return image, extracted_filepaths



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file', default="./Models/20180402-114759.pb")
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

# Measure the distance to Input_image with Collected_features
facenet.detection(imglist[0]) # input argv[1]

