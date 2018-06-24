#!/usr/bin/python3
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from time import time
print("tuture", time())


import cv2
import tensorflow as tf
import numpy as np
import sys, os, argparse
#sys.path.append("/home/yanai-lab/araki-t/Git/facenet/src/")
import facenet
import facenets.src.align.detect_face
import pickle, scipy
from scipy import misc
from time import sleep

print("imported", time())


with tf.Graph().as_default():
    
    with tf.Session() as sess:
        
        # Load the model
        facenet.load_model("./Models/20180408-102900.pb")
print("model loaded", time())
