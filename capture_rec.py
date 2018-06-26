#!/usr/bin/python3
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import tensorflow as tf
import numpy as np
import sys, os, argparse
#sys.path.append("/home/yanai-lab/araki-t/Git/facenet/src/")
import facenet
import facenets.src.align.detect_face
import pickle, scipy
from scipy import misc
import time

img_paths_list = [] #{./Face/image1, ./Face/image2,...}
imglist = [] # {image1,image2,image....}
#distance = {} # {[image:distance],[:],...}
#likelist = [] # alike image

def main(args):
    ######## Tensor flow
    args_filepaths = args.image_files
    print(args_filepaths)
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
            #

            ###### Capture Camra
            # 定数定義
            ESC_KEY = 27     # Escキー
            INTERVAL= 33     # 待ち時間
            FRAME_RATE = 10  # fps

            ORG_WINDOW_NAME = "CAP"
            # GAUSSIAN_WINDOW_NAME = "gaussian"

            DEVICE_ID = 0
    #        i = 0
            # 分類器の指定
            cascade_file = "./Models/haarcascade_frontalface_alt2.xml"
            cascade = cv2.CascadeClassifier(cascade_file)

            # カメラ映像取得
            cap = cv2.VideoCapture(DEVICE_ID)
            cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap_rate = 0.0625
            size = (int(cap_width * cap_rate), int(cap_height * cap_rate)) # size=(320,240)
            print(size)
            # 初期フレームの読込
            try:
                end_flag, c_frame = cap.read()
                c_frame = cv2.resize(c_frame, size)
                #height, width, channels = c_frame.shape
            except:
                print("device error: Please Try again")
                exit()
            # ウィンドウの準備
            cv2.namedWindow(ORG_WINDOW_NAME)
            # cv2.namedWindow(GAUSSIAN_WINDOW_NAME)
    
            # 変換処理ループ
            while end_flag == True:

            # 画像の取得と顔の検出
                img = c_frame
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face_list = cascade.detectMultiScale(img_gray, minSize=(110, 110))
        
            # 検出した顔に印を付ける
                for (x, y, w, h) in face_list:
                    cv2.imwrite(args_filepaths,img)
                    time.sleep(1)
                    color = (0, 0, 225)
                    pen_w = 3
                    # cv2.rectangle(img_gray, (x, y), (x+w, y+h), color, thickness = pen_w)
                    cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness = pen_w)
                    # break

                    #####################
                    ## Tensorflow
            
                    # Get input and output tensors
                    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                    print("\n\n",args_filepaths,"\n\n")
            
                    for i in range(0, len(args_filepaths), batch_size):
                        target_filepaths = args_filepaths[i:i+batch_size]
                        print(target_filepaths)
                        #print("target_filepaths len:{}".format(len(target_filepaths)))
                        try:
                            images, target_filepaths = load_and_align_data(target_filepaths, image_size, margin, gpu_memory_fraction)
                        except:
                            pass
                        # Run forward pass to calculate embeddings
                        feed_dict = { images_placeholder: images, phase_train_placeholder:False }
                        emb = sess.run(embeddings, feed_dict=feed_dict)
                        print("emb len:{}".format(len(emb)))
                        
                        for j in range(len(target_filepaths)):
                            print(target_filepaths[j])
                            extracted_filepaths.append(target_filepaths[j])
                            embs.append(emb[j, :])
                           # print(target_filepaths[j])
                    save_embs(embs, extracted_filepaths)
#                    print(os.path.basename(target_filepaths[0]))
                    
                    facenet.detection(os.path.basename(target_filepaths[0])) #argv[1] #after your task, not only one shot.

                    #####################
                    ####################
                    
                # フレーム表示
                cv2.imshow(ORG_WINDOW_NAME, c_frame)
                # cv2.imshow(GAUSSIAN_WINDOW_NAME, img_gray)
                # Escキーで終了
                key = cv2.waitKey(INTERVAL)
                if key == ESC_KEY:
                    break
                # 次のフレーム読み込み
                end_flag, c_frame = cap.read()

            # 終了処理
            cv2.destroyAllWindows()
            cap.release()



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
        data.update(reps)
        pickle.dump(data, f)

def load_and_align_data(image_path, image_size, margin, gpu_memory_fraction):
    default_error = ([[[]]], image_path) # not correct 
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


#    nrof_samples = len(image_path)
    img_list = [] #[None] * nrof_samples
#    for i in range(nrof_samples):
#    print(image_path)
    img_paths_list.append(image_path)
    img = misc.imread(os.path.expanduser(image_path))
    img_size = np.asarray(img.shape)[0:2]
    # Try Detect to face And Crop face image!
    bounding_boxes, _ = facenets.src.align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    try:
        det = np.squeeze(bounding_boxes[0,0:4])
    except:
        print("No face")
        return default_error
    
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
    bb[3] = np.minimum(det[3]+margin/2, img_size[0])
    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    try:
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
        extracted_filepaths.append(image_path)
    except:
        print("cannot extract_image_align")
        return default_error
        
    image = np.stack(img_list)
    return image, extracted_filepaths




def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file', default="./Models/20180408-102900.pb")
    parser.add_argument('--image_files', type=str, nargs='+', help='Images to compare', default="./Img/cap.jpg")
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
