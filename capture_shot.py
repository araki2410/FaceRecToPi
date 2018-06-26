#!/usr/bin/python3
# -*- coding:utf-8 -*-

# """https://qiita.com/mix_dvd/items/98feedc8c98bc7790b30"""

import cv2, sys
from time import sleep

if __name__ == '__main__':
    try:
        image_path = sys.argv[1]
    except:
        image_path = "Img/cap.jpg"

    print(image_path)
#    exit()
    # 定数定義
    ESC_KEY = 27     # Escキー
    INTERVAL= 33     # 待ち時間
    FRAME_RATE = 15  # fps

    ORG_WINDOW_NAME = "org"
    GAUSSIAN_WINDOW_NAME = "gaussian"

    DEVICE_ID = 0

    # 分類器の指定
    cascade_file = "./Models/haarcascade_frontalface_alt2.xml"
    cascade = cv2.CascadeClassifier(cascade_file)

    # カメラ映像取得
    cap = cv2.VideoCapture(DEVICE_ID)
    cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap_rate = 0.25
    size = (int(cap_width * cap_rate), int(cap_height * cap_rate)) # size=(320,240)
    print(size)
    # 初期フレームの読込
    end_flag, c_frame = cap.read()
    c_frame = cv2.resize(c_frame, size)
    #height, width, channels = c_frame.shape

    # ウィンドウの準備
    cv2.namedWindow(ORG_WINDOW_NAME)
#    cv2.namedWindow(GAUSSIAN_WINDOW_NAME)

    # 変換処理ループ
    while end_flag == True:

        # 画像の取得と顔の検出
        img = c_frame
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_list = cascade.detectMultiScale(img_gray, minSize=(100, 100))
        
        # 検出した顔に印を付ける
        for (x, y, w, h) in face_list:
            cv2.imwrite(image_path, img)
#            sleep(0.5)
            color = (0, 0, 225)
            pen_w = 3
#            cv2.rectangle(img_gray, (x, y), (x+w, y+h), color, thickness = pen_w)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness = pen_w)

        # フレーム表示
        cv2.imshow(ORG_WINDOW_NAME, c_frame)
#        cv2.imshow(GAUSSIAN_WINDOW_NAME, img_gray)

        # Escキーで終了
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        # 次のフレーム読み込み
        end_flag, c_frame = cap.read()

    # 終了処理
    cv2.destroyAllWindows()
    cap.release()


### ""This is shooting the face capture by camera.""
