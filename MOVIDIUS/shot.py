#!/usr/bin/python3
# -*- coding:utf-8 -*-

# """https://qiita.com/mix_dvd/items/98feedc8c98bc7790b30"""

import cv2, sys, os
from time import sleep

if __name__ == '__main__':
    try:
        image_path = sys.argv[1]
    except:
        image_path = "Img/"

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
    cascade_file = "haarcascade_frontalface_alt2.xml"
    cascade = cv2.CascadeClassifier(cascade_file)


    
    input_image_filename_list = os.listdir(image_path)
    input_image_filename_list = [image_path + i for i in input_image_filename_list if i.endswith('.jpg')]
    if (len(input_image_filename_list) < 1):
        # no images to show
        print('No .jpg files found')
        exit()

#    print(input_image_filename_list)

    for cap in input_image_filename_list:
        # 画像の取得と顔の検出
        img = cv2.imread(cap)
#        cv2.imshow("cap", img)
#        cv2.waitKey(0)
        image_name=cap.split("/")[-1]
        print(image_name)
        #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            face_list = cascade.detectMultiScale(img, minSize=(90, 90))
            # 検出した顔に印を付ける
            for (x, y, w, h) in face_list:
                croped_image = img[y:y+h, x:x+w]
                croped_image = cv2.resize(croped_image, (160,160))
                gray_img = cv2.cvtColor(croped_image, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(image_path+"Crop/Color/"+image_name, croped_image)
                cv2.imwrite(image_path+"Crop/Gray/"+image_name, gray_img)
                # フレーム表示
                cv2.imshow(image_name, croped_image)
                cv2.waitKey(0)
#                cv2.imshow(GAUSSIAN_WINDOW_NAME, img_gray)
        except:
            print("missed!")
            pass
        # Escキーで終了
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        # 次のフレーム読み込み
#        end_flag, c_frame = cap.read()

    # 終了処理
    cv2.destroyAllWindows()
#    cap.release()


### ""This is shooting the face capture by camera.""
