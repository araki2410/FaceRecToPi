#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys, os
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLineEdit, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QLabel, QComboBox, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage, QFont 
from PyQt5.QtCore import QTimer, Qt
import time
from time import sleep, gmtime, strftime
from mvnc import mvncapi as mvnc
import numpy 
import pickle


class App:
    def __init__(self, video_source=0):
        app = QApplication(sys.argv)
        self.vid = MyVideoCapture(video_source)
        self.f_net =Facenet()
        self.window = QWidget()
        self.initLayout()

        # auto refresh 15FPS 
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1000./15)
        self.window.update()

        self.window.show()
        
        sys.exit(app.exec_())

    def capture_image(self):
        self.caped_image.setPixmap(QPixmap("./Img/tmp.jpg"))
#        self.window.addWidget(self.caped_image)


        
    def initLayout(self):
        self.window.showFullScreen()
        self.window.setWindowTitle('FaceNet on Raspberrypi with MOVIDIUS')
        #self.window.resize(250, 150)

        self.textbox = QLineEdit()
        # self.textbox.move(70, 50)
        self.textbox.resize(150,50)
        
        # self.caped_image = QLabel(self.window)
        self.caped_image = QLabel()
        self.caped_image.resize(400, 400)
        self.caped_image.setStyleSheet("background-color:#33AA55;")
        self.croped_image = QLabel()
        self.croped_image.resize(200, 200)
        # self.croped_image.setStyleSheet("background-color:#3355AA;")
        # self.caped_image.move(500,50)
        self.colored_label = QLabel()
        # self.colored_label.resize(200, 200)
        self.colored_label.setStyleSheet("background-color:#3355AA;")

        
        font = QFont("メイリオ")
        font.setPointSize(25)
        font.setBold(True)
        self.top_name = QLabel()
        self.top_name.setFont(font)
        self.sec_name = QLabel()
        self.thr_name = QLabel()

        self.top_prop = QLabel()
        self.sec_prop = QLabel()
        self.thr_prop = QLabel()
        green = QLabel("1")
        yegre = QLabel("2")
        yelow = QLabel("3")
        green.setStyleSheet("background-color:#55BB55;")
        yegre.setStyleSheet("background-color:#99BB55;")
        yelow.setStyleSheet("background-color:#AAAA11;")
        sizepolicy_color = green.sizePolicy()
        sizepolicy_label = self.top_name.sizePolicy()
        # 1:9 の比率でlabelを配置
        sizepolicy_color.setHorizontalStretch(1)
        sizepolicy_label.setHorizontalStretch(9)
        self.top_name.setSizePolicy(sizepolicy_label)
        self.sec_name.setSizePolicy(sizepolicy_label)
        self.thr_name.setSizePolicy(sizepolicy_label)
        green.setSizePolicy(sizepolicy_color)
        yegre.setSizePolicy(sizepolicy_color)
        yelow.setSizePolicy(sizepolicy_color)
        

        self.fret=False
        shot_button = QPushButton()
        shot_button.setText('さつえい')
        shot_button.clicked.connect(self.text2filename)
        shot_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        drop_button = QPushButton()
        drop_button.setText('さつえい')
        drop_button.clicked.connect(self.drop2filename)
        drop_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)

        # Layput Boxes Design
        # QVBox QHBox QGrid
        # name_layout = QVBoxLayout()
        # name_layer0 = QVBoxLayout()
        # name_layer0.addWidget(self.top_name)
        # name_layer0.addWidget(self.top_prop)
        # name_layer1 = QVBoxLayout()
        # name_layer1.addWidget(self.sec_name)
        # name_layer1.addWidget(self.sec_prop)
        # name_layer2 = QVBoxLayout()
        # name_layer2.addWidget(self.thr_name)
        # name_layer2.addWidget(self.thr_prop)
        name_layout = QGridLayout()
        name_layout.addWidget(green, 0,0)
        name_layout.addWidget(yegre, 1,0)
        name_layout.addWidget(yelow, 2,0)
        #name_layout.addLayout(name_layer0, 0,1)
        #name_layout.addLayout(name_layer1, 1,1)
        #name_layout.addLayout(name_layer2, 2,1)
        name_layout.addWidget(self.top_name, 0,1)
        name_layout.addWidget(self.sec_name, 1,1)
        name_layout.addWidget(self.thr_name, 2,1)
        name_layout.addWidget(self.top_prop, 0,2)
        name_layout.addWidget(self.sec_prop, 1,2)
        name_layout.addWidget(self.thr_prop, 2,2)

        self.dropdown_init()
        input_layer = QGridLayout()
        input_layer.addWidget(QLabel("なまえ にゅうりょく"), 0,0)
        input_layer.addWidget(self.textbox, 0,1)
        input_layer.addWidget(shot_button, 0,2)
        input_layer.addWidget(QLabel("なまえ せんたく"), 1,0)
        input_layer.addWidget(self.dropdown, 1,1)
        input_layer.addWidget(drop_button, 1,2)
        # input_layer.addLayout(input_slayer, alignment=(Qt.AlignCenter))

        self.output_message = QLabel()
        input_layout = QVBoxLayout()
        input_layout.addLayout(input_layer)
        input_layout.addWidget(self.output_message)
        
        main_layout = QGridLayout()        
        main_layout.addWidget(self.caped_image, 0,0)
        main_layout.addLayout(name_layout, 0,1)
        main_layout.addWidget(self.colored_label, 1,0)
        main_layout.addWidget(self.croped_image, 1,0, alignment=(Qt.AlignCenter))

        main_layout.addLayout(input_layout, 1,1)
        
        self.window.setLayout(main_layout)

    def dropdown_init(self):
        self.dropdown = QComboBox()
        self.dropdown.addItem("----")
        self.drop_list = []
        for i in self.f_net.input_image_filename_list:
            name = i.split("/")[-1].split("_")[0]
            if name not in self.drop_list:
                self.drop_list.append(name)
                self.dropdown.addItem(name)
        self.dropdown.activated[str].connect(self.drop_text)
        self.droptext = ""

    def drop_text(self, text):
        self.droptext = self.dropdown.currentText()
        if self.droptext == "----":
            self.droptext = ""

        
    def text2filename(self):
        self.filename = self.textbox.text()
        # 魔法のことばの入力で別の関数を呼び出す。
        # データの削除関数を呼び出す。
        if self.filename == "せんたくしたデータをさくじょ":
            if self.droptext == "":
                # 選ばれていない場合なにもしない。(めっせーじを空にする)
                self.output_message.setText("")
                return 0
            
            rmdir = self.f_net.IMAGES_DIR
            rmname = self.droptext
            rmdatapath = os.path.join(rmdir, rmname)
            print(rmdatapath)
            os.system("rm "+rmdatapath+"*")
            self.output_message.setText(rmdatapath + "* さくじょ しました")
        else:
            #self.dropdown.addItem(self.filename)
            if len(self.filename) < 10:
                if self.filename not in self.drop_list:
                    self.drop_list.append(self.filename)
                    self.dropdown.addItem(self.filename)
            self.clip(self)
            
    def drop2filename(self):
        self.filename = self.droptext
        self.clip(self)
        
    def clip(self, _):
        if self.filename == "":
            self.output_message.setText("なまえ が ありません。")
            return 0
        elif len(self.filename) > 10:
            self.output_message.setText("なまえ が ながすぎます。(10もじまで)")
            return 0
        if self.fret:
            self.fret = False
            # self.corename = self.filename
            savename =("./Img/" + self.filename + "_" + time.strftime("%Y%m%d_%H%M%S") + ".jpg")
            cv2.imwrite(savename, cv2.cvtColor(self.face_frame, cv2.COLOR_RGB2BGR))        
            self.f_net.new_emb(savename)
            # 背景の色をかえる
            self.colored_label.setStyleSheet("background-color:#33AA55;")
            # 保存した名前を表示する
            message = "save:" + savename
        else:
            message = "あたらしい かお が みつかりません。"
        self.output_message.setText(message)

        
    def update(self):
        ret, frame = self.vid.get_frame()
        fret, top_fname, face_frame = self.f_net.face_rec(ret, frame)

        if ret:
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (360,270))
            img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pix = QPixmap.fromImage(img)
            self.caped_image.setPixmap(pix)
        if fret:
            self.fret = fret
            self.face_frame = face_frame
            self.top_fname = top_fname
            fpix = QPixmap.fromImage(QImage(face_frame, face_frame.shape[1], face_frame.shape[0], QImage.Format_RGB888))
            self.colored_label.setStyleSheet("background-color:#3355AA;")
            self.croped_image.setPixmap(fpix)
            # top = self.top_fname[0][0].split("/")[-1].split("_")[0]
            # self.top_name.setText(top)
            try:
                # self.top_name.setText(self.top_fname[0][0].split("/")[-1].split("_")[0])
                # self.sec_name.setText(self.top_fname[1][0].split("/")[-1].split("_")[0])
                # self.thr_name.setText(self.top_fname[2][0].split("/")[-1].split("_")[0])
                self.top_name.setText(self.top_fname[0][0])
                self.top_prop.setText('{0:.1%}'.format(self.top_fname[0][1]).rjust(7,' '))
                self.sec_name.setText(self.top_fname[1][0])
                self.sec_prop.setText('{0:.1%}'.format(self.top_fname[1][1]).rjust(7,' '))
                self.thr_name.setText(self.top_fname[2][0])
                self.thr_prop.setText('{0:.1%}'.format(self.top_fname[2][1]).rjust(7,' '))
            except:
                pass

            
            
        # frame from cameraを取得して画面に表示する準備,のあと、画面をupdate
        self.window.update
        
class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
               # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


            
class Facenet:
    def save_emb(self, emb, path):
        # 特徴量の取得
        reps = {}
        # print(numpy.sqrt(numpy.sum(emb*emb)))
        try:
            basename = os.path.basename(path)
            reps[basename] = emb
        except:
            print('error %1d: %s' % (i, path) )

        try:
            f = open(self.PKL_PATH, 'rb')
            if sys.version_info.major == 2:
                data = pickle.load(f)
            elif sys.version_info.major == 3:
                data = pickle.load(f) #, encoding='latin-1')
            f.close()
        except:
            data = {}
            g = open(self.PKL_PATH, 'wb')
            pickle.dump(data, g)
            g.close

        # 特徴量の保存
        with open(self.PKL_PATH, 'wb') as f:
            reps.update(data)
            pickle.dump(reps, f)

    # Run an inference on the passed image
    # image_to_classify is the image on which an inference will be performed
    #    upon successful return this image will be overlayed with boxes
    #    and labels identifying the found objects within the image.
    # ssd_mobilenet_graph is the Graph object from the NCAPI which will
    #    be used to peform the inference.
    def run_inference(self, image_to_classify, facenet_graph):

        # get a resized version of the image that is the dimensions
        # SSD Mobile net expects
        resized_image = self.preprocess_image(image_to_classify)


        # ***************************************************************
        # Send the image to the NCS
        # ***************************************************************
        facenet_graph.LoadTensor(resized_image.astype(numpy.float16), None)

        # ***************************************************************
        # Get the result from the NCS
        # ***************************************************************

        output, userobj = facenet_graph.GetResult()
        return output


    # overlays the boxes and labels onto the display image.
    # display_image is the image on which to overlay to
    # image info is a text string to overlay onto the image.
    # matching is a Boolean specifying if the image was a match.
    # returns None
    def overlay_on_image(self, display_image, image_info, matching):
        rect_width = 10
        offset = int(rect_width/2)
        if (image_info != None):
            cv2.putText(display_image, image_info, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            if (matching):
                # match, green rectangle
                cv2.rectangle(display_image, (0+offset, 0+offset),
                              (display_image.shape[1]-offset-1, display_image.shape[0]-offset-1),
                              (0, 255, 0), 10)
            else:
                # not a match, red rectangle
                cv2.rectangle(display_image, (0+offset, 0+offset),
                              (display_image.shape[1]-offset-1, display_image.shape[0]-offset-1),
                              (0, 0, 255), 10)


    # whiten an image
    def whiten_image(self, source_image):
        source_mean = numpy.mean(source_image)
        source_standard_deviation = numpy.std(source_image)
        std_adjusted = numpy.maximum(source_standard_deviation, 1.0 / numpy.sqrt(source_image.size))
        whitened_image = numpy.multiply(numpy.subtract(source_image, source_mean), 1 / std_adjusted)
        return whitened_image

    # create a preprocessed image from the source image that matches the
    # network expectations and return it
    def preprocess_image(self, src):
        # scale the image
        NETWORK_WIDTH = 160
        NETWORK_HEIGHT = 160
        preprocessed_image = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

        #convert to RGB
        preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
        #whiten
        preprocessed_image = self.whiten_image(preprocessed_image)

        # return the preprocessed image
        return preprocessed_image

    # determine if two images are of matching faces based on the
    # the network output for both images.
    def face_match(face1_output, face2_output):
        if (len(face1_output) != len(face2_output)):
            print('length mismatch in face_match')
            return False
        total_diff = 0
        for output_index in range(0, len(face1_output)):
            this_diff = numpy.square(face1_output[output_index] - face2_output[output_index])
            total_diff += this_diff

        if (total_diff < FACE_MATCH_THRESHOLD):
            # the total difference between the two is under the threshold so
            # the faces match.
            return True

        # differences between faces was over the threshold above so
        # they didn't match.
        return False

    # determine if two images are of matching faces based on the
    # the network output for both images.
    ### Return distance
    def face_match_dist(self, face1_output, face2_output):
        if (len(face1_output) != len(face2_output)):
            print('length mismatch in face_match')
            # return False
            return 100
        total_diff = 0
        
        # for output_index in range(0, len(face1_output)):
        #     this_diff = numpy.square(face1_output[output_index] - face2_output[output_index])
        #     total_diff += this_diff
        # print(total_diff)
        total_diff = numpy.sum((face1_output.astype(numpy.float64) - face2_output.astype(numpy.float64))**2)
        # print(total_diff)
        return total_diff 



    # handles key presses
    # raw_key is the return value from cv2.waitkey
    # returns False if program should end, or True if should continue
    def handle_keys(raw_key):
        ascii_code = raw_key & 0xFF
        if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
            return False
        
        return True

    def new_emb(self, input_image_file):
        # run a single inference on the image and overwrite the
        # boxes and labels
        input_image_file_name = input_image_file.split('/')[-1]
        infer_image = cv2.imread(input_image_file)
        graph_output = self.run_inference(infer_image, self.graph)
        #print("save!" + input_image_file_name)
        self.save_emb(graph_output, input_image_file)
        self.input_image_filename_list.append(input_image_file)
        return graph_output
        
    # def run_images(self, valid_output, validated_image_filename, graph, input_image_filename_list):
    def run_images(self, valid_output, graph, input_image_filename_list):
#        self.distance_list = {}
        self.name2dist = {}
        for input_image_file in input_image_filename_list :        
            try:
                f = open(self.PKL_PATH, 'rb')
                data = pickle.load(f)
            except:
                data = {}
                g = open(self.PKL_PATH, 'wb')
                pickle.dump(data, g)
                g.close

            input_image_file_name = input_image_file.split('/')[-1]
            if input_image_file_name in list(data.keys()):
                test_output = data[input_image_file_name]
                corename = input_image_file_name.split("/")[-1].split("_")[0]
                dist = self.face_match_dist(valid_output, test_output)
                if corename in self.name2dist.keys():
                    self.name2dist[corename].append(int(dist < 0.8))
                else:
                    self.name2dist[corename] = [int(dist < 0.8)]
            else:
                # read one of the images to run an inference on from the disk
                test_output = self.new_emb(input_image_file)



            
#            self.distance_list[input_image_file] = self.face_match_dist(valid_output, test_output)
        sig = [0]
        for i in self.name2dist.values():
            sig.append(sum(i)/float(len(i)))
        sig = sum(sig)
        if sig == 0:
            return [("がいとうなし", 0.0),("がいとうなし", 0.0),("がいとうなし", 0.0)]
        for k,v in self.name2dist.items():
            ave = sum(v)/float(len(v))
            self.name2dist[k] = ave/sig
            #self.name2dist[k] = sum(v)/float(len(v))


            
        self.name2dist = (sorted(self.name2dist.items(), key=lambda x:x[1], reverse=True))
            
#        self.distance_list = sorted(self.distance_list.items(), key=lambda x:x[1])  # transed to list object
       
        # top_image_name = self.distance_list[0][0]
        # top_name = top_image_name.split("/")[-1].split("_")[0]
        # if len(self.distance_list) > 2:
        #     sec_image_name = self.distance_list[1][0]
        #     thr_image_name = self.distance_list[2][0]
        #     top3_image_name = top_image_name +", "+ sec_image_name +", "+ thr_image_name
        #     sec_image = cv2.imread(sec_image_name)
        #     thr_image = cv2.imread(thr_image_name)

        return(self.name2dist)

        # try:
        #     # return(top_name)
        #     return(self.distance_list[0:3])
        # except:
        #     return("No data")
            ###
            ### input data to register here
            ###
        
    def face_rec(self, ret, c_frame):
        #size = (320,240)
        res = ""
        #img_gray = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)
        #face_list = self.cascade.detectMultiScale(img_gray, minSize=(90,90))
        face_list = self.cascade.detectMultiScale(c_frame, minSize=(90,90))
        for (x,y,w,h) in face_list:
            # 顔検出した
            croped_image = c_frame[y:y+h, x:x+w]
            croped_image = cv2.resize(croped_image, (160,160))
            valid_frame_output = self.run_inference(croped_image, self.graph)
            colored_image = cv2.cvtColor(croped_image, cv2.COLOR_BGR2RGB)
            res = self.run_images(valid_frame_output, self.graph, self.input_image_filename_list)
            return True, res, croped_image
        
        return False, res, c_frame

    # Clean up the graph and the device
#        graph.DeallocateGraph()
#        device.CloseDevice()

    def __init__(self):

        self.EXAMPLES_BASE_DIR='../../'
        self.IMAGES_DIR = './Img/' #Crop/Color/'
        
        GRAPH_FILENAME = "facenet_celeb_ncs.graph"

        # name of the opencv window
        self.CV_WINDOW_NAME = "ArakiNet on Raspi with MOVIDIUS"

        CAMERA_INDEX = 0
        REQUEST_CAMERA_WIDTH = 160
        REQUEST_CAMERA_HEIGHT = 120

        # the same face will return 0.0
        # different faces return higher numbers
        # this is NOT between 0.0 and 1.0
        #FACE_MATCH_THRESHOLD = 1.2
        FACE_MATCH_THRESHOLD = 0.8

        ## added when camera caputure code
        ##
        ORG_WINDOW_NAME = "CAP"
        INTERVAL = 33
        ESC_KEY = 27     # Escキー
        # 分類器の指定
        CASCADE_FILE = "./haarcascade_frontalface_alt2.xml"
        self.PKL_PATH = "img_facenet_MOVIDIUS.pkl"
        

        # Get a list of ALL the sticks that are plugged in
        # we need at least one
        devices = mvnc.EnumerateDevices()
        if len(devices) == 0:
            print('No NCS devices found')
            quit()

        # Pick the first stick to run the network
        self.device = mvnc.Device(devices[0])

        # Open the NCS
        self.device.OpenDevice()

        # The graph file that was created with the ncsdk compiler
        graph_file_name = GRAPH_FILENAME

        # read in the graph file to memory buffer
        with open(graph_file_name, mode='rb') as f:
            graph_in_memory = f.read()

        # create the NCAPI graph instance from the memory buffer containing the graph file.
        self.graph = self.device.AllocateGraph(graph_in_memory)

        # get list of all the .jpg files in the image directory
        input_image_filename_list = os.listdir(self.IMAGES_DIR)
        self.input_image_filename_list = [self.IMAGES_DIR + i for i in input_image_filename_list if i.endswith('.jpg')]
        if (len(input_image_filename_list) < 1):
            ### no images to show
            # print('No .jpg files found')
            # return 1
            os.system("cp ./あらき_20181101_181928.jpg ./Img")
            self.input_image_filename_list.append("./Img/あらき_20181101_181928.jpg")



        self.cascade = cv2.CascadeClassifier(CASCADE_FILE)

        #cv2.namedWindow(ORG_WINDOW_NAME)
        




App()

