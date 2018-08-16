#!/usr/bin/python3
# -*- coding:utf-8 -*-

import cv2
import tkinter as tk
import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
from mvnc import mvncapi as mvnc
import numpy
import cv2
import sys
import os
import pickle
from time import sleep, gmtime, strftime

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.window.attributes("-zoomed",True)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = 320, height = 240)
                                     #self.vid.width, height = self.vid.height)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
    def cap_frame(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            #return(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            return(frame)

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

        self.window.after(self.delay, self.update)


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
    def save_emb(emb, path):
        # 特徴量の取得
        reps = {}                       

        try:
            basename = os.path.basename(path)
            reps[basename] = emb
        except:
            print('error %1d: %s' % (i, path) )

        # with open(PKL_PATH, 'rb') as f:
        #     data = pickle.load(f)
        # f.close()
        try:
            f = open(PKL_PATH, 'rb')
            if sys.version_info.major == 2:
                data = pickle.load(f)
            elif sys.version_info.major == 3:
                data = pickle.load(f) #, encoding='latin-1')
            f.close()
        except:
            data = {}
            g = open(PKL_PATH, 'wb')
            pickle.dump(data, g)
            g.close

        # 特徴量の保存
        with open(PKL_PATH, 'wb') as f:
            reps.update(data)
            pickle.dump(reps, f)

    # Run an inference on the passed image
    # image_to_classify is the image on which an inference will be performed
    #    upon successful return this image will be overlayed with boxes
    #    and labels identifying the found objects within the image.
    # ssd_mobilenet_graph is the Graph object from the NCAPI which will
    #    be used to peform the inference.
    def run_inference(image_to_classify, facenet_graph):

        # get a resized version of the image that is the dimensions
        # SSD Mobile net expects
        resized_image = preprocess_image(image_to_classify)

        #cv2.imshow("preprocessed", resized_image)

        # ***************************************************************
        # Send the image to the NCS
        # ***************************************************************
        facenet_graph.LoadTensor(resized_image.astype(numpy.float16), None)

        # ***************************************************************
        # Get the result from the NCS
        # ***************************************************************
        output, userobj = facenet_graph.GetResult()

        #print("Total results: " + str(len(output)))
        #print(output)
        return output


    # overlays the boxes and labels onto the display image.
    # display_image is the image on which to overlay to
    # image info is a text string to overlay onto the image.
    # matching is a Boolean specifying if the image was a match.
    # returns None
    def overlay_on_image(display_image, image_info, matching):
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
    def whiten_image(source_image):
        source_mean = numpy.mean(source_image)
        source_standard_deviation = numpy.std(source_image)
        std_adjusted = numpy.maximum(source_standard_deviation, 1.0 / numpy.sqrt(source_image.size))
        whitened_image = numpy.multiply(numpy.subtract(source_image, source_mean), 1 / std_adjusted)
        return whitened_image

    # create a preprocessed image from the source image that matches the
    # network expectations and return it
    def preprocess_image(src):
        # scale the image
        NETWORK_WIDTH = 160
        NETWORK_HEIGHT = 160
        preprocessed_image = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

        #convert to RGB
        preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)

        #whiten
        preprocessed_image = whiten_image(preprocessed_image)

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
        print('Total Difference is: ' + str(total_diff))

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
    def face_match_dist(face1_output, face2_output):
        if (len(face1_output) != len(face2_output)):
            print('length mismatch in face_match')
            # return False
            return 100
        total_diff = 0
        
        for output_index in range(0, len(face1_output)):
            this_diff = numpy.square(face1_output[output_index] - face2_output[output_index])
            total_diff += this_diff
        #    print('Total Difference is: ' + str(total_diff))
        return total_diff


    # handles key presses
    # raw_key is the return value from cv2.waitkey
    # returns False if program should end, or True if should continue
    def handle_keys(raw_key):
        ascii_code = raw_key & 0xFF
        if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
            return False
        
        return True


    def run_images(valid_output, validated_image_filename, graph, input_image_filename_list):
        #cv2.namedWindow(CV_WINDOW_NAME)
        distance_list = {}
        for input_image_file in input_image_filename_list :
        
            try:
                f = open(PKL_PATH, 'rb')
                data = pickle.load(f)
            except:
                data = {}
                g = open(PKL_PATH, 'wb')
                pickle.dump(data, g)
                g.close


            input_image_file_name = input_image_file.split('/')[-1]
            if input_image_file_name in list(data.keys()):
                test_output = data[input_image_file_name]
            else:
                # read one of the images to run an inference on from the disk
                infer_image = cv2.imread(input_image_file)
                # run a single inference on the image and overwrite the
                # boxes and labels
                test_output = run_inference(infer_image, graph)
                print("save!" + input_image_file_name)
                save_emb(test_output, input_image_file)


                
            distance_list[input_image_file] = face_match_dist(valid_output, test_output)

        distance_list = sorted(distance_list.items(), key=lambda x:x[1])  # transed to list object
        #    print(distance_list)
        top_image_name = distance_list[0][0]
        top_image = cv2.imread(top_image_name)
        cv2.imshow("1st", top_image)
        top_name = top_image_name.split("/")[-1].split("_")[0]
        try:
            print(register[top_name])
        except:
            print("who is he? I have image but have not data of " + top_name)
            ###
            ### input data to register here
            ###
        
        if len(distance_list) > 2:
            sec_image_name = distance_list[1][0]
            thr_image_name = distance_list[2][0]
            top3_image_name = top_image_name +", "+ sec_image_name +", "+ thr_image_name
            sec_image = cv2.imread(sec_image_name)
            thr_image = cv2.imread(thr_image_name)
            cv2.imshow("2nd", sec_image)
            cv2.imshow("3rd", thr_image)
#        print(top3_image_name)
#    cv2.waitKey(0)

        # cv2.putText(infer_image, match_text + " - Hit key for next.", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)


    def __init__(self):
        self.register = {}   # {"juntoku eri": "準得 映理"}
        with open("register", "r") as l:
            #    i = [j.strip() for j in l.readlines()]
            i = l.readlines()
            for j in i:
                n,m = j.split(",")
                self.register[n] = m.strip()
                #    print(register)
        l.close

        self.EXAMPLES_BASE_DIR='../../'
        #IMAGES_DIR = './Img/Crop/Gray/'
        self.IMAGES_DIR = './Img/Crop/Color/'
        
        # VALIDATED_IMAGES_DIR = IMAGES_DIR + 'validated_images/'
        self.VALIDATED_IMAGES_DIR = './Img/validated_images/'
        #validated_image_filename = VALIDATED_IMAGES_DIR + 'valid.jpg'
        self.validated_image_filename = VALIDATED_IMAGES_DIR + 'araki_20180627.jpg'

    
        GRAPH_FILENAME = "facenet_celeb_ncs.graph"

        # name of the opencv window
        self.CV_WINDOW_NAME = "FaceNet"

        CAMERA_INDEX = 0
        REQUEST_CAMERA_WIDTH = 640
        REQUEST_CAMERA_HEIGHT = 480

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
        self.CASCADE_FILE = "./haarcascade_frontalface_alt2.xml"
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
        self.graph_file_name = GRAPH_FILENAME

        # read in the graph file to memory buffer
        with open(graph_file_name, mode='rb') as f:
            graph_in_memory = f.read()

        # create the NCAPI graph instance from the memory buffer containing the graph file.
        self.graph = device.AllocateGraph(graph_in_memory)

        #    validated_image = cv2.imread(validated_image_filename)
        #    valid_output = run_inference(validated_image, graph)

        # get list of all the .jpg files in the image directory
        input_image_filename_list = os.listdir(IMAGES_DIR)
        self.input_image_filename_list = [IMAGES_DIR + i for i in input_image_filename_list if i.endswith('.jpg')]
        if (len(input_image_filename_list) < 1):
            # no images to show
            print('No .jpg files found')
            return 1



        self.cascade = cv2.CascadeClassifier(CASCADE_FILE)

        cv2.namedWindow(ORG_WINDOW_NAME)


    def face_rec(c_frame):
        size = (320,240)
        c_frame = cv2.resize(c_frame, size)
        img_gray = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)
        valid_frame_output = run_inference(c_frame, graph)
        face_list = self.cascade.detectMultiScale(img_gray, minSize=(90,90))
        for (x,y,w,h) in face_list:
            # 顔検出した
            #            croped_image = img_gray[y:y+h, x:x+w]
            croped_image = c_frame[y:y+h, x:x+w]
            croped_image = cv2.resize(croped_image, (160,160))
            # croped_image = preprocess_image(croped_image)
            cv2.imshow("crop!", croped_image)
            run_images(valid_frame_output, croped_image, self.graph, self.input_image_filename_list)
            #run_images(valid_frame_output, c_frame, self.graph, self.input_image_filename_list)


        # Clean up the graph and the device
#        graph.DeallocateGraph()
#        device.CloseDevice()


    # main entry point for program. we'll call main() to do what needs to be done.
# if __name__ == "__main__":
#     sys.exit(main())


# Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")
