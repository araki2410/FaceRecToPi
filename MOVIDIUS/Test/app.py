#! /usr/bin/env python3
from mvnc import mvncapi as mvnc
import sys
import numpy
import cv2

path_to_networks = './'
path_to_images = '../Img/'
graph_filename = 'graph'
#image_filename = path_to_images + 'cat.jpg'
image_filename = path_to_images + 'shot.png'

devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print('No devices found')
    quit()

device = mvnc.Device(devices[0])
device.OpenDevice()

with open(path_to_networks + graph_filename, mode='rb') as f:
    graphfile = f.read()
#    print(type(graphfile))
mean = 128
std = 1/128

categories = []
with open(path_to_networks + 'categories.txt', 'r') as f:
    for line in f:
        cat = line.split('\n')[0]
        if cat != 'classes':
            categories.append(cat)
    f.close()
    print('Number of categories:', len(categories))

with open(path_to_networks + 'inputsize.txt', 'r') as f:
    reqsize = int(f.readline().split('\n')[0])

graph = device.AllocateGraph(graphfile)

cap = cv2.VideoCapture(0)
end_flag, c_frame = cap.read()
while end_flag == True:
    #ret, frame = cap.read()
    #cv2.imshow("Show FLAME Image",c_frame)
    cv2.imwrite(image_filename, c_frame)

    img = cv2.imread(image_filename).astype(numpy.float32)
    dx,dy,dz= img.shape
    delta=float(abs(dy-dx))
    if dx > dy:
        img=img[int(0.5*delta):dx-int(0.5*delta),0:dy]
    else:
        img=img[0:dx,int(0.5*delta):dy-int(0.5*delta)]
    img = cv2.resize(img, (reqsize, reqsize))

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    for i in range(3):
        img[:,:,i] = (img[:,:,i] - mean) * std

    print('Start download to NCS...')
    graph.LoadTensor(img.astype(numpy.float16), 'user object')
    output, userobj = graph.GetResult()
    top_inds = output.argsort()[::-1][:5]

    print(''.join(['*' for i in range(79)]))
    print('inception-v3 on NCS')
    print(''.join(['*' for i in range(79)]))
    for i in range(5):
        print(top_inds[i], categories[top_inds[i]], output[top_inds[i]])
        text = categories[top_inds[0]] + str("{0:.4f}".format(output[top_inds[0]]*100)) + "%"


    print(''.join(['*' for i in range(79)]))
    #graph.DeallocateGraph()
    #device.CloseDevice()
    print('Finished')

    font = cv2.FONT_HERSHEY_DUPLEX
    font_size = 1
    font_thickness =2

    cv2.putText(c_frame,text,(20,40),font,font_size,(0,140,255),font_thickness,cv2.LINE_AA)
    cv2.imshow("Movidius™ NCS & RaspberryPi",c_frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    end_flag, c_frame = cap.read()

cap.release()
cv2.destroyAllWindows()
