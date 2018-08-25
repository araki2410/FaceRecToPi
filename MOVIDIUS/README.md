Real time face identification on RaspberryPi with MOVIDIUS stick.
====

# Requirements
- RaspberryPi 3 B+ (with rasbian stretch 20180629 model)
- micro SD (up to 32Gb)
- MOVIDIUS stick
<a href="https://developer.movidius.com/"><img src="https://ncsuploads.movidius.com/images/made/images/remote/http_ncsuploads.movidius.com/general/95b5762864eba21d410dbe1ac7c6f3de/Buy_page_image_1000_474_85.jpg"></a>
- python3.5
- Opencv
- tensorflow

# Install
The $ mark is only used to indicate that you typed the command, so you do not hit it in practice.

$マークは、コマンドを入力したことを示すためにのみ使用されるため、実際には打つことはありません。
## Opencv
```shell
$ cd
```
(cd your favorite dir)
```shell
https://qiita.com/mt08/items/e8e8e728cf106ac83218
```
## Tensorflow
```shell
$ cd
```
(cd your favorite dir)
```shell
https://github.com/samjabrahams/tensorflow-on-raspberry-pi
https://qiita.com/ekzemplaro/items/553db4c229632af79607
```
## FaceRecToPi
```shell
$ cd
```
(cd your favorite dir)
```shell
$ git clone https://github.com/araki2410/FaceRecToPi.git
$ cd FaceRecToPi
$ git checkout -b local origin/MOVIDIUS
$ cd MOVIDIUS
$ cp /usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml ./
```


## ncsdk (github)
```shell
$ cd
```
(cd your favorite dir)
```shell
$ git clone https://github.com/movidius/ncsdk.git
```

now(25, Aug, 2018) ncsdk.git is down. so...
read README.md, old vertion is opened. donwload tar.gz file...

## ncappzoo
```shell
$ cd
```
(cd your favorite dir)
```shell
$ git clone https://github.com/movidius/ncappzoo.git
$ cd ncappzoo/tensorflow/facenet
$ sudo make
$ cp facenet_celeb_ncs.graph <MOVIDIUS DIR>
```