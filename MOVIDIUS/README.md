Real time face identification on RaspberryPi with MOVIDIUS stick.
====

# Requirements
- RaspberryPi 3 B+ (with rasbian stretch 20180629 model)
- micro SD (more than 32Gb)
- MOVIDIUS stick
<a href="https://developer.movidius.com/" width="100"><img src="https://ncsuploads.movidius.com/images/made/images/remote/http_ncsuploads.movidius.com/general/95b5762864eba21d410dbe1ac7c6f3de/Buy_page_image_1000_474_85.jpg"></a>
- python3.x
- OpenCV
- Tensorflow
- PyQt5

# Install
The $ mark is only used to indicate that you typed the command, so you do not hit it in practice.

$マークは、コマンドを入力したことを示すためにのみ使用されるため、実際には打つことはありません。
## Opencv
(cd your favorite dir)
```shell
$ cd
```
<a href="https://qiita.com/mt08/items/e8e8e728cf106ac83218">ラズパイ3にOpenCV3を簡単に導入</a>

## Tensorflow
4,Aug,2018 official supported :(<a href="https://github.com/samjabrahams/tensorflow-on-raspberry-pi">github.com/samjabrahams/tensorflow-on-raspberry-pi</a>)
```shell
$ sudo apt install libatlas-base-dev
$ pip3 install tensorflow
```
refarence of how to install tensorflow before official supported.

pipできなかった頃参考してたもの.

- <a href="https://qiita.com/ekzemplaro/items/553db4c229632af79607">Raspberry Pi に python3 用の　tensorflow をインストール</a>
- <a href="https://github.com/samjabrahams/tensorflow-on-raspberry-pi/blob/master/old_readme.md">tensorflow-on-raspberry-pi README</a>

## PyQt5
参考 <a href="https://tomosoft.jp/design/?p=10991">PyQt5をRaspberry Pi「stretch」にインストール</a>
 
pyqt5 default
```shell
$ sudo apt-get install qt5-default
```

Download sip from <a href="https://sourceforge.net/projects/pyqt/files/sip/">PyQt sip</a> . 
(Checked version: sip-4.19.8 / 29,11,2018)
```shell
$ tar -zxvf sip-4.19.8.tar.gz
(or unzip)
$ cd sip-4.19.8
$ python3 configure.py　
$ make
$ sudo make install
$ cd
(goto your working directory)
```

Download PyQt5 from <a href="https://sourceforge.net/projects/pyqt/files/PyQt5/">PyQt PyQt5</a> . 
(Checked version: PyQt-5.10.1 / 29,11,2018)
```shell
$ tar -zxvf PyQt5_gpl-5.10.1.tar.gz
(or unzip)
$ cd PyQt5_gpl-5.10.1.tar.gz
$ python3 configure.py
$ make
$ sudo make install
$ cd
(goto your working directory)
```



## FaceRecToPi
(cd your favorite dir)
```shell
$ cd
```
```shell
$ git clone https://github.com/araki2410/FaceRecToPi.git
$ cd FaceRecToPi
$ git checkout -b local origin/MOVIDIUS
$ cd MOVIDIUS
$ cp /usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml ./
```


## ncsdk (github)
```shell
(cd your favorite dir)
$ cd
```
```shell
$ git clone https://github.com/movidius/ncsdk.git
```

now(25, Aug, 2018) ncsdk.git is down. so...
read README.md, old vertion is opened. donwload tar.gz file...

## ncappzoo
(cd your favorite dir)
```shell
$ cd
```
```shell
$ git clone https://github.com/movidius/ncappzoo.git
$ cd ncappzoo/tensorflow/facenet
$ sudo make
$ cp facenet_celeb_ncs.graph <MOVIDIUS DIR>
```
