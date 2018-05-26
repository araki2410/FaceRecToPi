# sudo raspi-config
> interfacing option -> camera -> camera enable
> locale ja_JP.UTF-8
> timezone ASIA/Tokyo

# install Opencv2 to Pi(jessie 201705) \
$ sudo apt-get update ; sudo apt-get upgrade \
$ sudo apt-get install build-essential git cmake pkg-config libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libgtk2.0-dev libatlas-base-dev gfortran \

$sudo apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev 
[https://stackoverflow.com/questions/41328451/ssl-module-in-python-is-not-available-when-installing-package-with-pip3]

$ cd 
($ cd [prefer DIR])

$ wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.3.0.zip 
($ wget --no-check-certificate -O opencv.zip...) 
$ unzip opencv.zip 
$ wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.3.0.zip 
$ unzip opencv_contrib.zip \

$ cd opencv-3.3.0 
$ mkdir build 
$ cd build 
$ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_PYTHON_EXAMPLES=ON -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.3.0/modules -D BUILD_EXAMPLES=ON ..

$ make -j4 \
(It takes 2 to 5 hours. Watch https://www.youtube.com/movies) \
$ sudo make install

--------------------
# Install python modules
## scipy
## tensorflow \
$ wget https://github.com/samjabrahams/tensorflow-on-raspberry-pi/releases/download/v1.1.0/tensorflow-1.1.0-cp34-cp34m-linux_armv7l.whl \

"tensorflow-on-raspberry-pi", [https://github.com/samjabrahams/tensorflow-on-raspberry-pi/] \
"Raspberry Pi に python3 用の　tensorflow をインストール", [https://qiita.com/ekzemplaro/items/553db4c229632af79607] \
"pipのパッケージのインストール先について"(pip version?), [https://teratail.com/questions/9769] \