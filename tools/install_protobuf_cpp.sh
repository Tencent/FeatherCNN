#!/bin/bash

sudo apt-get -y install autoconf automake libtool curl make g++ unzip pkg-config

##Download the latest protobuf
mkdir -p protobuf
pushd protobuf
URL_SUFFIX=$(wget -qO- https://github.com/google/protobuf/releases/latest | grep "/protobuf-cpp*"|cut -d\" -f 2|grep ".tar.gz")
VERSION=$(echo $URL_SUFFIX | cut -d\/ -f6 | cut -dv -f2)
FILE_NAME=$(echo "$URL_SUFFIX"|cut -d\/ -f7)
DIR_NAME=protobuf-$VERSION
echo $URL_SUFFIX
echo $FILE_NAME
echo $DIR_NAME
wget -c https://github.com/$URL_SUFFIX
tar zxvf $FILE_NAME

pushd $DIR_NAME
./autogen.sh
./configure --prefix=/usr/local/
make 
make check
sudo make install
sudo ldconfig # refresh shared library cache.
popd
popd
