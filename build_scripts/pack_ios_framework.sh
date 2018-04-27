#!/bin/bash

NAME=feather

##### package android lib
#ANDROIDPKGNAME=${NAME}-android-lib
#rm -rf $ANDROIDPKGNAME
#mkdir -p $ANDROIDPKGNAME
#mkdir -p $ANDROIDPKGNAME/armeabi-v7a
#mkdir -p $ANDROIDPKGNAME/arm64-v8a
#mkdir -p $ANDROIDPKGNAME/include
#cp build-android-armv7/install/lib/lib${NAME}.a $ANDROIDPKGNAME/armeabi-v7a/
#cp build-android-aarch64/install/lib/lib${NAME}.a $ANDROIDPKGNAME/arm64-v8a/
#cp build-android-aarch64/install/include/* $ANDROIDPKGNAME/include/
#rm -f $ANDROIDPKGNAME.zip
#zip -9 -r $ANDROIDPKGNAME.zip $ANDROIDPKGNAME

##### package ios framework
IOSPKGNAME=./build-ios/${NAME}.framework
rm -rf $IOSPKGNAME
mkdir -p $IOSPKGNAME/Versions/A/Headers
mkdir -p $IOSPKGNAME/Versions/A/Resources
ln -s A $IOSPKGNAME/Versions/Current
ln -s Versions/Current/Headers $IOSPKGNAME/Headers
ln -s Versions/Current/Resources $IOSPKGNAME/Resources
ln -s Versions/Current/${NAME} $IOSPKGNAME/${NAME}
lipo -create \
    build-ios/arm64/install/${NAME}/lib/lib${NAME}.a \
    build-ios/armv7s/install/${NAME}/lib/lib${NAME}.a \
    build-ios/simulator/install/${NAME}/lib/lib${NAME}.a \
    -o $IOSPKGNAME/Versions/A/${NAME}
#build-ios-sim/install/${NAME}/lib/lib${NAME}.a \
cp -r build-ios/arm64/install/${NAME}/include/* $IOSPKGNAME/Versions/A/Headers/

#HEADER_PATH=$IOSPKGNAME/Versions/A/Headers
#HEADERS_TO_EDIT=$HEADER_PATH/feather_simple_generated.h\ $HEADER_PATH/flatbuffers/flatbuffers.h\ $HEADER_PATH/flatbuffers/base.h
#HEADERS_TO_EDIT=$HEADER_PATH/flatbuffers/flatbuffers.h
#HEADERS_TO_EDIT=$HEADER_PATH/flatbuffers/base.h

# Fix the relative path for the framework package.
#for FILE in $HEADERS_TO_EDIT
#do
#	echo $FILE
#	sed -i.bak 's/flatbuffers\//feather\/flatbuffers\//' $FILE
#	echo $FILE.bak
#	rm $FILE.bak
#done

cp ./build_scripts/Info.plist ${IOSPKGNAME}/Versions/A/Resources/
rm -f $IOSPKGNAME.zip
zip -9 -y -r $IOSPKGNAME.zip $IOSPKGNAME
