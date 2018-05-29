LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

OPENCV_INC_DIR := /home/leejohnnie/tools/OpenCV-android-sdk/sdk/native/jni/include
FEATHER_INC_DIR := ../build-android/arm64-v8a/install/feather/include
FEATHER_LIB_DIR := ../build-android/arm64-v8a/install/feather/lib

OPENCV_LIB_DIR := /home/leejohnnie/tools/OpenCV-android-sdk/sdk/native/staticlibs/arm64-v8a
THIRDPART_LIB_DIR := /home/leejohnnie/tools/OpenCV-android-sdk/sdk/native/3rdparty/libs/arm64-v8a

LOCAL_MODULE := test
LOCAL_SRC_FILES := ../test.cpp
LOCAL_CPPFLAGS := -fexceptions -frtti -std=c++11 -Wall -fPIE  -I$(FEATHER_INC_DIR) -I$(OPENCV_INC_DIR)
LOCAL_LDLIBS := -fopenmp -L$(SYSROOT)/usr/lib -llog -fPIE -pie -L$(FEATHER_LIB_DIR) -lfeather -L$(OPENCV_LIB_DIR) -lopencv_core -lopencv_highgui -lopencv_imgproc \
/home/leejohnnie/tools/OpenCV-android-sdk/sdk/native/staticlibs/arm64-v8a/libopencv_core.a /home/leejohnnie/tools/OpenCV-android-sdk/sdk/native/staticlibs/arm64-v8a/libopencv_highgui.a /home/leejohnnie/tools/OpenCV-android-sdk/sdk/native/staticlibs/arm64-v8a/libopencv_imgproc.a /home/leejohnnie/tools/OpenCV-android-sdk/sdk/native/staticlibs/arm64-v8a/libopencv_imgcodecs.a /home/leejohnnie/tools/OpenCV-android-sdk/sdk/native/staticlibs/arm64-v8a/libopencv_videoio.a /home/leejohnnie/tools/OpenCV-android-sdk/sdk/native/staticlibs/arm64-v8a/libopencv_imgcodecs.a /home/leejohnnie/tools/OpenCV-android-sdk/sdk/native/staticlibs/arm64-v8a/libopencv_imgproc.a /home/leejohnnie/tools/OpenCV-android-sdk/sdk/native/staticlibs/arm64-v8a/libopencv_core.a /home/leejohnnie/tools/OpenCV-android-sdk/sdk/native/3rdparty/libs/arm64-v8a/liblibjpeg.a /home/leejohnnie/tools/OpenCV-android-sdk/sdk/native/3rdparty/libs/arm64-v8a/liblibwebp.a /home/leejohnnie/tools/OpenCV-android-sdk/sdk/native/3rdparty/libs/arm64-v8a/libcpufeatures.a /home/leejohnnie/tools/OpenCV-android-sdk/sdk/native/3rdparty/libs/arm64-v8a/liblibpng.a /home/leejohnnie/tools/OpenCV-android-sdk/sdk/native/3rdparty/libs/arm64-v8a/liblibtiff.a /home/leejohnnie/tools/OpenCV-android-sdk/sdk/native/3rdparty/libs/arm64-v8a/liblibjasper.a /home/leejohnnie/tools/OpenCV-android-sdk/sdk/native/3rdparty/libs/arm64-v8a/libIlmImf.a -lz -llog /home/leejohnnie/tools/OpenCV-android-sdk/sdk/native/3rdparty/libs/arm64-v8a/libtbb.a

include $(BUILD_EXECUTABLE)
