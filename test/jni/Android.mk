LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := test
LOCAL_SRC_FILES := ../test.cpp
LOCAL_CPPFLAGS := -std=c++11 -Wall -fPIE  -I../build-android/feather/include/feather       # whatever g++ flags you like
LOCAL_LDLIBS := -fopenmp -L$(SYSROOT)/usr/lib -llog -fPIE -pie -L../build-android/feather/armeabi-v7a -lfeather # whatever ld flags you like

include $(BUILD_EXECUTABLE)
