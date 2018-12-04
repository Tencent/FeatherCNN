LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := feather
LOCAL_SRC_FILES := ../../../build-android/feather/arm64-v8a/libfeather.a
LOCAL_EXPORT_CPPFLAGS:= -I../../build-android/feather/include/feather/ -DFEATHER_OPENCL
include $(PREBUILT_STATIC_LIBRARY)
include $(CLEAR_VARS)

LOCAL_MODULE := feather_test
LOCAL_SRC_FILES := ../test_bin.cpp
LOCAL_CPPFLAGS := -std=c++11 -Wall -fPIE -fexceptions
LOCAL_LDLIBS := -fopenmp -llog -fPIE -pie -latomic -L$(SYSROOT)/usr/lib
LOCAL_STATIC_LIBRARIES := feather
include $(BUILD_EXECUTABLE)
