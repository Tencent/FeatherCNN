LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := test
LOCAL_SRC_FILES := ../test_bin.cpp 
LOCAL_CPPFLAGS := -std=c++11 -Wall -fPIE  -I../build-android/feather/include/feather
LOCAL_LDLIBS := -fopenmp -L$(SYSROOT)/usr/lib -llog -fPIE -pie -latomic
#LOCAL_LDLIBS += -L../build-android/feather/armeabi-v7a -lfeather
LOCAL_LDLIBS += -L../build-android/feather/arm64-v8a -lfeather

include $(BUILD_EXECUTABLE)
