LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := booster
LOCAL_SRC_FILES := ../../../build-android/booster/arm64-v8a/libbooster.a
LOCAL_EXPORT_CPPFLAGS:= -I../../build-android/booster/include/
include $(PREBUILT_STATIC_LIBRARY)
include $(CLEAR_VARS)

LOCAL_MODULE := booster_test
LOCAL_SRC_FILES := ../conv_test.cpp ../utils.cpp
LOCAL_CPPFLAGS := -std=c++11 -Wall -fPIE  -DTEST_SGECONV
#-I../../build-android/booster/include/
LOCAL_LDLIBS := -fopenmp -L$(SYSROOT)/usr/lib -llog -fPIE -pie -latomic
LOCAL_STATIC_LIBRARIES := booster
include $(BUILD_EXECUTABLE)
