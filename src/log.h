#pragma once

#if 0
    #include <android/log.h>
    #define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  "FeatherLib", __VA_ARGS__)
    #define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "FeatherLib", __VA_ARGS__)
    #define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "FeatherLib", __VA_ARGS__)
#else
    #include <stdio.h>
    #define LOGI(...) fprintf(stdout, __VA_ARGS__);fprintf(stdout,"\n");
    #define LOGD(...) fprintf(stdout, __VA_ARGS__);fprintf(stdout,"\n");
    #define LOGE(...) fprintf(stderr, __VA_ARGS__);fprintf(stderr,"\n");
#endif