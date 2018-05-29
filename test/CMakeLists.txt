cmake_minimum_required(VERSION 3.11)

set(OpenCV_DIR "/home/leejohnnie/tools/OpenCV-android-sdk/sdk/native/jni")

find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
if(NOT OpenCV_FOUND)
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
endif()

include(FindOpenMP)
if(OPENMP_FOUND)
	message(STATUS ${OpenMP_C_FLAGS})
	message(STATUS ${OpenMP_CXX_FLAGS})
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
message(STATUS "OpenMP flags ${CMAKE_CXX_FLAGS}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../build-android/arm64-v8a/install/feather/lib)

add_executable(feather_test test.cpp)
target_link_libraries(feather_test feather ${OpenCV_LIBS})
