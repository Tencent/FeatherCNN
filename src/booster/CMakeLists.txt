cmake_minimum_required(VERSION 2.8.10)

file(GLOB LIB_HEADERS ./include/booster/*.h)

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O2 -std=c++11")
else()
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -Wno-format")
endif()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -std=c++11 -Wall")

include_directories("./include/")

if(BOOSTER_AVX)
message(STATUS "Compiling booster AVX version.")
add_subdirectory(./avx)
  if(COMPILE_OPENCL)
    add_subdirectory(./cl)
    add_library(booster STATIC $<TARGET_OBJECTS:booster_avx_obj> $<TARGET_OBJECTS:booster_cl_obj>)
  else()
    add_library(booster STATIC $<TARGET_OBJECTS:booster_avx_obj>)
  endif()
elseif(BOOSTER_ARM)
add_subdirectory(./arm)
  if(COMPILE_OPENCL)
    add_subdirectory(./cl)
    add_library(booster STATIC $<TARGET_OBJECTS:booster_arm_obj> $<TARGET_OBJECTS:booster_cl_obj>)
  else()
    add_library(booster STATIC $<TARGET_OBJECTS:booster_arm_obj>)
  endif()

endif()


set(BOOSTER_INSTALL_DIR "${PROJECT_BINARY_DIR}/install/booster")

message(Library headers: ${LIB_HEADERS})
install(TARGETS booster DESTINATION "${BOOSTER_INSTALL_DIR}/lib")
install(FILES ${LIB_HEADERS} DESTINATION "${BOOSTER_INSTALL_DIR}/include/booster")
