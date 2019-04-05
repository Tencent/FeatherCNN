file(GLOB ARM_SRC ./*.cpp)
file(GLOB ARM_HEADERS ../include/*.h)
list(REMOVE_ITEM ARM_SRC "${CMAKE_CURRENT_SOURCE_DIR}/./sgemm_legacy.cpp")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -Wno-format")

add_library(booster_arm_obj OBJECT ${ARM_SRC} ${ARM_HEADERS})
#add_library(arm_backend STATIC ${ARM_SRC} ${ARM_HEADERS})

#target_include_directories(arm_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
#set(ARM_INSTALL_DIR "${PROJECT_BINARY_DIR}/install/booster_arm/")
#install(TARGETS arm_backend DESTINATION ${ARM_INSTALL_DIR}/lib)
#install(FILES ${ARM_HEADERS} DESTINATION "${ARM_INSTALL_DIR}/include")
