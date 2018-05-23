file(GLOB ARM_SRC ./*.cpp)
file(GLOB ARM_HEADERS ./*.h)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS}  -O3  -Wno-format -Wno-unused-parameter")

#message(STATUS "General backend flag ${CMAKE_CXX_FLAGS}")

add_library(general_backend_obj OBJECT ${ARM_SRC} ${ARM_HEADERS})