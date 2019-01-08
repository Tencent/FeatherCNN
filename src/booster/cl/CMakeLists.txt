file(GLOB CL_SRC ./*.cpp)
file(GLOB CL_HEADERS ./*.h)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -Wall -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -Wno-format -std=c++11 -Wno-unused-parameter -Wno-deprecated-declarations")

if(COMPILE_OPENCL)
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DFEATHER_OPENCL")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DFEATHER_OPENCL")
	#add_library(booster_cl_obj OBJECT ${LAYER_CL_SRC} ${LAYER_CL_HEADERS})
endif()

add_library(booster_cl_obj OBJECT ${CL_SRC} ${CL_HEADERS})
#add_library(arm_backend STATIC ${AVX_SRC} ${AVX_HEADERS})

#target_include_directories(arm_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
#set(AVX_INSTALL_DIR "${PROJECT_BINARY_DIR}/install/feather_backend_avx/")
#install(TARGETS arm_backend DESTINATION ${AVX_INSTALL_DIR}/lib)
#install(FILES ${AVX_HEADERS} DESTINATION "${AVX_INSTALL_DIR}/include")
