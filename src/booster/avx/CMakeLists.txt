file(GLOB AVX_SRC ./*.cpp)
file(GLOB AVX_HEADERS ./*.h)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a -fopenmp")

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
else()
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -std=c++11 -march=core-avx2 -g -Wall")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -std=c++11 -march=core-avx2 -O3 -Wno-format -Wno-unused-parameter")
endif()

add_library(booster_avx_obj OBJECT ${AVX_SRC} ${AVX_HEADERS})
#add_library(arm_backend STATIC ${AVX_SRC} ${AVX_HEADERS})

#target_include_directories(arm_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
#set(AVX_INSTALL_DIR "${PROJECT_BINARY_DIR}/install/feather_backend_avx/")
#install(TARGETS arm_backend DESTINATION ${AVX_INSTALL_DIR}/lib)
#install(FILES ${AVX_HEADERS} DESTINATION "${AVX_INSTALL_DIR}/include")
