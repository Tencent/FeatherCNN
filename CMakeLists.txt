if(CMAKE_TOOLCHAIN_FILE)
set(LIBRARY_OUTPUT_PATH_ROOT ${CMAKE_BINARY_DIR} CACHE PATH "root for library output, set this to change where android libs are compiled to")
# get absolute path, but get_filename_component ABSOLUTE only refer with source dir, so find_file here :(
get_filename_component(CMAKE_TOOLCHAIN_FILE_NAME ${CMAKE_TOOLCHAIN_FILE} NAME)
find_file(CMAKE_TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE_NAME} PATHS ${CMAKE_SOURCE_DIR} NO_DEFAULT_PATH)
message(STATUS "CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
endif()

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

project(feather)

cmake_minimum_required(VERSION 3.10)

#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

option(FEATHER_OPENMP "openmp support" ON)

if(FEATHER_OPENMP)
	if(CMAKE_HOST_APPLE)
		if(IOS)
			message(STATUS "iOS doesn't support OpenMP, use GCD instead.")
			set(OPENMP_FOUND false)
			set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fembed-bitcode")
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fembed-bitcode")
		else()
			set(OpenMP_C_FLAGS -fopenmp=libomp)
			set(OpenMP_CXX_FLAGS -fopenmp=libomp)
			set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
			message(STATUS ${OpenMP_C_FLAGS})
			message(STATUS ${OpenMP_CXX_FLAGS})
		endif()
	else()
		#find_package(OpenMP)
		include(FindOpenMP)
		if(OPENMP_FOUND)
			#if(OpenMP_CXX_FOUND OR OPENMP_FOUND)
			#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
			#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
			message(STATUS ${OpenMP_C_FLAGS})
			message(STATUS ${OpenMP_CXX_FLAGS})
			set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
		endif()
		message(STATUS "OpenMP flags ${CMAKE_CXX_FLAGS}")
	endif()
endif()

#add_definitions(-Wall -Wextra -Wno-unused-function)
add_definitions(-fPIC)
add_definitions(-Ofast)
add_definitions(-ffast-math)
# add_definitions(-march=native)

# add_definitions(-flto)

add_definitions(-fvisibility=hidden -fvisibility-inlines-hidden)

if(ANDROID)
    # disable shared library on android
    set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS FALSE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions")
elseif(IOS)
    # disable shared library on xcode ios
    add_definitions(-isysroot ${IOS_SDK_PATH} -arch ${IOS_ARCH})
    set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS FALSE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions")
endif()

##############################################

# add_subdirectory(examples)
# add_subdirectory(benchmark)
add_subdirectory(src)
#if(NOT ANDROID AND NOT IOS)
#	add_subdirectory(tools)
#endif()
