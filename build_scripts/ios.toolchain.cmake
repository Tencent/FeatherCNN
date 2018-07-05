# For cross-compiling on arm64 Linux using gcc-aarch64-linux-gnu package:
# - install AArch64 tool chain:
#   $ sudo apt-get install g++-aarch64-linux-gnu
# - cross-compiling config
#   $ cmake -DCMAKE_TOOLCHAIN_FILE=../dynamorio/make/toolchain-arm64.cmake ../dynamorio
# You may have to set CMAKE_FIND_ROOT_PATH to point to the target enviroment, e.g.
# by passing -DCMAKE_FIND_ROOT_PATH=/usr/aarch64-linux-gnu on Debian-like systems.
set(CMAKE_SYSTEM_NAME Darwin)
set(CMAKE_SYSTEM_VERSION 1)
set(UNIX True)
set(APPLE True)
set(IOS True)

# specify the cross compiler as clang.
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

# To build the tests, we need to set where the target environment containing
# the required library is. 
set(CMAKE_FIND_ROOT_PATH ${IOS_SDK_PATH})
# search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)


# Set additional variables.
# If we don't set some of these, CMake will end up using the host version.
# We want the full path, however, so we can pass EXISTS and other checks in
# the our CMake code.
find_program(CC_FULL_PATH clang)
if (NOT CC_FULL_PATH)
  message(FATAL_ERROR "Cross-compiler clang not found")
endif ()
get_filename_component(CC_DIR ${CC_FULL_PATH} PATH)
message(STATUS "CC path is ${CC_FULL_PATH}")
#set(IOS_ARCH arm64)

#SET(CMAKE_LINKER       ${CC_DIR}/aarch64-${TARGET_ABI}-ld      CACHE FILEPATH "linker")
#SET(CMAKE_ASM_COMPILER ${CC_DIR}/aarch64-${TARGET_ABI}-as      CACHE FILEPATH "assembler")
#SET(CMAKE_OBJCOPY      ${CC_DIR}/aarch64-${TARGET_ABI}-objcopy CACHE FILEPATH "objcopy")
#SET(CMAKE_STRIP        ${CC_DIR}/aarch64-${TARGET_ABI}-strip   CACHE FILEPATH "strip")
#SET(CMAKE_CPP          ${CC_DIR}/aarch64-${TARGET_ABI}-cpp     CACHE FILEPATH "cpp")

set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE 1)
# Without this, Xcode adds -fembed-bitcode-marker compile options instead of -fembed-bitcode set(CMAKE_C_FLAGS "-fembed-bitcode ${CMAKE_C_FLAGS}")
set(CMAKE_XCODE_ATTRIBUTE_BITCODE_GENERATION_MODE "bitcode") 
set(BITCODE_FLAGS "-fembed-bitcode")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${BITCODE_FLAGS}"  CACHE INTERNAL "ios c compiler flags" FORCE) 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BITCODE_FLAGS}" CACHE INTERNAL "ios c compiler flags" FORCE)



