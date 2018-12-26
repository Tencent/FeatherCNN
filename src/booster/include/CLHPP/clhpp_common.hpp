#ifndef CLHPP_COMMON_HPP
#define CLHPP_COMMON_HPP

#include <cstring>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>

#include "CLHPP/cl2_head.hpp"
#include "log.h"

const std::string g_pre_kernel_dir = "/sdcard/feather_cl/";

/**
 * \brief Convert OpenCL error numbers to their string form.
 * \details Uses the error number definitions from cl.h.
 * \param[in] errorNumber The error number returned from an OpenCL command.
 * \return A name of the error.
 */
const std::string OpenCLErrorToString(cl_int errorNumber);
/**
 * \brief Check an OpenCL error number for errors.
 * \details If errorNumber is not CL_SUCESS, the function will print the string form of the error number.
 * \param[in] errorNumber The error number returned from an OpenCL command.
 * \return False if errorNumber != CL_SUCCESS, true otherwise.
 */
bool checkSuccess(cl_int errorNumber);
bool fileIsExists(std::string fileAddress);
bool dirIsExists(std::string dirAddress);
bool dirCreate(std::string dirAddress);

int buildProgramFromSource(const cl::Context& context,
                           const cl::Device& device,
                           cl::Program& program,
                           const std::string& kernel_code,
                           std::string build_opts);

int buildProgramFromSource(const cl::Context& context,
                           const cl::Device & device,
                           cl::Program& program,
                           const std::string& kernel_code,
                           std::string build_opts,
                           std::string kernelAddr);

int buildProgramFromPrecompiledBinary( const cl::Context& context,
                                       const cl::Device & device,
                                       cl::Program& program,
                                       const std::string& kernel_code,
                                       std::string build_opts,
                                       std::string kernelAddr);

int buildProgram( const cl::Context& context,
                  const cl::Device & device,
                  cl::Program& program,
                  const std::string& kernel_code,
                  std::string build_opts,
                  std::string kernelAddr);

#endif
