#ifndef CL_COMMON_H
#define CL_COMMON_H

#include <cstring>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include "CL/cl.h"
#include "log.h"
/**
 * \brief Create an OpenCL command queue for a given context.
 * \param[in] context The OpenCL context to use.
 * \param[out] commandQueue The created OpenCL command queue.
 * \param[out] device The device in which the command queue is created.
 * \return False if an error occurred, otherwise true.
 */
bool createCommandQueue(cl_context context, cl_command_queue* commandQueue, cl_device_id* device);
/**
 * \brief Convert OpenCL error numbers to their string form.
 * \details Uses the error number definitions from cl.h.
 * \param[in] errorNumber The error number returned from an OpenCL command.
 * \return A name of the error.
 */
std::string errorNumberToString(cl_int errorNumber);
/**
 * \brief Check an OpenCL error number for errors.
 * \details If errorNumber is not CL_SUCESS, the function will print the string form of the error number.
 * \param[in] errorNumber The error number returned from an OpenCL command.
 * \return False if errorNumber != CL_SUCCESS, true otherwise.
 */
bool checkSuccess(cl_int errorNumber);

int buildProgramFromSource(const cl_context& context, const cl_device_id& device, cl_program& program,
                            const std::string& kernel_code, std::string build_opts);
#endif
