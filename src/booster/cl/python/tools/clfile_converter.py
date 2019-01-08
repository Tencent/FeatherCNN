import os
import sys

import jinja2

# python encrypt_opencl_codegen.py --cl_kernel_dir=./mace/ops/opencl/cl/  \
#     --output_path=./mace/codegen/opencl_encrypt/opencl_encrypted_program.cc

def encrypt_code(code_str):
    encrypted_arr = []
    for i in range(len(code_str)):
        encrypted_char = hex(ord(code_str[i]))
        encrypted_arr.append(encrypted_char)
    return encrypted_arr

def encrypt_opencl_codegen(cl_kernel_dir, output_path):
    if not os.path.exists(cl_kernel_dir):
        print("Input cl_kernel_dir " + cl_kernel_dir + " doesn't exist!")

    header_code = ""
    for file_name in os.listdir(cl_kernel_dir):
        file_path = os.path.join(cl_kernel_dir, file_name)
        if file_path[-2:] == ".h":
            with open(file_path, "r") as f:
                header_code += f.read()
  
    encrypted_code_maps = {}
    for file_name in os.listdir(cl_kernel_dir):
        file_path = os.path.join(cl_kernel_dir, file_name)
        if file_path[-3:] == ".cl":
            with open(file_path, "r") as f:
                code_str = ""
                for line in f.readlines():
                    if "#include <common.h>" in line:
                        code_str += header_code
                    else:
                        code_str += line
                encrypted_code_maps[file_name[:-3]] = encrypt_code(code_str)

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(sys.path[0]))
    cpp_cl_encrypted_kernel = env.get_template(
        'str2map.h.jinja2').render(
            maps=encrypted_code_maps,
            variable_name='opencl_kernel_string_map')

    # output_dir = os.path.dirname(output_path)
    # if os.path.exists(output_dir):
    #     if os.path.isdir(output_dir):
    #         try:
    #             shutil.rmtree(output_dir)
    #         except OSError:
    #             raise RuntimeError(
    #                 "Cannot delete directory %s due to permission "
    #                 "error, inspect and remove manually" % output_dir)
    #     else:
    #         raise RuntimeError(
    #             "Cannot delete non-directory %s, inspect ",
    #             "and remove manually" % output_dir)
    # os.makedirs(output_dir)

    with open(output_path, "w") as w_file:
        w_file.write(cpp_cl_encrypted_kernel)

    print('Generate OpenCL kernel done.')

if __name__ == '__main__':
    encrypt_opencl_codegen("../../CL_kernels/", "../../opencl_kernels.cpp")
