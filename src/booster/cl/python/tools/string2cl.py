import sys
import os


root_path = "../kernels/"
out_path = "../../src/CL_kernel/"
flst = os.listdir(root_path)
print(flst)

for txt in flst:

    f = open(root_path + txt)
    outF = open(out_path + txt[:-4] + ".cl", 'w+')
    while True:

        thisline = f.readline()
        if not thisline:
            break
        sidx = thisline.find('"');
        eidx = thisline.rfind('\\n');
        filter_line = thisline[sidx+1:eidx]
        filter_line = filter_line.rstrip() + "\n"

        outF.write(filter_line)

    # outF.write('"')
    # outF.write(thisline[:-1])
    # outF.write('    \\')
    # outF.write('n')
    # outF.write('"')
    # outF.write(' \\')
    # outF.write('\n')

    f.close()
    outF.close()
