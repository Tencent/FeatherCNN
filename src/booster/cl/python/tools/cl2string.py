import sys

f = open(sys.argv[1])
outF = open(sys.argv[2], 'w+')

while True:
    thisline = f.readline()
    if not thisline:
        break
    outF.write('"')
    outF.write(thisline[:-1])
    outF.write('    \\')
    outF.write('n')
    outF.write('"')
    outF.write(' \\')
    outF.write('\n')
    
f.close()
outF.close()