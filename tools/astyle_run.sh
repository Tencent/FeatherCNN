#!/bin/bash
find ../src/ -not -path "../src/booster/include/CL/*" -not -path "../src/booster/include/booster/thpool.h" | grep -iE '(\.h|\.cpp|\.cc|\.hpp)$' | xargs astyle --style=allman --indent=spaces=4 --indent-classes --unpad-paren --pad-oper --pad-header --convert-tabs --indent-switches --indent-cases
find ../src/ -type f -name "*.orig" -print | xargs rm
