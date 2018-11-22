#ifndef HALF_H
#define HALF_H

typedef unsigned short half;
unsigned short hs_floatToHalf (float f);
int hs_halfToFloatRep (unsigned short c);
float hs_halfToFloat (unsigned short c);

#endif