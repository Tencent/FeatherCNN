#include "sgemm.h"

void externalPackA(int M, int L, float* packA, float* a, int lda){}//External packing for A, requires space allocation for packA
void block_sgemm_external_pack_threading( int M, int N, int L, float *A, float *B, float *C, int num_threads){}


void externalPackA8(int M, int L, float* packA, float* a, int lda){}//External packing for A, requires space allocation for packA
void block_sgemm_external_pack_threading_8x8( int M, int N, int L, float *A, float *B, float *C, int num_threads){}