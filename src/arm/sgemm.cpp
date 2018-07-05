#include "sgemm.h"
#include "helper.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>

#ifdef _OPENMP
#include <omp.h>
#endif

int align_ceil(int num, int align)
{
	return num + (align - (num % align)) % align;
}

void (*inner_kernel_Nx8)(int K, float *packA, float *packB, float *c, int ldc);

void inner_kernel_8x8(int K, float *packA, float *packB, float *c, int ldc)
{
	float *aptr = packA;
	float *bptr = packB;
	float *cptr = c;
	float32x4_t va, va1;
	float32x4_t vb0, vb1;
	float32x4_t vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vcA, vcB, vcC, vcD, vcE, vcF;

	vc0 = vld1q_f32(cptr); 
	vc1 = vld1q_f32(cptr + 4);
	cptr += ldc;
	vc2 = vld1q_f32(cptr);
	vc3 = vld1q_f32(cptr + 4);
	cptr += ldc;
	vc4 = vld1q_f32(cptr);
	vc5 = vld1q_f32(cptr + 4);
	cptr += ldc;
	vc6 = vld1q_f32(cptr);
	vc7 = vld1q_f32(cptr + 4);
	cptr += ldc;
	vc8 = vld1q_f32(cptr);
	vc9 = vld1q_f32(cptr + 4);
	cptr += ldc;
	vcA = vld1q_f32(cptr);
	vcB = vld1q_f32(cptr + 4);
	cptr += ldc;
	vcC = vld1q_f32(cptr);
	vcD = vld1q_f32(cptr + 4);
	cptr += ldc;
	vcE = vld1q_f32(cptr);
	vcF = vld1q_f32(cptr + 4);
	cptr += ldc;
	vb0 = vld1q_f32(bptr);
	vb1 = vld1q_f32(bptr + 4);
	for(int p = 0; p < (K-1); ++p){
		va = vld1q_f32(aptr);
		va1 = vld1q_f32(aptr + 4);
		vc0 = vfmaq_laneq_f32(vc0, vb0, va, 0);
		vc1 = vfmaq_laneq_f32(vc1, vb1, va, 0);
		vc2 = vfmaq_laneq_f32(vc2, vb0, va, 1);
		vc3 = vfmaq_laneq_f32(vc3, vb1, va, 1);
		vc4 = vfmaq_laneq_f32(vc4, vb0, va, 2);
		vc5 = vfmaq_laneq_f32(vc5, vb1, va, 2);
		vc6 = vfmaq_laneq_f32(vc6, vb0, va, 3);
		vc7 = vfmaq_laneq_f32(vc7, vb1, va, 3);
		vc8 = vfmaq_laneq_f32(vc8, vb0, va1, 0);
		vc9 = vfmaq_laneq_f32(vc9, vb1, va1, 0);
		vcA = vfmaq_laneq_f32(vcA, vb0, va1, 1);
		vcB = vfmaq_laneq_f32(vcB, vb1, va1, 1);
		vcC = vfmaq_laneq_f32(vcC, vb0, va1, 2);
		vcD = vfmaq_laneq_f32(vcD, vb1, va1, 2);
		vcE = vfmaq_laneq_f32(vcE, vb0, va1, 3);
		vcF = vfmaq_laneq_f32(vcF, vb1, va1, 3);

		vb0 = vld1q_f32(bptr + 8);
		vb1 = vld1q_f32(bptr + 12);
		bptr += 8;
		aptr+=8;
	}

	cptr = c;
	va = vld1q_f32(aptr);
	va1 = vld1q_f32(aptr + 4);
	vc0 = vfmaq_laneq_f32(vc0, vb0, va, 0);
	vc1 = vfmaq_laneq_f32(vc1, vb1, va, 0);
	vst1q_f32(cptr, vc0);
	vst1q_f32(cptr + 4, vc1);
	vc2 = vfmaq_laneq_f32(vc2, vb0, va, 1);
	vc3 = vfmaq_laneq_f32(vc3, vb1, va, 1);
	cptr += ldc;
	vst1q_f32(cptr, vc2);
	vst1q_f32(cptr + 4, vc3);
	vc4 = vfmaq_laneq_f32(vc4, vb0, va, 2);
	vc5 = vfmaq_laneq_f32(vc5, vb1, va, 2);
	cptr += ldc;
	vst1q_f32(cptr, vc4);
	vst1q_f32(cptr + 4, vc5);
	vc6 = vfmaq_laneq_f32(vc6, vb0, va, 3);
	vc7 = vfmaq_laneq_f32(vc7, vb1, va, 3);
	cptr += ldc;
	vst1q_f32(cptr, vc6);
	vst1q_f32(cptr + 4, vc7);
	vc8 = vfmaq_laneq_f32(vc8, vb0, va1, 0);
	vc9 = vfmaq_laneq_f32(vc9, vb1, va1, 0);
	cptr += ldc;
	vst1q_f32(cptr, vc8);
	vst1q_f32(cptr + 4, vc9);
	vcA = vfmaq_laneq_f32(vcA, vb0, va1, 1);
	vcB = vfmaq_laneq_f32(vcB, vb1, va1, 1);
	cptr += ldc;
	vst1q_f32(cptr, vcA);
	vst1q_f32(cptr + 4, vcB);
	vcC = vfmaq_laneq_f32(vcC, vb0, va1, 2);
	vcD = vfmaq_laneq_f32(vcD, vb1, va1, 2);
	cptr += ldc;
	vst1q_f32(cptr, vcC);
	vst1q_f32(cptr + 4, vcD);
	vcE = vfmaq_laneq_f32(vcE, vb0, va1, 3);
	vcF = vfmaq_laneq_f32(vcF, vb1, va1, 3);
	cptr += ldc;
	vst1q_f32(cptr, vcE);
	vst1q_f32(cptr + 4, vcF);
}

template<int N>
void inner_kernel_Nx8_template(int K, float *packA, float *packB, float *c, int ldc)
{
	float *aptr = packA;
	float *bptr = packB;
	float *cptr = c;
	float32x4_t va, va1;
	float32x4_t vb0, vb1;
	float32x4_t vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vcA, vcB, vcC, vcD, vcE, vcF;

	vc0 = vld1q_f32(cptr); 
	vc1 = vld1q_f32(cptr + 4);
	cptr += ldc;
	if(N > 1)
	{
		vc2 = vld1q_f32(cptr);
		vc3 = vld1q_f32(cptr + 4);
		cptr += ldc;
	}
	if(N > 2)
	{
		vc4 = vld1q_f32(cptr);
		vc5 = vld1q_f32(cptr + 4);
		cptr += ldc;
	}
	if(N > 3)
	{
		vc6 = vld1q_f32(cptr);
		vc7 = vld1q_f32(cptr + 4);
		cptr += ldc;
	}
	if(N > 4)
	{
		vc8 = vld1q_f32(cptr);
		vc9 = vld1q_f32(cptr + 4);
		cptr += ldc;
	}
	if(N > 5)
	{
		vcA = vld1q_f32(cptr);
		vcB = vld1q_f32(cptr + 4);
		cptr += ldc;
	}
	if(N > 6)
	{
		vcC = vld1q_f32(cptr);
		vcD = vld1q_f32(cptr + 4);
		cptr += ldc;
	}
	if(N > 7)
	{
		vcE = vld1q_f32(cptr);
		vcF = vld1q_f32(cptr + 4);
		cptr += ldc;
	}
	vb0 = vld1q_f32(bptr);
	vb1 = vld1q_f32(bptr + 4);
	for(int p = 0; p < (K-1); ++p){
		va = vdupq_n_f32(aptr[0]);
		vc0 = vfmaq_f32(vc0, vb0, va);
		vc1 = vfmaq_f32(vc1, vb1, va);
		if(N > 1){
			va = vdupq_n_f32(aptr[1]);
			vc2 = vfmaq_f32(vc2, vb0, va);
			vc3 = vfmaq_f32(vc3, vb1, va);
		}
		if(N > 2){
			va = vdupq_n_f32(aptr[2]);
			vc4 = vfmaq_f32(vc4, vb0, va);
			vc5 = vfmaq_f32(vc5, vb1, va);
		}
		if(N > 3){
			va = vdupq_n_f32(aptr[3]);
			vc6 = vfmaq_f32(vc6, vb0, va);
			vc7 = vfmaq_f32(vc7, vb1, va);
		}
		if(N > 4){
			va = vdupq_n_f32(aptr[4]);
			vc8 = vfmaq_f32(vc8, vb0, va);
			vc9 = vfmaq_f32(vc9, vb1, va);
		}
		if(N > 5){
			va = vdupq_n_f32(aptr[5]);
			vcA = vfmaq_f32(vcA, vb0, va);
			vcB = vfmaq_f32(vcB, vb1, va);
		}
		if(N > 6){
			va = vdupq_n_f32(aptr[6]);
			vcC = vfmaq_f32(vcC, vb0, va);
			vcD = vfmaq_f32(vcD, vb1, va);
		}
		if(N > 7){
			va = vdupq_n_f32(aptr[7]);
			vcE = vfmaq_f32(vcE, vb0, va);
			vcF = vfmaq_f32(vcF, vb1, va);
		}

		vb0 = vld1q_f32(bptr + 8);
		vb1 = vld1q_f32(bptr + 12);
		//tool::print_floats(bptr + 8, 8);
		bptr += 8;
		aptr+=N;
	}

	cptr = c;
	if(N > 0){
		va = vdupq_n_f32(aptr[0]);
		vc0 = vfmaq_f32(vc0, vb0, va);
		vc1 = vfmaq_f32(vc1, vb1, va);
		vst1q_f32(cptr, vc0);
		vst1q_f32(cptr + 4, vc1);
	}
	if(N > 1){
		va = vdupq_n_f32(aptr[1]);
		vc2 = vfmaq_f32(vc2, vb0, va);
		vc3 = vfmaq_f32(vc3, vb1, va);
		cptr += ldc;
		vst1q_f32(cptr, vc2);
		vst1q_f32(cptr + 4, vc3);
	}
	if(N > 2){
		va = vdupq_n_f32(aptr[2]);
		vc4 = vfmaq_f32(vc4, vb0, va);
		vc5 = vfmaq_f32(vc5, vb1, va);
		cptr += ldc;
		vst1q_f32(cptr, vc4);
		vst1q_f32(cptr + 4, vc5);
	}
	if(N > 3){
		va = vdupq_n_f32(aptr[3]);
		vc6 = vfmaq_f32(vc6, vb0, va);
		vc7 = vfmaq_f32(vc7, vb1, va);
		cptr += ldc;
		vst1q_f32(cptr, vc6);
		vst1q_f32(cptr + 4, vc7);
	}
	if(N > 4){
		va = vdupq_n_f32(aptr[4]);
		vc8 = vfmaq_f32(vc8, vb0, va);
		vc9 = vfmaq_f32(vc9, vb1, va);
		cptr += ldc;
		vst1q_f32(cptr, vc8);
		vst1q_f32(cptr + 4, vc9);
	}
	if(N > 5){
		va = vdupq_n_f32(aptr[5]);
		vcA = vfmaq_f32(vcA, vb0, va);
		vcB = vfmaq_f32(vcB, vb1, va);
		cptr += ldc;
		vst1q_f32(cptr, vcA);
		vst1q_f32(cptr + 4, vcB);
	}
	if(N > 6){
		va = vdupq_n_f32(aptr[6]);
		vcC = vfmaq_f32(vcC, vb0, va);
		vcD = vfmaq_f32(vcD, vb1, va);
		cptr += ldc;
		vst1q_f32(cptr, vcC);
		vst1q_f32(cptr + 4, vcD);
	}
	if(N > 7){
		va = vdupq_n_f32(aptr[7]);
		vcE = vfmaq_f32(vcE, vb0, va);
		vcF = vfmaq_f32(vcF, vb1, va);
		cptr += ldc;
		vst1q_f32(cptr, vcE);
		vst1q_f32(cptr + 4, vcF);
	}
}

void set_kernel(int k)
{
	switch(k)
	{
		case 1:
			inner_kernel_Nx8 = inner_kernel_Nx8_template<1>;
			break;
		case 2:
			inner_kernel_Nx8 = inner_kernel_Nx8_template<2>;
			break;
		case 3:
			inner_kernel_Nx8 = inner_kernel_Nx8_template<3>;
			break;
		case 4:
			inner_kernel_Nx8 = inner_kernel_Nx8_template<4>;
			break;
		case 5:
			inner_kernel_Nx8 = inner_kernel_Nx8_template<5>;
			break;
		case 6:
			inner_kernel_Nx8 = inner_kernel_Nx8_template<6>;
			break;
		case 7:
			inner_kernel_Nx8 = inner_kernel_Nx8_template<7>;
			break;
		case 0:
			inner_kernel_Nx8 = inner_kernel_Nx8_template<8>;
			break;
	}
}

template<bool fuseBias, bool fuseRelu>
inline void compute_block_activation(int M, int nc, int kc, float* packA, float* packB, float* loadC, float *C, int ldc, float* bias, int bias_len)
{
	const int COL_BATCH = 8;
	const int ROW_BATCH = 8;
	//M is already aligned. 
	const int nc_ceil = align_ceil(nc, COL_BATCH);
	const int nc_floor = nc - nc % 8;
	for(int i = 0; i < M - M % ROW_BATCH; i += ROW_BATCH)
	{
		//Load C into cache
		float* rC = C + i * ldc;
		for(int m = 0; m < ROW_BATCH; ++m)
		{
			float* pC = rC + m * ldc;
			float* pL = loadC + m * nc_ceil;
			for(int n = 0; n < nc_floor; n += 8)
			{
				vst1q_f32(pL + n, vld1q_f32(pC + n));
				vst1q_f32(pL + n + 4, vld1q_f32(pC + n + 4));
			}
			for(int n = nc - nc % 8; n < nc; ++n)
			{
				pL[n] = pC[n];
			}
		}
		
		int step = COL_BATCH * kc;	
		float* pA = packA + i * kc;
		float* pB = packB;
		for(int j = 0; j < nc_ceil; j+=COL_BATCH)
		{
			float* pC = loadC + j;
			inner_kernel_8x8(kc, pA, pB, pC, nc_ceil);
			pB += step;
		}
		//Write Results
		for(int m = 0; m < ROW_BATCH; ++m)
		{
			float* pC = rC + m * ldc;
			float* pL = loadC + m * nc_ceil;
			float32x4_t vZero = vdupq_n_f32(0.f);
			float32x4_t vBias = vZero;
			if(m + i < bias_len){
				if(fuseBias)
					vBias = vdupq_n_f32(bias[i + m]);
			}
			for(int n = 0; n < nc_floor; n += 8)
			{
				float32x4_t vec0 = vld1q_f32(pL + n);
				float32x4_t vec1 = vld1q_f32(pL + n + 4);
				if(fuseBias)
				{
					vec0 = vaddq_f32(vec0, vBias);
					vec1 = vaddq_f32(vec1, vBias);
				}
				if(fuseRelu)
				{
					vec0 = vmaxq_f32(vec0, vZero);
					vec1 = vmaxq_f32(vec1, vZero);
				}

				vst1q_f32(pC + n, vec0);
				vst1q_f32(pC + n + 4, vec1);
			}
			//Last column batch.
			for(int n = nc - nc % 8; n < nc; ++n)
			{
				float l = pL[n];
				if(fuseBias && ((i + m) < bias_len))
					l += bias[i + m];
				if(fuseRelu)
					l = (l > 0) ? l : 0;
				pC[n] = l;
			}
		}
	}
	int m_len = M % ROW_BATCH;
	if(m_len)
	{
		int i = M - M % ROW_BATCH;
		//Load C into cache
		float* rC = C + i * ldc;
		for(int m = 0; m < m_len; ++m)
		{
			float* pC = rC + m * ldc;
			float* pL = loadC + m * nc_ceil;
			for(int n = 0; n < nc_floor; n += 8)
			{
				vst1q_f32(pL + n, vld1q_f32(pC + n));
				vst1q_f32(pL + n + 4, vld1q_f32(pC + n + 4));
			}
			for(int n = nc - nc % 8; n < nc; ++n)
			{
				pL[n] = pC[n];
			}
		}
		for(int j = 0; j < nc_ceil; j+=COL_BATCH)
		{
			float* pC = loadC + j;
			float* pA = packA + i * kc;
			float* pB = packB + j * kc;
			inner_kernel_Nx8(kc, pA, pB, pC, nc_ceil);
		}
		//Write Results
		for(int m = 0; m < m_len; ++m)
		{
			float* pC = rC + m * ldc;
			float* pL = loadC + m * nc_ceil;
			float32x4_t vZero = vdupq_n_f32(0.f);
			float32x4_t vBias = vZero;
			if(m + i < bias_len){
				if(fuseBias)
					vBias = vdupq_n_f32(bias[i + m]);
			}
			for(int n = 0; n < nc_floor; n += 8)
			{
				float32x4_t vec0 = vld1q_f32(pL + n);
				float32x4_t vec1 = vld1q_f32(pL + n + 4);
				if(fuseBias)
				{
					vec0 = vaddq_f32(vec0, vBias);
					vec1 = vaddq_f32(vec1, vBias);
				}
				if(fuseRelu)
				{
					vec0 = vmaxq_f32(vec0, vZero);
					vec1 = vmaxq_f32(vec1, vZero);
				}

				vst1q_f32(pC + n, vec0);
				vst1q_f32(pC + n + 4, vec1);
			}
			//Last column batch.
			for(int n = nc - nc % 8; n < nc; ++n)
			{
				float l = pL[n];
				if(fuseBias && ((i + m) < bias_len))
					l += bias[i + m];
				if(fuseRelu)
					l = (l > 0) ? l : 0;
				pC[n] = l;
			}
		}
	}
}

//Decide how many rows should be packed together.
template<int ROW_BATCH>
void packed_sgemm_init(int M, int K, int kc, float* packA, float* A, int lda)
{
	//int M_align = align_ceil(M, 6);
	for(int p = 0; p < K; p += kc)
	{
		
		//The last row batch may not have sufficient rows
		//Implicit padding so as to reduce code complexity for packed_sgemm
		//float* pPack = packA + (p / kc) * M_align * kc;
		float* pPack = packA + (p / kc) * M * kc;
		for(int i = 0; i < M; i += ROW_BATCH)
		{
			int k_len = kc;
			int j_len = ROW_BATCH;
			if(M - i < ROW_BATCH){
				j_len = M - i;
			}
			float* pA = A + i * lda + p;
			if(K - p < kc)
				k_len = K - p;
			//Every ROW_BATCH rows are batched together.
			for(int k = 0; k < k_len; ++k)
			{
				for(int j = 0; j < j_len; ++j)
				{
					pPack[j] = pA[j * lda];
				}
				pPack += j_len;
				pA++;
			}
		}
	}
}


//template void packed_sgemm_init<6>(int M, int K, int kc, float* packedA, float* A, int lda);
template void packed_sgemm_init<8>(int M, int K, int kc, float* packedA, float* A, int lda);

void pack_B_neon(int kc, int nc, float* packB, float* B, int ldb)
{
	//const int COL_BATCH = 16;
	const int COL_BATCH = 8;
	int nc_floor = nc - nc % COL_BATCH;
	for(int k = 0; k < kc; ++k)
	{
		float* pB = B + k * ldb;

		float* pPack = packB + k*COL_BATCH;
		int    step = COL_BATCH*kc;
		for(int j = 0; j < nc_floor; j += COL_BATCH)
		{
//			float* pPack = packB + (j / COL_BATCH) * kc * COL_BATCH + k * COL_BATCH;
			vst1q_f32(pPack,     vld1q_f32(pB));
			vst1q_f32(pPack + 4, vld1q_f32(pB + 4));
			pB += 8;
			pPack += step;
		}
		if(nc_floor < nc)
		{
			int j = nc_floor;
			int n_len = nc - nc_floor;
			float* pPack = packB + j  * kc  + k*COL_BATCH;
			for(int i = 0; i < n_len; ++i)
			{
				pPack[i] = pB[i];
			}
		}
	}
}
	
template<bool fuseBias, bool fuseRelu>
void packed_sgemm_activation(int M, int N, int K, float *packA, float *b, int ldb, float *c, int ldc, int nc, int kc, float* bias, int num_threads, float* pack_array)
{
	const int ROW_BATCH = 8;
	const int COL_BATCH = 8;
	set_kernel(M % ROW_BATCH);
	for(int i = 0; i < M; ++i){
		memset(c + ldc * i, 0, sizeof(float) * N);
	}

	int M_align = align_ceil(M, ROW_BATCH);
	int N_align = align_ceil(N, COL_BATCH);

	int NBlocks = (N_align + nc - 1) / nc;
	int KBlocks = (K + kc - 1) / kc;

	//printf("Nblocks %d K blocks %d\n", NBlocks, KBlocks);
	
	//float *packB = (float*) malloc(kc * nc * sizeof(float));
	//Our GEMM is implemented in GEPB fashion, as the operands are row-major
	float* packB = pack_array;
	float* loadC = pack_array + kc * nc;
	int k_len = kc;
	int n_len = nc;
	for(int kt = 0; kt < KBlocks - 1; ++kt)
	{
		//k_len = (kt == KBlocks - 1) ? (K - kt * kc) : kc;
#pragma omp parallel for num_threads(num_threads) schedule(static)
      	for(int nt = 0; nt < NBlocks; ++nt)
      	{
#ifdef _OPENMP
			int tid = omp_get_thread_num();
			packB = pack_array + tid * (kc + 8) * nc;
			loadC = packB + kc * nc;
#endif
			//float loadC[ROW_BATCH * nc];
			//float packB[kc * nc];
			//float* pA = packA + kt * kc * M_align;
			float* pA = packA + kt * kc * M;
			float* pB = b + kt * kc * ldb + nt * nc;
			float* pC = c + nt * nc; 
			if(nt == NBlocks - 1)
				n_len = N - nt * nc;
			else
				n_len = nc;
			//I'm going to pack B in here.
//			memset(packB, 0, sizeof(float) * kc * nc);
			pack_B_neon(k_len, n_len, packB, pB, N);
			compute_block_activation<false, false>(M, n_len, k_len, pA, packB, loadC, pC, ldc, bias, M);
		}
	}
	{
		int kt = KBlocks - 1;
		k_len = (K - kt * kc);
#pragma omp parallel for num_threads(num_threads) schedule(static)
      	for(int nt = 0; nt < NBlocks; ++nt)
      	{
#ifdef _OPENMP
			int tid = omp_get_thread_num();
			packB = pack_array + tid * (kc + 8) * nc;
			loadC = packB + kc * nc;
#endif
			//float loadC[ROW_BATCH * nc];
			//float packB[kc * nc];
			float* pA = packA + kt * kc * M;
			float* pB = b + kt * kc * ldb + nt * nc;
			float* pC = c + nt * nc; 
			if(nt == NBlocks - 1)
				n_len = N - nt * nc;
			else
				n_len = nc;
//			memset(packB, 0, sizeof(float) * kc * nc);
			pack_B_neon(k_len, n_len, packB, pB, N);
			compute_block_activation<fuseBias, fuseRelu>(M, n_len, k_len, pA, packB, loadC, pC, ldc, bias, M);
		}
	}
	//free(packB);
}

template void packed_sgemm_activation<false, false>(int, int, int, float *, float *, int, float *, int , int , int , float* , int, float*);
template void packed_sgemm_activation<false,  true>(int, int, int, float *, float *, int, float *, int , int , int , float* , int, float*);
template void packed_sgemm_activation<true,  false>(int, int, int, float *, float *, int, float *, int , int , int , float* , int, float*);
template void packed_sgemm_activation<true,   true>(int, int, int, float *, float *, int, float *, int , int , int , float* , int, float*);
