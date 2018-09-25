#include <booster/sgemm.h>

#include <immintrin.h>
#include <string.h>

int align_ceil(int num, int align)
{
	return num + (align - (num % align)) % align;
}

void (*inner_kernel_Nx16)(int K, float *packA, float *packB, float *c, int ldc);

#include <stdio.h>
template<int N>
void inner_kernel_Nx16_template(int K, float *packA, float *packB, float *c, int ldc)
{
	float *aptr = packA;
	float *bptr = packB;
	float *cptr = c;
	__m256 va, va1, va2, va3, va4, va5;
	__m256 vb0, vb1, vb2, vB0, vB1, vB2;
	__m256 vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vcA, vcB;

	//N is 1, 2, 3, 4, 5, 6
	vc0 = _mm256_loadu_ps(cptr); 
	vc6 = _mm256_loadu_ps(cptr + 8);
	cptr += ldc;
	if(N > 1)//N is 2, 3, 4, 5, 6
	{
		vc1 = _mm256_loadu_ps(cptr);
		vc7 = _mm256_loadu_ps(cptr + 8);
		cptr += ldc;
	}
	if(N > 2)//N is 3, 4, 5, 6
	{
		vc2 = _mm256_loadu_ps(cptr);
		vc8 = _mm256_loadu_ps(cptr + 8);
		cptr += ldc;
	}
	if(N > 3)//N is 4, 5, 6
	{
		vc3 = _mm256_loadu_ps(cptr);
		vc9 = _mm256_loadu_ps(cptr + 8);
		cptr += ldc;
	}
	if(N > 4)//N is 5, 6
	{
		vc4 = _mm256_loadu_ps(cptr);
		vcA = _mm256_loadu_ps(cptr + 8);
		cptr += ldc;
	}
	if(N > 5)//N is 6
	{
		vc5 = _mm256_loadu_ps(cptr);
		vcB = _mm256_loadu_ps(cptr + 8);
	}
	vb0 = _mm256_loadu_ps(bptr);
	vb1 = _mm256_loadu_ps(bptr + 8);
	for(int p = 0; p < (K-1); ++p){
		va = _mm256_broadcast_ss(aptr);
		vc0 = _mm256_fmadd_ps(vb0, va, vc0);
		vc6 = _mm256_fmadd_ps(vb1, va, vc6);

		if(N > 1){
		va = _mm256_broadcast_ss(aptr+1);
		vc1 = _mm256_fmadd_ps(vb0, va, vc1);
		vc7 = _mm256_fmadd_ps(vb1, va, vc7);
		}

		if(N > 2){
		va = _mm256_broadcast_ss(aptr+2);
		vc2 = _mm256_fmadd_ps(vb0, va, vc2);
		vc8 = _mm256_fmadd_ps(vb1, va, vc8);
		}

		if(N > 3){
		va = _mm256_broadcast_ss(aptr+3);
		vc3 = _mm256_fmadd_ps(vb0, va, vc3);
		vc9 = _mm256_fmadd_ps(vb1, va, vc9);
		}

		if(N > 4){
		va = _mm256_broadcast_ss(aptr+4);
		vc4 = _mm256_fmadd_ps(vb0, va, vc4);
		vcA = _mm256_fmadd_ps(vb1, va, vcA);
		}


		if(N > 5){
		va = _mm256_broadcast_ss(aptr+5);
		vc5 = _mm256_fmadd_ps(vb0, va, vc5);
		vcB = _mm256_fmadd_ps(vb1, va, vcB);
		}

		vb0 = _mm256_loadu_ps(bptr + 16);
		vb1 = _mm256_loadu_ps(bptr + 24);
		bptr+=16;
		aptr+=N;
	}
	cptr = c;
	va = _mm256_broadcast_ss(aptr);
	vc0 = _mm256_fmadd_ps(vb0, va, vc0);
	vc6 = _mm256_fmadd_ps(vb1, va, vc6);
	_mm256_storeu_ps(cptr, vc0);
	_mm256_storeu_ps(cptr + 8, vc6);
	if(N > 1){
		va = _mm256_broadcast_ss(aptr+1);
		vc1 = _mm256_fmadd_ps(vb0, va, vc1);
		vc7 = _mm256_fmadd_ps(vb1, va, vc7);
		cptr += ldc;
		_mm256_storeu_ps(cptr, vc1);
		_mm256_storeu_ps(cptr + 8, vc7);
	}
	if(N > 2){
		va = _mm256_broadcast_ss(aptr+2);
		vc2 = _mm256_fmadd_ps(vb0, va, vc2);
		vc8 = _mm256_fmadd_ps(vb1, va, vc8);
		cptr += ldc;
		_mm256_storeu_ps(cptr, vc2);
		_mm256_storeu_ps(cptr + 8, vc8);
	}
	if(N > 3){
		va = _mm256_broadcast_ss(aptr+3);
		vc3 = _mm256_fmadd_ps(vb0, va, vc3);
		vc9 = _mm256_fmadd_ps(vb1, va, vc9);
		cptr += ldc;
		_mm256_storeu_ps(cptr, vc3);
		_mm256_storeu_ps(cptr + 8, vc9);
	}
	if(N > 4){
		va = _mm256_broadcast_ss(aptr+4);
		vc4 = _mm256_fmadd_ps(vb0, va, vc4);
		vcA = _mm256_fmadd_ps(vb1, va, vcA);
		cptr += ldc;
		_mm256_storeu_ps(cptr, vc4);
		_mm256_storeu_ps(cptr + 8, vcA);
	}
	if(N > 5){
		va = _mm256_broadcast_ss(aptr+5);
		vc5 = _mm256_fmadd_ps(vb0, va, vc5);
		vcB = _mm256_fmadd_ps(vb1, va, vcB);
		cptr += ldc;
		_mm256_storeu_ps(cptr, vc5);
		_mm256_storeu_ps(cptr + 8, vcB);
	}
}

void set_kernel(int k)
{
	switch(k)
	{
		case 1:
			inner_kernel_Nx16 = inner_kernel_Nx16_template<1>;
			break;
		case 2:
			inner_kernel_Nx16 = inner_kernel_Nx16_template<2>;
			break;
		case 3:
			inner_kernel_Nx16 = inner_kernel_Nx16_template<3>;
			break;
		case 4:
			inner_kernel_Nx16 = inner_kernel_Nx16_template<4>;
			break;
		case 5:
			inner_kernel_Nx16 = inner_kernel_Nx16_template<5>;
			break;
		case 0:
			inner_kernel_Nx16 = inner_kernel_Nx16_template<6>;
			break;
	}
}


template<bool fuseBias, bool fuseRelu>
inline void compute_block_activation(int M, int nc, int kc, float* packA, float* packB, float* loadC, float *C, int ldc, float* bias, int bias_len)
{
	//M is already aligned. 
	int nc_ceil = align_ceil(nc, 16);
	int nc_floor = nc - nc % 8;
	for(int i = 0; i < M - M % 6; i += 6)
	{
		//Load C into cache
		float* rC = C + i * ldc;
		for(int m = 0; m < 6; ++m)
		{
			float* pC = rC + m * ldc;
			float* pL = loadC + m * nc_ceil;
			for(int n = 0; n < nc_floor; n += 8)
			{
				_mm256_storeu_ps(pL + n, _mm256_loadu_ps(pC + n));
			}
			for(int n = nc - nc % 8; n < nc; ++n)
			{
				pL[n] = pC[n];
			}
		}
		for(int j = 0; j < nc_ceil; j+=16)
		{
			float* pC = loadC + j;
			float* pA = packA + i * kc;
			float* pB = packB + j * kc;
			inner_kernel_Nx16_template<6>(kc, pA, pB, pC, nc_ceil);
		}
		//Write Results
		for(int m = 0; m < 6; ++m)
		{
			float* pC = rC + m * ldc;
			float* pL = loadC + m * nc_ceil;
			__m256 vZero = _mm256_set1_ps(0.f);
			__m256 vBias = vZero;
			if(m + i < bias_len){
				if(fuseBias)
					vBias = _mm256_broadcast_ss(bias + i + m);
			}
			for(int n = 0; n < nc_floor; n += 8)
			{
				__m256 vec = _mm256_loadu_ps(pL + n);
				if(fuseBias)
					vec = _mm256_add_ps(vec, vBias);
				if(fuseRelu)
					vec = _mm256_max_ps(vec, vZero);

				_mm256_storeu_ps(pC + n, vec);
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
	int m_len = M % 6;
	if(m_len)
	{
		int i = M - M % 6;
		//Load C into cache
		float* rC = C + i * ldc;
		for(int m = 0; m < m_len; ++m)
		{
			float* pC = rC + m * ldc;
			float* pL = loadC + m * nc_ceil;
			for(int n = 0; n < nc_floor; n += 8)
			{
				_mm256_storeu_ps(pL + n, _mm256_loadu_ps(pC + n));
			}
			for(int n = nc - nc % 8; n < nc; ++n)
			{
				pL[n] = pC[n];
			}
		}
		for(int j = 0; j < nc_ceil; j+=16)
		{
			float* pC = loadC + j;
			float* pA = packA + i * kc;
			float* pB = packB + j * kc;
			inner_kernel_Nx16(kc, pA, pB, pC, nc_ceil);
		}
		//Write Results
		for(int m = 0; m < m_len; ++m)
		{
			float* pC = rC + m * ldc;
			float* pL = loadC + m * nc_ceil;
			__m256 vZero = _mm256_set1_ps(0.f);
			__m256 vBias = vZero;
			if(m + i < bias_len){
				if(fuseBias)
					vBias = _mm256_broadcast_ss(bias + i + m);
			}
			for(int n = 0; n < nc_floor; n += 8)
			{
				__m256 vec = _mm256_loadu_ps(pL + n);
				if(fuseBias)
					vec = _mm256_add_ps(vec, vBias);
				if(fuseRelu)
					vec = _mm256_max_ps(vec, vZero);

				_mm256_storeu_ps(pC + n, vec);
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

template void packed_sgemm_init<6>(int M, int K, int kc, float* packedA, float* A, int lda);

void pack_B_avx(int kc, int nc, float* packB, float* B, int ldb)
{
	const int COL_BATCH = 16;
	int nc_floor = nc - nc % COL_BATCH;
	for(int k = 0; k < kc; ++k)
	{
		float* pB = B + k * ldb;
		for(int j = 0; j < nc_floor; j+=COL_BATCH)
		{
			float* pPack = packB + (j / COL_BATCH) * kc * COL_BATCH + k * COL_BATCH;
			_mm256_storeu_ps(pPack, _mm256_loadu_ps(pB));
			_mm256_storeu_ps(pPack + 8, _mm256_loadu_ps(pB + 8));
			pB += 16;
		}
		if(nc_floor < nc)
		{
			int j = nc_floor;
			int n_len = nc - nc_floor;
			float* pPack = packB + (j / COL_BATCH) * kc * COL_BATCH + k * COL_BATCH;
			for(int i = 0; i < n_len; ++i)
			{
				pPack[i] = pB[i];
			}
		}
	}
}
	
template<bool fuseBias, bool fuseRelu>
void packed_sgemm_activation(int M, int N, int K, float *packA, float *b, int ldb, float *c, int ldc, int nc, int kc, float* bias, int num_threads, float* pack_array)
// void packed_sgemm_activation(int M, int N, int K, float *packA, float *b, int ldb, float *c, int ldc, int nc, int kc, float* bias)
{
	set_kernel(M % 6);
	for(int i = 0; i < M; ++i){
		memset(c + ldc * i, 0, sizeof(float) * N);
	}

	int M_align = align_ceil(M, 6);
	int N_align = align_ceil(N, 16);

	int NBlocks = (N_align + nc - 1) / nc;
	int KBlocks = (K + kc - 1) / kc;
	
	//float* packB = (float *) _mm_malloc(sizeof(float) * kc * nc, 32);
	//float* loadC = (float *) _mm_malloc(sizeof(float) * 6 * nc, 32);
	//printf("loadC %x %d\n", loadC, ((size_t) loadC) % 32);
	
	//Our GEMM is implemented in GEPB fashion, as the operands are row-major
	int k_len = kc;
	int n_len = nc;
	for(int kt = 0; kt < KBlocks - 1; ++kt)
	{
		//k_len = (kt == KBlocks - 1) ? (K - kt * kc) : kc;
#pragma omp parallel for num_threads(2)
		for(int nt = 0; nt < NBlocks; ++nt)
		{
			__attribute__((aligned(32))) float loadC[6 * nc];
			__attribute__((aligned(32))) float packB[kc * nc];
			//float* pA = packA + kt * kc * M_align;
			float* pA = packA + kt * kc * M;
			float* pB = b + kt * kc * ldb + nt * nc;
			float* pC = c + nt * nc; 
			if(nt == NBlocks - 1)
				n_len = N - nt * nc;
			else
				n_len = nc;
			memset(packB, 0, sizeof(float) * kc * nc);
			pack_B_avx(k_len, n_len, packB, pB, N);
			compute_block_activation<false, false>(M, n_len, k_len, pA, packB, loadC, pC, ldc, bias, M);
		}
	}
	{
		int kt = KBlocks - 1;
		k_len = (K - kt * kc);
		__attribute__((aligned(32))) float loadC[6 * nc];
#pragma omp parallel for num_threads(2)
		for(int nt = 0; nt < NBlocks; ++nt)
		{
			//float loadC[6 * nc];
			//float* pA = packA + kt * kc * M_align;
			__attribute__((aligned(32))) float loadC[6 * nc];
			__attribute__((aligned(32))) float packB[kc * nc];
			float* pA = packA + kt * kc * M;
			float* pB = b + kt * kc * ldb + nt * nc;
			float* pC = c + nt * nc; 
			if(nt == NBlocks - 1)
				n_len = N - nt * nc;
			else
				n_len = nc;
			//I'm going to pack B in here.
			memset(packB, 0, sizeof(float) * kc * nc);
			pack_B_avx(k_len, n_len, packB, pB, N);
			compute_block_activation<fuseBias, fuseRelu>(M, n_len, k_len, pA, packB, loadC, pC, ldc, bias, M);
		}
	}
	//_mm_free(packB);
	//_mm_free(loadC);
}

template void packed_sgemm_activation<false, false>(int, int, int, float *, float *, int, float *, int , int , int , float* , int, float*);
template void packed_sgemm_activation<false,  true>(int, int, int, float *, float *, int, float *, int , int , int , float* , int, float*);
template void packed_sgemm_activation<true,  false>(int, int, int, float *, float *, int, float *, int , int , int , float* , int, float*);
template void packed_sgemm_activation<true,   true>(int, int, int, float *, float *, int, float *, int , int , int , float* , int, float*);
