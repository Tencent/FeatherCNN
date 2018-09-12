//Tencent is pleased to support the open source community by making FeatherCNN available.

//Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.

//Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
//in compliance with the License. You may obtain a copy of the License at
//
//https://opensource.org/licenses/BSD-3-Clause
//
//Unless required by applicable law or agreed to in writing, software distributed
//under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//CONDITIONS OF ANY KIND, either express or implied. See the License for the
//specific language governing permissions and limitations under the License.

#include <booster/sgemv.h>
#include <booster/helper.h>

#include <assert.h>
#include <immintrin.h>
#include <string.h>
#if 0
void fully_connected_inference_direct(const int input_size, const int output_size, const float *x, const float *y, float *z, const int num_threads)
{
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int i = 0; i < output_size; i++)
    {
        float sum = 0.f;
        for (int j = 0; j < input_size; j++)
            sum += x[j] * y[i * input_size + j];
        z[i] = sum;
    }
}

void fully_connected_transpose_inference_sse8(const int input_size, const int output_size, const float *x, const float *y, float *z, const int num_threads)
{
    assert(input_size % 8 == 0);
    assert(output_size % 8 == 0);
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int k = 0; k < output_size / 8; k++)
    {
        const float *yPtr = y + k * 8 * input_size;
        __m128 res = {0.0, 0.0, 0.0, 0.0};
        __m128 res1 = {0.0, 0.0, 0.0, 0.0};
        __m128 va0, va1, va2, va3, vb0, vb1, vb2, vb3, vb4, vb5, vb6, vb7;
        for (int i = 0; i < input_size; i += 4)
        {
            //          __m128 v1, v2;
            //va = _mm_load_ps(x + i);

            vb0 = _mm_load_ps(yPtr);
            vb1 = _mm_load_ps(yPtr + 4);
            vb2 = _mm_load_ps(yPtr + 8);
            vb3 = _mm_load_ps(yPtr + 12);
            vb4 = _mm_load_ps(yPtr + 16);
            vb5 = _mm_load_ps(yPtr + 20);
            vb6 = _mm_load_ps(yPtr + 24);
            vb7 = _mm_load_ps(yPtr + 28);

	    va0 = _mm_broadcast_ss(x + i);
	    va1 = _mm_broadcast_ss(x + i + 1);
	    va2 = _mm_broadcast_ss(x + i + 2);
	    va3 = _mm_broadcast_ss(x + i + 3);

	    res = _mm_fmadd_ps(vb0, va0, res);
	    res1 = _mm_fmadd_ps(vb1, va0, res1);
	    res = _mm_fmadd_ps(vb2, va1, res);
	    res1 = _mm_fmadd_ps(vb3, va1, res1);
	    res = _mm_fmadd_ps(vb4, va2, res);
	    res1 = _mm_fmadd_ps(vb5, va2, res1);
	    res = _mm_fmadd_ps(vb6, va3, res);
	    res1 = _mm_fmadd_ps(vb7, va3, res1);

            yPtr += 32;
        }
        _mm_store_ps((z + 8 * k), res);
        _mm_store_ps((z + 8 * k + 4), res1);
    }
}

#include <stdio.h>
#include <stdlib.h>

//For fully connected layers, the weights matrix is transposed.
//To reduce memory allocation, a KxSTRIDE packed matrix is sufficient.
template<int STRIDE>
void packed_sgemv_transposed_init(const int N, const int K, float* matrix, float* packed_buffer)
{
	size_t N_aligned = N - N % STRIDE;
	for (int j = 0; j < N_aligned; j += STRIDE)
	{
		float* pMatrix = matrix + j * K;
		float* pPacked = packed_buffer;
		for(int k = 0; k < K; ++k)
		{
			for(int i = 0; i < STRIDE; ++i)
				pPacked[i] = pMatrix[i * K + k];
			pPacked += STRIDE;
		}
		memcpy(pMatrix, packed_buffer, STRIDE * K * sizeof(float));
	}
	int rem = N % STRIDE;
	if(rem > 0)
	{
		float* pMatrix = matrix + N_aligned * K;
		float* pPacked = packed_buffer;
		for(int k = 0; k < K; ++k)
		{
			for(int i = 0; i < rem; ++i)
				pPacked[i] = pMatrix[i * K + k];
			pPacked += rem;
		}
		memcpy(pMatrix, packed_buffer, rem * K * sizeof(float));
	}
}
template void packed_sgemv_transposed_init<8>(const int N, const int K, float* matrix, float* packed_matrix);
template void packed_sgemv_transposed_init<16>(const int N, const int K, float* matrix, float* packed_matrix);

//The packed matrix will only be scanned once. Cache miss would only occur with very large K, which is not a very common case.

template<bool fuseBias, bool fuseRelu>
void packed_sgemv(const int N, const int K, const float* A, const float* B, float* C, const float* bias_data, const int num_threads)
{
	assert(K % 4 == 0);
	int N_aligned = N - N % 16;
	__m256 vZero = _mm256_set1_ps(0.f);
#pragma omp parallel for
	for(int j = 0; j < N_aligned; j += 16)
	{
		const float *pB = B + j * K;
		float *pC = C + j;
		__m256 acc0 = _mm256_set1_ps(0.f);
		__m256 acc1 = _mm256_set1_ps(0.f);
		if(fuseBias)
		{
			acc0 = _mm256_load_ps(bias_data + j);
			acc1 = _mm256_load_ps(bias_data + j + 8);
		}
		__m256 va0, va1, va2, va3;
		__m256 vb0, vb1, vb2, vb3, vb4, vb5, vb6, vb7;
		for(int k = 0; k < K; k += 4)
		{
			vb0 = _mm256_load_ps(pB);
			vb1 = _mm256_load_ps(pB + 8);
			vb2 = _mm256_load_ps(pB + 16);
			vb3 = _mm256_load_ps(pB + 24);
			vb4 = _mm256_load_ps(pB + 32);
			vb5 = _mm256_load_ps(pB + 40);
			vb6 = _mm256_load_ps(pB + 48);
			vb7 = _mm256_load_ps(pB + 56);

			//print_floats(pB, 8);
			va0 = _mm256_broadcast_ss(A + k);
			va1 = _mm256_broadcast_ss(A + k + 1);
			va2 = _mm256_broadcast_ss(A + k + 2);
			va3 = _mm256_broadcast_ss(A + k + 3);

			acc0 = _mm256_fmadd_ps(vb0, va0, acc0);
			acc1 = _mm256_fmadd_ps(vb1, va0, acc1);
			acc0 = _mm256_fmadd_ps(vb2, va1, acc0);
			acc1 = _mm256_fmadd_ps(vb3, va1, acc1);
			acc0 = _mm256_fmadd_ps(vb4, va2, acc0);
			acc1 = _mm256_fmadd_ps(vb5, va2, acc1);
			acc0 = _mm256_fmadd_ps(vb6, va3, acc0);
			acc1 = _mm256_fmadd_ps(vb7, va3, acc1);
			pB += 64;	
		}
		if(fuseRelu)
		{
			acc0 = _mm256_max_ps(vZero, acc0);
			acc1 = _mm256_max_ps(vZero, acc1);
		}
		_mm256_store_ps(pC, acc0);
		_mm256_store_ps(pC + 8, acc1);
	}

	int rem = N % 16;
	if(rem == 8)
	{
		const float *pB = B + N_aligned * K;
		__m256 vacc = _mm256_set1_ps(0.f);
		if(fuseBias)
			vacc = _mm256_load_ps(bias_data + N_aligned);
		__m256 va, vb;
		float acc[4];
		for(int k = 0; k < K; ++k)
		{
			vb  = _mm256_loadu_ps(pB);
			va  = _mm256_broadcast_ss(A + k);
			vacc = _mm256_fmadd_ps(vb, va, vacc);
			pB += rem;
		}
		if(fuseRelu)
			vacc = _mm256_max_ps(vZero, vacc);
		_mm256_store_ps(C + N_aligned, vacc);
	} 
	else if(rem > 0)
	{
		const float *pB = B + N_aligned * K;
		float *pC = C + N_aligned;
		__m256 vacc = _mm256_set1_ps(0.f);
		if(fuseBias && rem >= 8)
			vacc = _mm256_load_ps(bias_data + N_aligned);

		__m256 va, vb;
		float acc[4];
		for(int i = 0; i < rem - 8; ++i)
		{
			if(fuseBias)
				acc[i] = bias_data[N_aligned + i];
			else
				acc[i] = 0;
		}
		for(int k = 0; k < K; ++k)
		{
			if(rem >= 8)
			{
				vb  = _mm256_loadu_ps(pB);
				va  = _mm256_broadcast_ss(A + k);
				vacc = _mm256_fmadd_ps(vb, va, vacc);
			}
			if(rem - 8 > 0)
			{
				for(int i = 0; i < rem - 8; ++i)	
					acc[i] = A[k] * pB[i + 8];
			}
			pB += rem;
		}
		if(fuseRelu)
		{
			vacc = _mm256_max_ps(vZero, vacc);
		}
		_mm256_store_ps(pC, vacc);
		for(int i = 0; i < rem - 8; ++i)
		{
			if(fuseRelu)
				pC[i + 8] = ((acc[i] > 0.f) ? acc[i] : 0.f);
			else
				pC[i + 8] = acc[i];
		}
	}
}

template void packed_sgemv<false, false>(const int N, const int K, const float* A, const float* B, float* C, const float* bias_data, const int num_threads);
template void packed_sgemv<true, false>(const int N, const int K, const float* A, const float* B, float* C, const float* bias_data, const int num_threads);
template void packed_sgemv<false, true>(const int N, const int K, const float* A, const float* B, float* C, const float* bias_data, const int num_threads);
template void packed_sgemv<true, true>(const int N, const int K, const float* A, const float* B, float* C, const float* bias_data, const int num_threads);



void fully_connected_inference_direct_BiasReLU(int input_size, int output_size, float *x, float *y, float *z, float* biasArr, int num_threads)
{
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int i = 0; i < output_size; i++)
    {
        float sum = 0.f;
        for (int j = 0; j < input_size; j++)
            sum += x[j] * y[i * input_size + j];

        sum += biasArr[i];
        if (sum < 0.f) sum = 0.f;
        z[i] = sum;
    }
}

void fully_connected_transpose_inference_sse8_BiasReLU(int input_size, int output_size, float *x, float *y, float *z, float* biasArr, int num_threads)
{
    assert(input_size % 8 == 0);
    assert(output_size % 8 == 0);
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int k = 0; k < output_size / 8; k++)
    {
        float *yPtr = y + k * 8 * input_size;
        const __m128 vzero = _mm_set1_ps(0.f);

        __m128 res  = _mm_load_ps(biasArr + k * 8);
        __m128 res1 = _mm_load_ps(biasArr + k * 8 + 4);

        __m128 va0, va1, va2, va3, vb0, vb1, vb2, vb3, vb4, vb5, vb6, vb7;
        for (int i = 0; i < input_size; i += 4)
        {
            vb0 = _mm_load_ps(yPtr);
            vb1 = _mm_load_ps(yPtr + 4);
            vb2 = _mm_load_ps(yPtr + 8);
            vb3 = _mm_load_ps(yPtr + 12);
            vb4 = _mm_load_ps(yPtr + 16);
            vb5 = _mm_load_ps(yPtr + 20);
            vb6 = _mm_load_ps(yPtr + 24);
            vb7 = _mm_load_ps(yPtr + 28);

	    va0 = _mm_broadcast_ss(x + i);
	    va1 = _mm_broadcast_ss(x + i + 1);
	    va2 = _mm_broadcast_ss(x + i + 2);
	    va3 = _mm_broadcast_ss(x + i + 3);

	    res = _mm_fmadd_ps(vb0, va0, res);
	    res1 = _mm_fmadd_ps(vb1, va0, res1);
	    res = _mm_fmadd_ps(vb2, va1, res);
	    res1 = _mm_fmadd_ps(vb3, va1, res1);
	    res = _mm_fmadd_ps(vb4, va2, res);
	    res1 = _mm_fmadd_ps(vb5, va2, res1);
	    res = _mm_fmadd_ps(vb6, va3, res);
	    res1 = _mm_fmadd_ps(vb7, va3, res1);

            yPtr += 32;
        }

        res  = _mm_max_ps(res, vzero);
        res1 = _mm_max_ps(res1, vzero);

        _mm_store_ps((z + 8 * k), res);
        _mm_store_ps((z + 8 * k + 4), res1);
    }
}

#else

template <bool fuseBias, bool fuseRelu>
void fully_connected_inference_direct(const int input_size, const int output_size, const float *x, const float *y, float *z, const int num_threads, float *bias_arr)
{
#pragma omp parallel for schedule(static) num_threads(num_threads)
	for (int i = 0; i < output_size; i++)
	{
		float sum = 0;
		for (int j = 0; j < input_size; j++)
			sum += x[j] * y[i * input_size + j];
		if (fuseBias)
			sum += bias_arr[i];
		if (fuseRelu)
			sum = (sum > 0.f) ? sum : 0.f;
		z[i] = sum;
	}
}

template <bool fuseBias, bool fuseRelu>
void fully_connected_transpose_inference(const int input_size, const int output_size, const float *x, const float *y, float *z, const int num_threads, float *bias_arr)
{
	assert(input_size % 8 == 0);
	assert(output_size % 8 == 0);
#pragma omp parallel for schedule(static) num_threads(num_threads)
	for (int k = 0; k < output_size / 8; k++)
	{
		__m128 vBias, vBias1; 
		const __m128 vZero = _mm_set1_ps(0.f);
		__m128 res = _mm_set1_ps(0.f);
		__m128 res1 = _mm_set1_ps(0.f);

		if(fuseBias)
		{
		vBias = _mm_load_ps(bias_arr + k * 8);
		vBias = _mm_load_ps(bias_arr + k * 8 + 4);
		}
		const float *yPtr = y + k * 8 * input_size;
		__m128 va0, va1, va2, va3, vb0, vb1, vb2, vb3, vb4, vb5, vb6, vb7;
		for (int i = 0; i < input_size; i += 4)
		{
			vb0 = _mm_load_ps(yPtr);
			vb1 = _mm_load_ps(yPtr + 4);
			vb2 = _mm_load_ps(yPtr + 8);
			vb3 = _mm_load_ps(yPtr + 12);
			vb4 = _mm_load_ps(yPtr + 16);
			vb5 = _mm_load_ps(yPtr + 20);
			vb6 = _mm_load_ps(yPtr + 24);
			vb7 = _mm_load_ps(yPtr + 28);

			va0 = _mm_broadcast_ss(x + i);
			va1 = _mm_broadcast_ss(x + i + 1);
			va2 = _mm_broadcast_ss(x + i + 2);
			va3 = _mm_broadcast_ss(x + i + 3);

			res = _mm_fmadd_ps(vb0, va0, res);
			res1 = _mm_fmadd_ps(vb1, va0, res1);
			res = _mm_fmadd_ps(vb2, va1, res);
			res1 = _mm_fmadd_ps(vb3, va1, res1);
			res = _mm_fmadd_ps(vb4, va2, res);
			res1 = _mm_fmadd_ps(vb5, va2, res1);
			res = _mm_fmadd_ps(vb6, va3, res);
			res1 = _mm_fmadd_ps(vb7, va3, res1);

			yPtr += 32;
		}

		if (fuseBias)
		{
			res  = _mm_add_ps(res, vBias);
			res1 = _mm_add_ps(res1, vBias1);
		}
		if (fuseRelu)
		{
			res = _mm_max_ps(res, vZero);
			res1 = _mm_max_ps(res1, vZero);
		}
		_mm_store_ps((z + 8 * k), res);
        _mm_store_ps((z + 8 * k + 4), res1);
	}
}

template void fully_connected_inference_direct<false, false>(const int, const int, const float *, const float *, float *, const int, float *);
template void fully_connected_inference_direct<false, true>(const int, const int, const float *, const float *, float *, const int, float *);
template void fully_connected_inference_direct<true, false>(const int, const int, const float *, const float *, float *, const int, float *);
template void fully_connected_inference_direct<true, true>(const int, const int, const float *, const float *, float *, const int, float *);

template void fully_connected_transpose_inference<false, false>(const int, const int, const float *, const float *, float *, const int, float *);
template void fully_connected_transpose_inference<false, true>(const int, const int, const float *, const float *, float *, const int, float *);
template void fully_connected_transpose_inference<true, false>(const int, const int, const float *, const float *, float *, const int, float *);
template void fully_connected_transpose_inference<true, true>(const int, const int, const float *, const float *, float *, const int, float *);

#endif

void matrixTranspose(float *array, size_t m, size_t n, float *buffer) //  A[m][n] -> A[n][m]
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			buffer[j * m + i] = array[i * n + j];
	memcpy(array, buffer, m * n * sizeof(float));
}
