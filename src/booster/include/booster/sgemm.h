//Tencent is pleased to support the open source community by making FeatherCNN available.

//Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.

//Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
//in compliance with the License. You may obtain a copy of the License at
//
//https://opensource.org/licenses/BSD-3-Clause
//
//Unless required by applicable law or agreed to in writing, software distributed
//under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//CONDITIONS OF ANY KIND, either express or implied. See the License for the
//specific language governing permissions and limitations under the License.

#pragma once

/*
 * Performs single-float matrix multiply C = A * B in row-major fashion,
 * where C is MxN, A is MxK and B is KxN.
 * Allocation requirement: C: get_aligned_size(M, N), A: get_aligned_size(M, K)
 */

int get_aligned_size(int M, int N);

template<int ROW_BATCH>
void packed_sgemm_init(int M, int K, int kc, float* packA, float* A, int lda);

//void packed_sgemm(int M, int N, int K, float *packA, float *B, int ldb, float *C, int ldc, int nc, int kc);
template<bool fuseBias, bool fuseRelu>
void packed_sgemm_activation(int M, int N, int K, float *packA, float *b, int ldb, float *c, int ldc, int nc, int kc, float* bias, int num_threads, float* pack_array);
