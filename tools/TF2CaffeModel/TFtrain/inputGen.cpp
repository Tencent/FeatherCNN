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


#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv)
{
	if(argc<5)
	{
		printf("Usage: %s N C H W OnePercentage\n", argv[0]);
		printf("Here NCHW are integers, OnePercentage is an integer in [0,100]\n");
		exit(0); 
	}

	int N, C, H, W, OneP=8;
	if(sscanf(argv[1], "%d", &N)!=1 || sscanf(argv[2], "%d", &C)!=1 || sscanf(argv[3], "%d", &H)!=1 || sscanf(argv[4], "%d", &W) !=1 )	
	{
		printf("Usage: %s N C H W OnePercentage\n", argv[0]);
		printf("Here NCHW are integers, OnePercentage is an integer in [0,100]\n");
		exit(0); 
	}
	
	if(argc==6)	sscanf(argv[5], "%d", &OneP);
//	printf("NCHW=[%dx%dx%dx%d]\n", N, C, H, W);

	srand((unsigned)time(NULL));  	
	for(int i=0;i<N;i++)
	{
		for(int j=0;j<C;j++)
		for(int k=0;k<H;k++)
		for(int l=0;l<W;l++)
		{
			float t = 0;
			if(rand()%100<OneP)	t=rand()%1000/1000.0;
			printf("%f ", t); 
		}
		printf("\n");
	}
	return 1;
}
