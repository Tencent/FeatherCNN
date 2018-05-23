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

#include <net.h>

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace feather;

int main(int argc, char *argv[]) {
	int i = 1, loopCnt = 1;
	char *pFname = (char *)"face.jpg";
	char *pModel = (char*)"regNet.feathermodel";
	char *pBlob = (char *)"mobilenet_v2_layer9_conv1x1";
	int num_threads = 1;
	struct timeval beg, end;

	printf("e.g.:  ./demo 1.jpg 48net.feathermodel prob1 10 1 \n");

	if (argc > 1) pFname = argv[i++];
	if (argc > 2) pModel = argv[i++];
	if (argc > 3) pBlob = argv[i++];
	if (argc > 4) loopCnt = atoi(argv[i++]);
	if (argc > 5) num_threads = atoi(argv[i++]);
	
	printf("img: %s model: %s blob: %s loopCnt: %d num_threads: %d\n", pFname, pModel, pBlob, loopCnt, num_threads);

	cv::Mat img = imread(pFname);
	if (img.empty())
	{
		printf("read img failed, %s\n", pFname);
		return -1;
	}

	img.convertTo(img, CV_32F, 1.0 / 128, -127.5/128);
	printf("c: %d w: %d h : %d step: %ld\n", img.channels(), img.cols, img.rows, img.step[0]);

	Net forward_net(num_threads);
	forward_net.InitFromPath(pModel);

	size_t data_size;
	forward_net.GetBlobDataSize(&data_size, pBlob);
	float *pOut = (float *)malloc(data_size*sizeof(float));

	gettimeofday(&beg, NULL);

	for(int loop = 0; loop < loopCnt; loop++)
	{
		int ret = forward_net.Forward((float*)img.data);
		forward_net.ExtractBlob(pOut, pBlob);
	}

	gettimeofday(&end, NULL);
	printf("\ntime: %ld ms, avg time : %.3f ms, loop: %d\n\n", (end.tv_sec*1000000 + end.tv_usec - beg.tv_sec*1000000 - beg.tv_usec)/1000, (end.tv_sec*1000000 + end.tv_usec - beg.tv_sec*1000000 - beg.tv_usec)/(1000.0*loopCnt), loopCnt);

	printf("out blob size: %lu\n", data_size);
	for(int i = 0 ; i < data_size; i++)
	{
		printf("%.3f ", pOut[i]);
	}

	free(pOut);
	printf("\n");
	return 0;
}
