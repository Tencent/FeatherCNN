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
