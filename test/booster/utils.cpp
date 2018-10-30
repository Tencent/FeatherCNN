#include "utils.h"

#include <stdlib.h>
#include <time.h>

template<class T>
int rand_fill(T* arr, int len)
{
    srand(time(NULL));
    for(int i = 0; i < len; ++i)
    {
        arr[i] = (T) (rand() % 1024);
    }
    return 0;
}

template int rand_fill<float>(float*, int);
