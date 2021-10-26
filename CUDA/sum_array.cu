
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_common.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>

__global__void sum_array_gpu(int*a, int*b,int*c,int size)
{
    int gid= threadIdx.x+ blockIdx*blockDim.x;
   if(gid < size)
   {
    c[gid]=a[gid]+b[gid];
   }
   
}

int main()
{
    int size=1000;
    int block_size=128;

    int NO_BYTES=size* size(int);
    int* h_a,*h_b,*gpu_results;
    h_a=(int*)malloc(NO_BYTES);
    h_b=(int*)malloc(NO_BYTES);
    gpu_results=(int*)malloc(NO_BYTES);

    time_t t;
    srand((unsigned)time(&t));

    for(int i = 0;i < size;i++)
    {
        h_a[i] = (int)(rand() & 0xff);
    }
    for(int i = 0;i < size;i++)
    {
        h_b[i] = (int)(rand() & 0xff);
    }
    //device pointer
    int* d_a,*d_b,*d_c;
    cudaMalloc((int**)&d_a,NO_BYTES);
    cudaMalloc((int**)&d_b,NO_BYTES);
    cudaMalloc((int**)&d_c,NO_BYTES);

    cudaMemcpy(d_a,h_a,NO_BYTES,cudaMemcpyHostToDevice);
     cudaMemcpy(d_b,h_b,NO_BYTES,cudaMemcpyHostToDevice);

    dim3 block(block_size);
    dim3 grid((size/block.x)+1);

    sum_array_gpu <<< grid,block>> (d_a,d_b,d_c,size);
    cudaDeviceSynchronize();

    cudaMemcpy(gpu_results ,d_c,NO_BYTES,cudaMemcpyDeviceToHost);


}