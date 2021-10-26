
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>



__global__ void mem_trd_test(int *input)
{
    int threads_in_block = (blockDim.x * blockDim.y * blockDim.z);
 
    int thread_offset = 
        threadIdx.x +   // X axis
        (threadIdx.y * blockDim.x) +  // Y axis
        (threadIdx.z * (blockDim.x * blockDim.y));  // Z axis
 
    int block_offset = 
        blockIdx.x * threads_in_block +     // X axis
        blockIdx.y * blockDim.x * threads_in_block +    // Y axis
        blockIdx.z * blockDim.x * blockDim.y * threads_in_block;    // Z axis
 
    int pos = block_offset + thread_offset;
 
    if(blockIdx.x <= 1 && blockIdx.y <= 1 && blockIdx.z <= 1){
        printf(
            "blockIdx.x = %d  blockIdx.y = %d  blockIdx.z = %d      "
            "threadIdx.x = %d  threadIdx.y = %d  threadIdx.z = %d       "
            "block_offset = %d  thread_offset = %d  position = %d  value = %d\n", 
            blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z,
            block_offset, thread_offset, pos, grid_device[pos]
        );
    }

}


int main()
{
    int size = 64;

    int byte_size = size * sizeof(int);

    int * h_input;

    h_input=(int*)malloc(byte_size);

    time_t t;
    srand((unsigned)time(&t));

    for(int i = 0;i < size;i++)
    {
        h_input[i] = (int)(rand() & 0xff);
    }

    int * d_input;
    cudaMalloc((void**)&d_input,byte_size);

    cudaMemcpy(d_input,h_input,byte_size,cudaMemcpyHostToDevice);
    dim3 block(4,4,4);
    dim3 grid(2,2,2);

    mem_trd_test<<< grid , block >>> (d_input);
    cudaDeviceSynchronize();
    free(h_input);
    cudaFree(d_input);


    cudaDeviceReset();
    return 0;

}