#include "cuda_runtime.h"

#include "device_launch_parameters.h"


#include <stdio.h>

#include<stdlib.h>

#include<time.h>



//---------------------------------------------------------------------------------------

/* helper function to log cuda error code*/

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)

{

    if (code != cudaSuccess)

    {

        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

        if (abort) exit(code);

    }

}


//---------------------------------------------------------------------------------------

/* helper function to compare device and host result*/

bool compare_results(int* d_result, int* h_result, int aray_size)

{

    for (int i = 0; i < aray_size; ++i)

    {

        if (d_result[i] != h_result[i])

        {

            return false;

        }

    }


    return true;

}


// kernal definition to calculate summation of 3 array

__global__ void summation_3_array_kernel(int *array1, int* array2, int* array3, int* result, int array_size)

{

    int tid = threadIdx.x + blockIdx.x * blockDim.x;


    // since calculating parallel on gpu, no need to have loops like traditional CPU calculation

    if (tid < array_size)

    {

         result[tid] = array1[tid] + array2[tid] + array3[tid];

    } 

}


// CUP side summation function

void summation_3_array_host(int* array1, int* array2, int* array3, int* result, int array_size)

{

    for (int i = 0; i < array_size; ++i)

    {

        result[i] = array1[i] + array2[i] + array3[i];

    }

}

//--------------------------------------------------------------------------------------------------------------------------


int main()

{

    int array_size = 1000000;

    int block_size = 512;

    int grid_size = (array_size / block_size) + 1;


    size_t NO_BYTES = sizeof(int) * array_size;

    int* h_array_a = (int*)malloc(NO_BYTES);

    int* h_array_b = (int*)malloc(NO_BYTES);

    int* h_array_c = (int*)malloc(NO_BYTES);

    int* h_sum_result = (int*)malloc(NO_BYTES);

    int* gpu_calculated_result = (int*)malloc(NO_BYTES);


    memset(gpu_calculated_result, 0, NO_BYTES);

    memset(h_sum_result, 0, NO_BYTES);


    time_t t;

    srand(time(&t));                   // randon no. generator seed


    for (int i = 0; i < array_size; ++i)                   // assigning random value from 0 - 4194304 to array

    {

        h_array_a[i] = rand() & 4194304;

        h_array_b[i] = rand() & 4194304;

        h_array_c[i] = rand() & 4194304;

    }

   

   

    /*Host side summation to validate calcuated result*/

    clock_t cpu_start, cpu_end;

    cpu_start = clock();

    summation_3_array_host(h_array_a, h_array_b, h_array_c, h_sum_result, array_size);

    cpu_end = clock();



//--------------------------------------------------------------------------------------------------------------------------

    /*Device side summation calculation*/

    int* d_array_a, * d_array_b, * d_array_c, *d_sum_result;


    gpuErrchk(cudaMalloc((int**)&d_array_a, NO_BYTES));

    gpuErrchk(cudaMalloc((int**)&d_array_b, NO_BYTES));

    gpuErrchk(cudaMalloc((int**)&d_array_c, NO_BYTES));

    gpuErrchk(cudaMalloc((int**)&d_sum_result, NO_BYTES));


    clock_t gpu_copy_start, gpu_copy_end;

    gpu_copy_start = clock();

    gpuErrchk(cudaMemcpy(d_array_a, h_array_a, NO_BYTES, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_array_b, h_array_b, NO_BYTES, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_array_c, h_array_c, NO_BYTES, cudaMemcpyHostToDevice));

    gpu_copy_end = clock();


    // Change block and grid size as per array_size


    dim3 block(block_size);

    dim3 grid(grid_size);


    clock_t gpu_execution_start, gpu_execution_end;

    gpu_execution_start = clock();

    summation_3_array_kernel << < grid, block >> > (d_array_a, d_array_b, d_array_c, d_sum_result, array_size);

    cudaDeviceSynchronize();

    gpu_execution_end = clock();

   

    clock_t gpu_copy_back_start, gpu_copy_back_end;

    gpu_copy_back_start = clock();

    gpuErrchk(cudaMemcpy(gpu_calculated_result, d_sum_result, NO_BYTES, cudaMemcpyDeviceToHost));

    gpu_copy_back_end = clock();


    cudaDeviceReset();


//--------------------------------------------------------------------------------------------------------------------------

    /*compare results cpu summation and gpu summation */

    bool res=compare_results(gpu_calculated_result, h_sum_result, array_size);

    if(res)

        printf("Result match: Success\n");

    else

        printf("Result do not match: Failure\n");


//-------------------------------------------------------------------------------------------------------------------------- 

    /*Printing CPU and GPU execution time*/

    double cpu_total_time = (double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC);


    double gpu_host_to_device_transfer_time = (double)((double)(gpu_copy_end - gpu_copy_start) / CLOCKS_PER_SEC);

    double gpu_execution_time = (double)((double)(gpu_execution_end - gpu_execution_start) / CLOCKS_PER_SEC);

    double gpu_device_to_host_transfer_time = (double)((double)(gpu_copy_back_end - gpu_copy_back_start) / CLOCKS_PER_SEC);


    double gpu_total_time = gpu_host_to_device_transfer_time + gpu_execution_time + gpu_device_to_host_transfer_time;


    printf("Block size: %d and grid size: %d\n", block_size, grid_size);


    printf("CPU total execution time : %4.8f \n", cpu_total_time);

    printf("Host to Device transfer time : %4.6f \n", gpu_host_to_device_transfer_time);

    printf("GPU execution time : %4.6f \n", gpu_execution_time);

    printf("Device to Host transfer time : %4.6f \n", gpu_device_to_host_transfer_time);

    printf("GPU total execution time : %4.6f \n", gpu_total_time);


//--------------------------------------------------------------------------------------------------------------------------

    free(h_array_a);

    free(h_array_b);

    free(h_array_c);

    free(h_sum_result);


    cudaFree(d_array_a);

    cudaFree(d_array_b);

    cudaFree(d_array_c);

    cudaFree(d_sum_result);

   

    return 0;

}