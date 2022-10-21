#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"


// return GB/sec
float GBPerSec(int bytes, float sec) {
  return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}


__global__ void
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
       result[index] = alpha * x[index] + y[index];
}


void saxpyCuda(int N, float alpha, float* xarray, float* yarray, float* resultarray) {

    int totalBytes = sizeof(float) * 3 * N;

    const int threadsPerBlock = 512;

    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* device_x;
    float* device_y;
    float* device_result;
    
    int bytes = sizeof(float)*N;
    cudaMalloc(&device_x, bytes);
    cudaMalloc(&device_y, bytes);
    cudaMalloc(&device_result, bytes);
        
    // start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();
    
    cudaMemcpy(device_x, xarray, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, yarray, bytes, cudaMemcpyHostToDevice);
    
    double kernel_startTime = CycleTimer::currentSeconds();
   
    // run CUDA kernel
    saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);

    cudaDeviceSynchronize();

    double kernel_endTime = CycleTimer::currentSeconds();
    
    cudaMemcpy(resultarray, device_result, bytes, cudaMemcpyDeviceToHost);
    
    // end timing after result has been copied back into host memory
    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n",
		errCode, cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    printf("Effective BW by CUDA saxpy: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, GBPerSec(totalBytes, overallDuration));

    printf("Time for kernel execution: %f\n", (kernel_endTime-kernel_startTime)*1000);
    
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_result);
    
}

void printCudaInfo() {

    // print out stats about the GPU in the machine.

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
