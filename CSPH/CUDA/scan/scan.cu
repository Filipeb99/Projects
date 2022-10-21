#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 1024
#define MAX_ELEMENTS_PER_BLOCK (THREADS_PER_BLOCK*2)
#define MAX_DEPTH_BANK 32
#define NUM_BLOCKS 16384

// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// Up-sweep kernel
__global__ void up_sweep(int *v_in, int *v_out, int n, int step){
    __shared__ int aux[MAX_ELEMENTS_PER_BLOCK];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int t_idx = threadIdx.x;

    if(t_idx < n/2){ // each thread picks 2 elements acording to the step, stores in shared memory.
        aux[2*t_idx] = v_in[step*(2*idx+1)-1]; 
        aux[2*t_idx+1] = v_in[step*(2*idx+2)-1];
    }

    int offset = 1; 

    for(int d = n>>1; d > 0; d >>= 1) {
        __syncthreads();
        if(t_idx < d){ // add a with b and store in b, pairs of (a,b) across the vector
            int a_idx = offset*(2*t_idx+1)-1; 
            int b_idx = offset*(2*t_idx+2)-1; // offset dictates the size between the chosen elements for addition
            aux[b_idx] += aux[a_idx]; // everything is acted upon the shared memory
        }
        offset *= 2;
    }

    __syncthreads();

    if(t_idx < n/2){ // each thread picks 2 elements acording to the step, stores in the global memory.
        v_out[step*(2*idx+1)-1] = aux[2*t_idx];
        v_out[step*(2*idx+2)-1] = aux[2*t_idx+1];
    }
}

// Down-sweep kernel
__global__ void down_sweep(int *v_in, int *v_out, int n, int step){
    __shared__ int aux[MAX_ELEMENTS_PER_BLOCK];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int t_idx = threadIdx.x;

    if(t_idx < n/2){
        aux[2*t_idx] = v_in[step*(2*idx+1)-1]; 
        aux[2*t_idx+1] = v_in[step*(2*idx+2)-1];
    }

    int offset = n; // same idea as up-sweep, but the offset decreases. 

    for(int d = 1; d < n; d *= 2){
        offset >>= 1;
        __syncthreads();
        if(t_idx < d){
            int a_idx = offset*(2*t_idx+1)-1; // different operations, according to the algorithm
            int b_idx = offset*(2*t_idx+2)-1;
            int t = aux[a_idx];
            aux[a_idx] = aux[b_idx];
            aux[b_idx] += t;
        }
    }
    __syncthreads();

    if(t_idx < n/2){
        v_out[step*(2*idx+1)-1] = aux[2*t_idx];
        v_out[step*(2*idx+2)-1] = aux[2*t_idx+1];
    }
}

void exclusive_scan(int* input, int N, int* result)
{
    // useful variables and arrays
    int num_blocks;
    int rounded_len = nextPow2(N);
    int size_bank[MAX_DEPTH_BANK];
    int cur_length = rounded_len;
    int i;
    int zero = 0;
    int last_bank;
    int step;

    // zero fill if not power of 2
    if (rounded_len != N) { 
        int *zero_fill = (int*)calloc((rounded_len - N),sizeof(int));
        cudaMemcpy((result+N), zero_fill, (rounded_len - N)*sizeof(int), cudaMemcpyHostToDevice);
        free(zero_fill);
    }

    // calculating the amount of up and down sweeps, along with the size of each "sub-problem"
    i = 0;
    while(1){
        size_bank[i] = cur_length;
        if(cur_length<=MAX_ELEMENTS_PER_BLOCK) break;
        cur_length = cur_length/MAX_ELEMENTS_PER_BLOCK;
        if(++i==MAX_DEPTH_BANK) exit(1);
    }

    last_bank = i; // sort of the depth of the to-be created sub-vectors

    step = 1; // successive up-sweeps, compressing.
    for(i=0;i<last_bank;i++){
        num_blocks = (size_bank[i]-1)/MAX_ELEMENTS_PER_BLOCK+1;
        up_sweep<<<num_blocks, THREADS_PER_BLOCK>>>(result, result, MAX_ELEMENTS_PER_BLOCK,step);
        step *= MAX_ELEMENTS_PER_BLOCK;
    }

    up_sweep<<<1, THREADS_PER_BLOCK>>>(result, result, size_bank[i], step); // final up-sweep

    cudaMemcpy(result+rounded_len-1, &zero, sizeof(int), cudaMemcpyHostToDevice); // zero the last element

    down_sweep<<<1, THREADS_PER_BLOCK>>>(result, result, size_bank[i], step); // first down-sweep

    step /= MAX_ELEMENTS_PER_BLOCK; // successive down-sweeps, decompressing.
    for(i=last_bank-1;i>=0;i--){
        num_blocks = (size_bank[i]-1)/MAX_ELEMENTS_PER_BLOCK+1;
        down_sweep<<<num_blocks, THREADS_PER_BLOCK>>>(result, result, MAX_ELEMENTS_PER_BLOCK, step);
        step /= MAX_ELEMENTS_PER_BLOCK;
    }
}

double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}

// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


__global__ void findRepeats (int N, int *input, int *output, int workload) {

    int start = (blockIdx.x * blockDim.x + threadIdx.x) * workload;
    int end = start + workload;
    if (end > N-1)
    {
        end = N-1;
    }

    for (int i = start; i < end; i++) {
        if (input[i] == input[i+1])
        {
            output[i] = 1;
        } else {
            output[i] = 0;
        }
    }
}

__global__ void writeRes (int N, int *input, int *output, int workload) {

    int start = (blockIdx.x * blockDim.x + threadIdx.x) * workload;
    int end = start + workload;
    if (end > N-1)
    {
        end = N-1;
    }

    for (int i = start; i < end; i++) {
        if (input[i] != input[i+1])
        {
            output[input[i]] = i;
        }
    }
}

// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {

    int num_threads = NUM_BLOCKS*THREADS_PER_BLOCK;

    int *tmp;
    int rounded_length = nextPow2(length);
    cudaMalloc((void**)&tmp, rounded_length*sizeof(int));

    if (length > num_threads)
    {
        int workload = (length + num_threads - 1) / num_threads;
        findRepeats<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(length, device_input, tmp, workload);
    } else {
        int num_blocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        findRepeats<<<num_blocks, THREADS_PER_BLOCK>>>(length, device_input, tmp, 1);
    }

    exclusive_scan(NULL, length, tmp);

    if (length > num_threads)
    {
        int workload = (length + num_threads - 1) / num_threads;
        writeRes<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(length, tmp, device_output, workload);
    } else {
        int num_blocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        writeRes<<<num_blocks, THREADS_PER_BLOCK>>>(length, tmp, device_output, 1);
    }

    int cnt = 0;

    cudaMemcpy(&cnt, (tmp+length-1), sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(tmp);

    return cnt; 
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
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
