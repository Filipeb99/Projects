#include <CL/sycl.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#define NNZ 76879419
#define DIM0 12092
#define DIM1 9184
#define DIM2 28818
#define MATRIX_DIM 64
#define MATRIX_VAL 0.23

// Info
const std::string tensorName = "nell-2.tns";

typedef std::vector<int> intVector;
typedef std::vector<float> floatVector;
typedef struct CSFtensor {
    intVector slcIdx, slcPtr;
    intVector fbrIdx, fbrPtr;
    intVector kIdx;
    floatVector values;
} CSFtensor;

using namespace sycl;

double ttmKernel (queue &q1, queue &q2, const CSFtensor &tensor, const floatVector &matrix, floatVector &output) {
    auto startKernel = std::chrono::steady_clock::now();
    
    auto workload = tensor.fbrIdx.size() / 5;
    
    range<2> localSize(1, MATRIX_DIM);
    
    range<2> deviceGlobalSize(1 * workload, MATRIX_DIM);
    nd_range<2> deviceNum_items(deviceGlobalSize, localSize);
    
    range<2> hostGlobalSize(4 * workload, MATRIX_DIM);
    nd_range<2> hostNum_items(hostGlobalSize, localSize);
    
    auto matrixD1 = malloc_device<float>(matrix.size(), q1);
    auto fbrPtrD1 = malloc_device<int>(tensor.fbrPtr.size(), q1), kIdxD1 = malloc_device<int>(tensor.kIdx.size(), q1);
    auto valuesD1 = malloc_device<float>(tensor.values.size(), q1), outputD1 = malloc_device<float>(output.size(), q1);
    q1.memcpy(matrixD1, matrix.data(), matrix.size());
    q1.memcpy(fbrPtrD1, tensor.fbrPtr.data(), tensor.fbrPtr.size());
    q1.memcpy(kIdxD1, tensor.kIdx.data(), tensor.kIdx.size());
    q1.memcpy(valuesD1, tensor.values.data(), tensor.values.size());
    
    auto matrixD2 = malloc_device<float>(matrix.size(), q2);
    auto fbrPtrD2 = malloc_device<int>(tensor.fbrPtr.size(), q2), kIdxD2 = malloc_device<int>(tensor.kIdx.size(), q2);
    auto valuesD2 = malloc_device<float>(tensor.values.size(), q2), outputD2 = malloc_device<float>(output.size(), q2);
    q2.memcpy(matrixD2, matrix.data(), matrix.size());
    q2.memcpy(fbrPtrD2, tensor.fbrPtr.data(), tensor.fbrPtr.size());
    q2.memcpy(kIdxD2, tensor.kIdx.data(), tensor.kIdx.size());
    q2.memcpy(valuesD2, tensor.values.data(), tensor.values.size());

    q1.wait();
    q2.wait();
    
    q1.submit([&](handler &h){
        h.parallel_for(deviceNum_items, [=](nd_item<2> index) {
            int fbr = index.get_global_id(0), col = index.get_local_id(1);
            float acc = 0.0;
            for (auto element = fbrPtrD1[fbr]; element < fbrPtrD1[fbr+1]; ++element) {
                auto k = kIdxD1[element];
                acc += valuesD1[element] * matrixD1[k * MATRIX_DIM + col];
            }
            outputD1[fbr * MATRIX_DIM + col] = acc;
        });
    });
    
    q2.submit([&](handler &h){
        h.parallel_for(hostNum_items, [=](nd_item<2> index) {
            int fbr = 1 * workload + index.get_global_id(0), col = index.get_local_id(1);
            float acc = 0.0;
            for (auto element = fbrPtrD2[fbr]; element < fbrPtrD2[fbr+1]; ++element) {
                auto k = kIdxD2[element];
                acc += valuesD2[element] * matrixD2[k * MATRIX_DIM + col];
            }
            outputD2[fbr * MATRIX_DIM + col] = acc;
        });
    });
    
    q1.wait();
    q2.wait();
    
    q1.memcpy(output.data(), outputD1, workload * MATRIX_DIM);
    q2.memcpy(output.data() + workload, outputD2, 4 * workload * MATRIX_DIM);
    
    q1.wait();
    q2.wait();
    
    free(matrixD1, q1);
    free(fbrPtrD1, q1);
    free(kIdxD1, q1);
    free(valuesD1, q1);
    free(outputD1, q1);
    
    free(matrixD2, q2);
    free(fbrPtrD2, q2);
    free(kIdxD2, q2);
    free(valuesD2, q2);
    free(outputD2, q2);
    
    auto endKernel = std::chrono::steady_clock::now();
    return (std::chrono::duration_cast<std::chrono::microseconds>(endKernel - startKernel).count());
}

int main (void) {
    device d1;
    try {
        d1 = device(gpu_selector());
    } catch (exception const &e) {
        std::cout << "Cannot select a GPU\n" << e.what() << std::endl;
        return 0;
    }
    queue q1(d1);
    
    device d2;
    try {
        d2 = device(host_selector());
    } catch (exception const &e) {
        std::cout << "Cannot select a CPU\n" << e.what() << std::endl;
        return 0;
    }
    queue q2(d2);
    
    std::cout << "Number of columns: " << MATRIX_DIM << std::endl;
    
    std::cout << "Device 1: " << d1.get_info<sycl::info::device::name>() << std::endl;
    int eu1 = d1.get_info<sycl::info::device::max_compute_units>();
    int freq1 = d1.get_info<sycl::info::device::max_clock_frequency>();
    int perf1 = eu1 * freq1;
    std::cout << "EUs: " << eu1 << " f_max: " << freq1 << " Perf: " << perf1 << std::endl;
    
    std::cout << "Device 2: " << d2.get_info<sycl::info::device::name>() << std::endl;
    int eu2 = d2.get_info<sycl::info::device::max_compute_units>();
    int freq2 = d2.get_info<sycl::info::device::max_clock_frequency>();
    int perf2 = eu2 * freq2;
    std::cout << "EUs: " << eu2 << " f_max: " << freq2 << " Perf: " << perf2 << std::endl;
    
    float workload = (float) perf1 / (perf1 + perf2);
    std::cout << workload << std::endl;

    // Load tensor
    CSFtensor tensor;
    tensor.kIdx.resize(NNZ);
    tensor.values.resize(NNZ);
    
    std::ifstream tensorFile(tensorName);
    int i, j;
    for (auto k = 0; k < NNZ; ++k) {
        tensorFile >> i >> j;
        if (tensor.slcIdx.empty() || tensor.slcIdx.back() != i) {
            tensor.slcIdx.push_back(i);
            tensor.slcPtr.push_back(tensor.fbrIdx.size());
            tensor.fbrIdx.push_back(j);
            tensor.fbrPtr.push_back(k);
        } else if (tensor.fbrIdx.back() != j) {
            tensor.fbrIdx.push_back(j);
            tensor.fbrPtr.push_back(k);
        }
        tensorFile >> tensor.kIdx[k] >> tensor.values[k];
    }
    tensor.slcPtr.push_back(tensor.fbrIdx.size());
    tensor.fbrPtr.push_back(tensor.kIdx.size());
    tensorFile.close();

    std::cout << "Number of fibers: " << tensor.fbrIdx.size() << std::endl;
    
    floatVector matrix(DIM2 * MATRIX_DIM, 0.23), output(tensor.fbrIdx.size() * MATRIX_DIM);
    
    auto discardSampleTime = ttmKernel(q1, q2, tensor, matrix, output);
    
    double execution_time = 0.0;
    for (int iteration = 0; iteration < 30; ++iteration) {
        execution_time += ttmKernel(q1, q2, tensor, matrix, output);
    }
    double average_time = execution_time/30;
    std::cout << average_time << std::endl;
    
    return 0;
}
