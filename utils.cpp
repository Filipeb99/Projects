#include "utils.h"
#define MATRIX_VAL 0.23

tensorInfo getTensorInfo (char *name) {
    std::string nameString(name);
    tensorInfo tensor;
    if (nameString.compare("nell-2") == 0) {
        tensor.name = "nell-2.tns";
        tensor.nnz = 76879419;
        tensor.dim0 = 12092;
        tensor.dim1 = 9184;
        tensor.dim2 = 28818;
    } else if (nameString.compare("vast") == 0) {
        std::cout << "Not yet implemented" << std::endl;
        exit(0);
    } else {
        std::cout << "Invalid tensor" << std::endl;
        exit(1);
    }
    return tensor;
}

void read3Dtensor (tensorInfo tensor, vec &slcIdx, vec &slcPtr, vec &fbrIdx, vec &fbrPtr, int *kIdx, float *values) {
    std::ifstream tensorFile(tensor.name);
    int i, j;
    for (auto k = 0; k < tensor.nnz; ++k) {
        tensorFile >> i >> j;
        if (slcIdx.empty() || slcIdx.back() != i) {
            slcIdx.push_back(i);
            slcPtr.push_back(fbrIdx.size());
            fbrIdx.push_back(j);
            fbrPtr.push_back(k);
        } else if (fbrIdx.back() != j) {
            fbrIdx.push_back(j);
            fbrPtr.push_back(k);
        }
        tensorFile >> kIdx[k] >> values[k];
    }
    slcPtr.push_back(fbrIdx.size());
    fbrPtr.push_back(tensor.nnz);
    tensorFile.close();
}

void printDeviceInfo (cl::sycl::device &d) {
    std::cout << "Device: " << d.get_info<cl::sycl::info::device::name>() << std::endl;
    std::cout << "Max compute units: " << d.get_info<cl::sycl::info::device::max_compute_units>() << std::endl;
    std::cout << "Frequency: " << d.get_info<cl::sycl::info::device::max_clock_frequency>() << std::endl;
    std::cout << "Max work_group size: " << d.get_info<cl::sycl::info::device::max_work_group_size>() << std::endl;
    std::cout << "Shared memory size: " << d.get_info<cl::sycl::info::device::local_mem_size>() << std::endl;
}

float* matrixAlloc (size_t size, cl::sycl::queue &q) {
    auto matrix = cl::sycl::malloc_shared<float>(size, q);
    for (auto it = 0; it < size; ++it) matrix[it] = MATRIX_VAL;
    return matrix;
}
