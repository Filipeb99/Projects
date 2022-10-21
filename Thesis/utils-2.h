#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "dpc_common.hpp"

#ifdef USE_USM
    typedef std::vector<int, cl::sycl::usm_allocator<int, cl::sycl::usm::alloc::shared> > vec;
#else
    typedef std::vector<int> vec;
#endif

typedef struct tensorInfo {
    std::string name;
    size_t NNZ;
    size_t DIM0;
    size_t DIM1;
    size_t DIM2;
} tensorInfo;

tensorInfo getTensorInfo (char *name);

void read3Dtensor (tensorInfo t, vec &slcIdx, vec &slcPtr, vec &fbrIdx, vec &fbrPtr, int *kIdx, float *values);

void printDeviceInfo (cl::sycl::device &d);

float* matrixAlloc (size_t size, cl::sycl::queue &q);
