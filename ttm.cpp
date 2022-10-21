#include "utils.h"
#define MATRIX_DIM 64

void ttm (cl::sycl::queue &q, const tensorInfo &tensor, const float *matrix, float *output);

int main (int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: ./ttm tensorName" << std::endl;
        exit(1);
    }
    
    cl::sycl::device d;
    try {
        #ifdef USE_CPU
            d = cl::sycl::device(cl::sycl::cpu_selector());
        #else
            d = cl::sycl::device(cl::sycl::gpu_selector());
        #endif
    } catch (cl::sycl::exception const &e) {
        std::cout << "Cannot select device\n" << e.what() << std::endl;
        return 0;
    }
    cl::sycl::queue q(d);
    printDeviceInfo(d);
    
    cl::sycl::usm_allocator<int, cl::sycl::usm::alloc::shared> myAlloc(q);
    vec slcIdx(0, myAlloc), slcPtr(0, myAlloc), fbrIdx(0, myAlloc), fbrPtr(0, myAlloc);
    
    tensorInfo tensor = getTensorInfo(argv[1]);
    tensor.kIdx = cl::sycl::malloc_shared<int>(tensor.nnz, q);
    tensor.values = cl::sycl::malloc_shared<float>(tensor.nnz, q);
    
    read3Dtensor(tensor, slcIdx, slcPtr, fbrIdx, fbrPtr, tensor.kIdx, tensor.values);
    tensor.fbrCnt = fbrIdx.size();
    tensor.fbrPtr = fbrPtr.data();
    
    auto matrix = matrixAlloc(tensor.dim2 * MATRIX_DIM, q);
    auto output = cl::sycl::malloc_shared<float>(fbrIdx.size() * MATRIX_DIM, q);
    std::cout << "Number of fibers: " << fbrIdx.size() << '\n' << "Number of columns: " << MATRIX_DIM << std::endl;
    
    // warm up device with one execution
    ttm(q, tensor, matrix, output);
    
    for (int iteration = 0; iteration < 5; ++iteration) ttm(q, tensor, matrix, output);
    
    std::cout << '\n' << "Verify output!!!" << std::endl;
    for (int alpha = 0; alpha < 3; alpha++) std::cout << output[64 * alpha] << std::endl;
    
    cl::sycl::free(tensor.kIdx, q);
    cl::sycl::free(tensor.values, q);
    cl::sycl::free(matrix, q);
    cl::sycl::free(output, q);
    
    return 0;
}

void ttm (cl::sycl::queue &q, const tensorInfo &tensor, const float *matrix, float *output) {
    auto start = std::chrono::steady_clock::now();
    
    cl::sycl::range<2> globalSize(tensor.fbrCnt, MATRIX_DIM);
    cl::sycl::range<2> localSize(1, MATRIX_DIM);
    cl::sycl::nd_range<2> num_items(globalSize, localSize);
    
    auto fbrPtr = tensor.fbrPtr;
    auto kIdx   = tensor.kIdx;
    auto values = tensor.values;
    
    q.submit([&](cl::sycl::handler &h){
        h.parallel_for(num_items, [=](cl::sycl::nd_item<2> index) {
            int fbr = index.get_global_id(0), col = index.get_local_id(1);
            float acc = 0.0;
            for (auto element = fbrPtr[fbr]; element < fbrPtr[fbr+1]; ++element) {
                auto k = kIdx[element];
                acc += values[element] * matrix[k * MATRIX_DIM + col];
            }
            output[fbr * MATRIX_DIM + col] = acc;
        });
    });
    
    q.wait();
    
    auto end = std::chrono::steady_clock::now();
    std::cout << (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << std::endl;
    return;
}