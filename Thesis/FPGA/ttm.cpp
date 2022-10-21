#include "utils.h"
#define MATRIX_DIM 8
#define SHIFT_REG 16
tensorInfo t;

void ttm (cl::sycl::queue &q, const std::vector<int> fbrPtr, const int *kIdx, const float *values, const float *matrix, float *output);

int main (int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: ./ttm tensorName" << std::endl;
        exit(1);
    } else {
        t = getTensorInfo(argv[1]);
    }
    
    cl::sycl::device d;
    try {
        #ifdef FPGA_EMULATOR
            cl::sycl::ext::intel::fpga_emulator_selector selector;
        #else
            cl::sycl::ext::intel::fpga_selector selector;
        #endif
        d = cl::sycl::device(selector);
    } catch (cl::sycl::exception const &e) {
        std::cout << "Cannot select a FPGA\n" << e.what() << std::endl;
        return 0;
    }
    cl::sycl::queue q(d);
    printDeviceInfo(d);
    
    std::vector<int> slcIdx(0), slcPtr(0), fbrIdx(0), fbrPtr(0);
    int *kIdx = (int *) malloc (t.NNZ * sizeof(int));
    float *values = (float *) malloc (t.NNZ * sizeof(float));
    
    // read tensor from file
    read3Dtensor(t, slcIdx, slcPtr, fbrIdx, fbrPtr, kIdx, values);
    std::cout << "Number of fibers: " << fbrIdx.size() << '\n' << "Number of columns: " << MATRIX_DIM << std::endl;
    
    float *matrix = (float *) malloc (t.DIM2 * MATRIX_DIM * sizeof(float));
    for (auto ele = 0; ele < t.DIM2 * MATRIX_DIM; ++ele) matrix[ele] = 0.23;
    float *output = (float *) malloc (fbrIdx.size() * MATRIX_DIM * sizeof(float));
    
    // warm up device with one execution
    ttm(q, fbrPtr, kIdx, values, matrix, output);
    
    for (int iteration = 0; iteration < 5; ++iteration) ttm(q, fbrPtr, kIdx, values, matrix, output);
    
    free(kIdx);
    free(values);
    free(matrix);
    free(output);
    
    return 0;
}

void ttm (cl::sycl::queue &q, const std::vector<int> fbrPtr, const int *kIdx, const float *values, const float *matrix, float *output) {
    auto start = std::chrono::steady_clock::now();
    
    int fbrCnt = fbrPtr.size() - 1;
    
    cl::sycl::buffer fbrPtrBuffer(fbrPtr);
    cl::sycl::buffer kIdxBuffer(kIdx, cl::sycl::range<1>(t.NNZ));
    cl::sycl::buffer valuesBuffer(values, cl::sycl::range<1>(t.NNZ));
    cl::sycl::buffer matrixBuffer(matrix, cl::sycl::range<1>(t.DIM2 * MATRIX_DIM));
    cl::sycl::buffer outputBuffer(output, cl::sycl::range<1>(fbrCnt * MATRIX_DIM));
    
    q.submit([&](cl::sycl::handler &h){
        cl::sycl::accessor accMatrix(matrixBuffer, h, cl::sycl::read_only);
        cl::sycl::accessor accFbrPtr(fbrPtrBuffer, h, cl::sycl::read_only);
        cl::sycl::accessor accKIdx(kIdxBuffer, h, cl::sycl::read_only);
        cl::sycl::accessor accValues(valuesBuffer, h, cl::sycl::read_only);
        cl::sycl::accessor accOutput(outputBuffer, h, cl::sycl::write_only, cl::sycl::no_init);

        h.single_task([=]() [[intel::kernel_args_restrict]] {
            float acc = 0.0f;
            
            // Step 1 : Declare multiple copies of variable acc
            float acc_copies[SHIFT_REG];
            
            for (auto fbr = 0; fbr < fbrCnt; ++fbr) {
                for (auto col = 0; col < MATRIX_DIM; ++col) {
                    acc = 0.0f;
                    
                    // Step 2 : Initialize all copies
                    for (uint8_t i = 0; i < SHIFT_REG; ++i) {
                        acc_copies[i] = 0.0f;
                    }
                    
                    for (auto ele = accFbrPtr[fbr]; ele < accFbrPtr[fbr+1]; ++ele) {
                        int k = accKIdx[ele];
                        
                        // Step 3 : Perform operation on the last copy
                        float cur = acc_copies[SHIFT_REG - 1] + (accValues[ele] * accMatrix[k * MATRIX_DIM + col]);
                        
                        // Step 4a : Shift copies
                        #pragma unroll
                        for (uint8_t j = SHIFT_REG - 1; j > 0; --j) {
                            acc_copies[j] = acc_copies[j-1];
                        }
                        
                        // Step 4b : Insert updated copy at the beginning
                        acc_copies[0] = cur;
                    }
                    
                    // Step 5 : Perform reduction on copies
                    #pragma unroll
                    for (uint8_t i = 0; i < SHIFT_REG; ++i) {
                        acc += acc_copies[i];
                    }
                    
                    accOutput[fbr * MATRIX_DIM + col] = acc;
                }
            }
        });
    });
    q.wait();
    
    auto end = std::chrono::steady_clock::now();
    std::cout << (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << std::endl;
    return;
}
