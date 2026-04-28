#include "common.h"
#include <vector>

void print_shape(const std::string& label, int n, int c, int h, int w){
    std::cout << "  " << label << ": [" << n << ", " << c
              << ", " << h << ", " << w << "]" << std::endl;
}

void print_gpu_tensor(const std::string& name, float* d_data, int count){
    std::vector<float> h_data(count);
    CHECK_CUDA(cudaMemcpy(h_data.data(), d_data, count * sizeof(float), 
                            cudaMemcpyDeviceToHost));
    std::cout << "  " << name << ": [";
    int show = (count < 6) ? count : 6;
    for(int i = 0; i < show; i++){
        std::cout << h_data[i];
        if(i < show - 1) std::cout << ", ";
    }
    if(count > 6) std::cout << ",...";
    std::cout << "] (" << count << " total)" << std::endl;
}