#include "common.h"
#include <random>
#include <cmath>

void print_shape(const std::string& label, int n, int c, int h, int w){
    std::cout << "  " << label << ": [" << n << ", " << c
              << ", " << h << ", " << w << "]" << std::endl;
}

void print_gpu_tensor(const std::string& name, float* d_data, int count){
    std::vector<float> h_data(count);
    CHECK_CUDA(cudaMemcpy(h_data.data(), d_data, count * sizeof(float), 
                            cudaMemcpyDeviceToHost));
    std::cout << "  " << name << ": [";
    int show = (count < 10) ? count : 10;
    for(int i = 0; i < show; i++){
        std::cout << h_data[i];
        if(i < show - 1) std::cout << ", ";
    }
    if(count > 10) std::cout << ",...";
    std::cout << "] (" << count << " total)" << std::endl;
}

// He initialization (Kaiming He et al., 2015): w ~ N(0, sqrt(2 / fan_in))
//random values break the symmetry between neurons, the He formula scale 
//keeps the signal variance constant across the layers. The factor of 2 is used
//for the ReLU which disgards the negative half of each layers output,
//fan_in = how many input values are summed in order for the one output value to be produced
//FC Layer = in_features
//Conv Layer = in_channels * kernel_h * kernel_w
void he_init(std::vector<float>& weights, int fan_in, unsigned seed){
    std::mt19937 rng(seed);
    const float stddev = std::sqrt(2.0f / fan_in);
    std::normal_distribution<float> gaussian(0.0f, stddev);

    for(size_t  idx = 0; idx < weights.size(); idx++){
        weights[idx] = gaussian(rng);
    }
}

