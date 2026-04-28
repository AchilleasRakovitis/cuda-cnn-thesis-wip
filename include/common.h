#pragma once

#include <iostream>
#include <string.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK_CUDNN(call) \
    do{ \
        cudnnStatus_t status = (call); \
        if(status != CUDNN_STATUS_SUCCESS){ \
            std::cerr << "cuDNN error at " << __FILE__ << ":" \
                      << __LINE__ << ": " \
                      << cudnnGetErrorString(status) << std::endl; \
                      std::exit(EXIT_FAILURE); \
        } \
    }while(0)

#define CHECK_CUDA(call) \
    do{ \
        cudaError_t err = (call); \
        if(err != cudaSuccess){ \
            std::cerr << "CUDA error at " << __FILE__ << ":" \
            << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        }   \
    }while(0)

#define CHECK_CUBLAS(call) \
    do{ \
        cublasStatus_t status = (call); \
        if(status != CUBLAS_STATUS_SUCCESS){ \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" \
                      << __LINE__ << ": " << status << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    }while(0)

void print_shape(const std::string& label, int n, int c, int h, int w);
void print_gpu_tensor(const std::string& name, float* d_data, int count);