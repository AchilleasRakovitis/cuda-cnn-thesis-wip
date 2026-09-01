#include "benchmark.h"
#include <vector>
#include <algorithm>
#include <fstream>
#include <cstdio>

const char* conv_impl_name(ConvImpl impl){
    switch (impl)
    {
    case CONV_CUDNN:
        return "cuDNN";
    case CONV_NAIVE:
        return "naive";
    default:
        return "unknown";
    }
}

static void run_conv_once(cudnnHandle_t cudnn, convLayer& layer, ConvImpl impl,
                            float* d_input, void* d_workspace, float* d_out,
                            const ConvDims& d){
    
    const float alpha = 1.0f;
    const float beta_overwrite = 0.0f;

    switch (impl)
    {
    case CONV_CUDNN:
        CHECK_CUDNN(cudnnConvolutionForward(
            cudnn,
            &alpha,
            layer.input_desc,
            d_input,
            layer.filter_desc,
            layer.d_filter,
            layer.conv_desc,
            layer.algo,
            d_workspace,
            layer.workspace_bytes,
            &beta_overwrite,
            layer.output_desc,
            d_out
        ));
        break;
    case CONV_NAIVE:
        launch_conv_naive(d_input, layer.d_filter, d_out, d);
        break;
    default:
        break;
    }
}

BenchResult bench_conv(cudnnHandle_t cudnn, convLayer& layer, int layer_id,
                        ConvImpl impl, float* d_input, void* d_workspace,
                        float* d_out, int warmup, int iters){
    
    ConvDims d;
    d.N = layer.in_n;
    d.C = layer.in_c;
    d.H = layer.in_h;
    d.W = layer.in_w;
    d.K = layer.out_c;
    d.R = layer.kernel_size;
    d.S = layer.kernel_size;
    d.pad = layer.kernel_size / 2;
    
    //events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    //warmup with no benchmarking
    for(int i = 0; i < warmup; i++){
        run_conv_once(cudnn, layer, impl, d_input, d_workspace, d_out, d);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    //Timed benchmarking after warmup
    std::vector<float> times_ms;
    times_ms.reserve(iters);

    for(int i = 0; i < iters; i++){
        CHECK_CUDA(cudaEventRecord(start));
        run_conv_once(cudnn, layer, impl, d_input, d_workspace, d_out, d);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        times_ms.push_back(ms);
    }

    //Median and min calculation
    std::sort(times_ms.begin(), times_ms.end());
    double ms_min = times_ms.front();
    double ms_median = times_ms[iters / 2];
    
    //Flops and rates
    double flops = 2.0 * d.N * d.K * d.H * d.W * d.C* d.R * d.S;
    double gflops = flops / (ms_median / 1000.0) / 1e9;
    double pct = gflops / 16300.0 * 100.0; //TITAN RTX fp32 peak

    //Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    //Save the results to the benchmark struct
    BenchResult b;
    b.impl = impl;
    b.layer_id = layer_id;
    b.ms_median = ms_median;
    b.ms_min = ms_min;
    b.gflops = gflops;
    b.pct_peak = pct;
    return b;
}

void run_all_benchmarks(cudnnHandle_t cudnn, convLayer& layer1, convLayer& layer2, convLayer& layer3,
                        float* d_input, void* d_workspace){

    const int warmup = 20;
    const int iters = 100;

    //The input of each layer is the same chained logic, the output of the previous
    //is the input of the next. Forward pass must have ran first
    convLayer* layers[3] = {&layer1, &layer2, &layer3};
    float* inputs[3] = {d_input, layer1.d_pool_out, layer2.d_pool_out};

    int max_elements = layer1.out_n* layer1.out_c * layer1.out_h * layer1.out_w;
    int e2 = layer2.out_n * layer2.out_c * layer2.out_h * layer2.out_w;
    int e3 = layer3.out_n * layer3.out_c * layer3.out_h * layer3.out_w;
    if(e2 > max_elements) max_elements = e2;
    if(e3 > max_elements) max_elements = e3;

    float* d_out;
    CHECK_CUDA(cudaMalloc(&d_out, max_elements * sizeof(float)));

    //Save the findings into CSV 
    std::ofstream csv("benchmark_results.csv");
    csv << "layer,impl,ms_median,ms_min,gflops,pct_peak\n";
    printf("\n=== CONV BENCHMARK (warmup=%d, iters=%d) ===\n", warmup, iters);
    printf("%-6s %-8s %12s %10s %10s %8s\n",
           "layer", "impl", "ms_median", "ms_min", "GFLOP/s", "%peak");

    for(int L = 0; L < 3; L++){
        for(int impl = 0; impl < CONV_IMPL_COUNT; impl++){
            BenchResult b = bench_conv(cudnn, *layers[L], L+1, (ConvImpl)impl, inputs[L], d_workspace, d_out, warmup, iters);

            printf("%-6d %-8s %12.4f %10.4f %10.1f %8.2f\n",
                    b.layer_id, conv_impl_name(b.impl), b.ms_median,
                    b.ms_min, b.gflops, b.pct_peak);

            csv << b.layer_id << ","
                << conv_impl_name(b.impl) << ","
                << b.ms_median << ","
                << b.ms_min << ","
                << b.gflops << ","
                << b.pct_peak << "\n";
        }
    }

    CHECK_CUDA(cudaFree(d_out));
    printf("\nCSV written to benchmark_results.csv\n");
}
