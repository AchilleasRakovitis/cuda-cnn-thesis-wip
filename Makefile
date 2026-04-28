NVCC = nvcc
FLAGS = -O2 -Iinclude -I$(CUDNN_PATH)/include -L$(CUDNN_PATH)/lib -lcudnn -lcublas -Xlinker -rpath=$(CUDNN_PATH)/lib

# Παλιά examples (single-file .cu)
%: %.cu
	$(NVCC) $(FLAGS) -o $@ $

# Mini-VGG main target
minivgg: src/main.cu src/data_loader.cpp src/cuda_kernels.cu src/common.cpp src/conv_layer.cu
	$(NVCC) $(FLAGS) -o minivgg src/main.cu src/data_loader.cpp src/cuda_kernels.cu src/common.cpp src/conv_layer.cu

clean:
	rm -f minivgg main data_loader