NVCC = nvcc
FLAGS = -O2 -Iinclude -I$(CUDNN_PATH)/include -L$(CUDNN_PATH)/lib -lcudnn -lcublas -Xlinker -rpath=$(CUDNN_PATH)/lib

# Παλιά examples (single-file .cu)
%: %.cu
	$(NVCC) $(FLAGS) -o $@ $

# Mini-VGG main target
minivgg: minivgg.cu src/data_loader.cpp src/cuda_kernels.cu
	$(NVCC) $(FLAGS) -o minivgg minivgg.cu src/data_loader.cpp src/cuda_kernels.cu

clean:
	rm -f minivgg main data_loader