NVCC = nvcc
FLAGS = -O2 -I$(CUDNN_PATH)/include -L$(CUDNN_PATH)/lib -lcudnn -Xlinker -rpath=$(CUDNN_PATH)/lib

# Παλιά examples (single-file .cu)
%: %.cu
	$(NVCC) $(FLAGS) -o $@ $

# Το νέο Mini-VGG που χρειάζεται και data_loader.cpp
minivgg: minivgg.cu data_loader.cpp
	$(NVCC) $(FLAGS) -o minivgg minivgg.cu data_loader.cpp

clean:
	rm -f 01_hello_cudnn example1 example2 example3 example4 example5 example6 minivgg main data_loader