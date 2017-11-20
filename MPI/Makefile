# Compiler flags...
CXX = mpicxx
CUDA_COMPILER = nvcc

# Platform dependent paths
# Lomonosov
CUDA_DIR = /opt/cuda/cuda-6.5
BOOST_DIR = ~/_scratch/boost_1_59_0
PROJECT_PATH = ~/_scratch/graph_library

# Mac OS
# CUDA_DIR = /usr/local/cuda/
# BOOST_DIR = ~/_scratch/boost_1_59_0
# PROJECT_PATH = /Users/afanasiev/Documents/graph_library/src/

# Titan
# CUDA_DIR = /usr/local/cuda-7.0
# BOOST_DIR = /home/afanasiev_ilya/boost_1_59_0
# PROJECT_PATH = /home/afanasiev_ilya/graph_library

# Paths to gpu algorithms from current sample
VPATH=./:../algorithm/apsp/gpu/:../algorithm/mst/gpu/:../include/node_data:../algorithm/community_detection/gpu:../algorithm/scc/gpu

# Include paths...
Include_Path = -I $(CUDA_DIR)/include -I $(BOOST_DIR) -I $(PROJECT_PATH)

# Library paths...
Library_Path =  -L $(CUDA_DIR)/lib -L $(CUDA_DIR)/lib64
Libraries = -lcudart -fopenmp

# Compiler flags...
CXXFLAGS = -O3 -w -m64 -fpermissive -std=c++0x 
CUDA_FLAGS = -O3 -w -m64 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35 --compiler-bindir /usr/bin/gcc

.DEFAULT_GOAL := all

all: mst_MPI

mst_MPI: create_folders mst_MPI.o node_data.o device_boruvka.o
	$(CXX) object_files/mst_MPI.o object_files/node_data.o object_files/device_boruvka.o $(Library_Path) $(Libraries) -o ./bin/mst_MPI

# CPP files

mst_MPI.o: mst_MPI.cpp
	$(CXX) $(CXXFLAGS) $(Include_Path) -c mst_MPI.cpp -o object_files/mst_MPI.o -fopenmp

node_data.o: node_data.cpp
	$(CXX) $(CXXFLAGS) $(Include_Path) -c $(PROJECT_PATH)/include/node_data/node_data.cpp -o object_files/node_data.o -fopenmp

# CUDA kernels

device_boruvka.o: device_boruvka.cu
	$(CUDA_COMPILER) $(CUDA_FLAGS) $(Include_Path) -c $(PROJECT_PATH)/algorithm/mst/gpu/device_boruvka.cu -o object_files/device_boruvka.o

create_folders:
	-mkdir -p ./bin
	-mkdir -p ./object_files

clean:
	-rm -f object_files/*.o
	-rmdir object_files
	-rm bin/mst_MPI