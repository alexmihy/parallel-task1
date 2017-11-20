#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <limits>

#include "include/architectures.h"

#ifdef __USE_GPU__
#include <cuda_runtime_api.h>
#include "include/common/cuda_error_hadling.h"
#endif

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//! \class  NodeData node_data.h "include/node_data/node_data.h"
//! \brief  Collects and stores information about CPU and all devices(Xeon PHI and NVIDIA GPUs), installed on current node
//! \author Ilya Afanasyev
//!
//! \details Uses openMP API and current OS API to get information about host, and cuda_runtine_api to get information about device. \n
//!          Xeon PHI not supported yet.
class NodeData
{
private:
	size_t host_total_memory, host_free_memory;

	int devices_count;                        //!< number of devices, located on current node

	vector<size_t> device_free_memory;           //!< stores information about amount of free global memory on each device
	vector<size_t> device_total_memory;          //!< stores information about amount of total global memory on each device

	#ifdef __USE_GPU__
	vector<cudaDeviceProp> device_properties; //!< store all information about all available devices
	#endif

	//! \brief collects data about host
	void collect_host_properties();

	//! \brief collects device count and device properties
	void collect_devices_properties();
    
    //! \brief set device properties
    void set_device_properties();
public:
	NodeData();
    ~NodeData();

	//! \brief prints information about devices, installed on current node
	void print_node_properties();

	//! \brief returns size of total host RAM
	size_t get_host_total_memory() { return host_total_memory; };

	//! \brief returns size of free host RAM
	size_t get_host_free_memory() { return host_free_memory; };

	//! \brief returns count of gpu devices, set on current node
	int get_devices_count() { return devices_count; };

	//! \brief returns amount of free global memory
	size_t get_device_free_memory(int _device_id) { return device_free_memory[_device_id]; };

	//! \brief returns amount of total global memory
	size_t get_device_total_memory(int _device_id) { return device_total_memory[_device_id]; };

	//! \brief updates data about free amount of data on host
	void update_host_memory_data() { collect_host_properties(); };

	//! \brief updates data about free amount of data on devices
	void update_device_memory_data();

	#ifdef __USE_GPU__
	//! \brief returns maximum number of threads in single block on device with specified ID
	int get_max_threads_per_block(int _device_id) { return device_properties[_device_id].maxThreadsPerBlock; };

	//! \brief returns maximum size of grid
	dim3 get_max_grid_size(int _device_id);
	#endif

	friend void init_library(NodeData &_system_data);

	//! \brief test function for GPU, have to be deleted in the future
	void set_test_free_memory(int _device_id, size_t _mem_size) { device_free_memory[_device_id] = _mem_size; };
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void init_library(NodeData &_system_data);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
