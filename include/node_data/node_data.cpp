#include "node_data.h"

#if defined(_WIN32)
#include <windows.h>
#elif defined(__unix__) || defined(__unix) || defined(unix)
#include <sys/sysinfo.h>
#elif defined(__APPLE__) && defined(__MACH__)

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// main function for initialize library
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void init_library(NodeData &_node_data)
{
	_node_data.collect_host_properties();
	_node_data.collect_devices_properties();
    
    _node_data.set_device_properties();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// public interface
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

NodeData::NodeData()
{
    devices_count = 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

NodeData::~NodeData()
{
    
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeData::print_node_properties()
{
	cout << "--------------------------- Init node result: -------------------------------" << endl << endl;

	cout << "Total host memory size: " << host_total_memory / (1024 * 1024 * 1024) << " GB" << endl;
	cout << "Free host memory size: " << host_free_memory / (1024 * 1024 * 1024) << " GB" << endl;

	cout << endl;
	cout << "There are " << devices_count << " devices on current node:" << endl;

	#ifdef __USE_GPU__
	for (int device_id = 0; device_id < devices_count; device_id++)
	{
		cout << device_properties[device_id].name << endl;
		cout << (double)device_free_memory[device_id] / (double)(1024 * 1024) << " Mb of global memory free " << endl;
		cout << (double)device_total_memory[device_id] / (double)(1024 * 1024) << " Mb of global memory total " << endl;
	}
	cout << endl;

	#endif

	cout << "-----------------------------------------------------------------------------" << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
dim3 NodeData::get_max_grid_size(int _device_id)
{

	return dim3(device_properties[_device_id].maxGridSize[0],
		        device_properties[_device_id].maxGridSize[1],
		        device_properties[_device_id].maxGridSize[2]);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeData::update_device_memory_data()
{
	#ifdef __USE_GPU__
	for (int device_id = 0; device_id < devices_count; device_id++)
	{
		cudaSetDevice(device_id);
		cudaMemGetInfo(&device_free_memory[device_id], &device_total_memory[device_id]);
	}
	#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// private functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeData::collect_host_properties()
{
	// collect RAM info

#if defined(_WIN32)
	MEMORYSTATUSEX status;
	status.dwLength = sizeof(status);
	GlobalMemoryStatusEx(&status);
	host_total_memory = (size_t)status.ullTotalPhys;
	host_free_memory = (size_t)status.ullTotalPhys - (size_t)status.dwMemoryLoad;

#elif defined(__unix__) || defined(__unix) || defined(unix)
	struct sysinfo info;
	sysinfo(&info);
	host_total_memory = info.totalram;
	host_free_memory = info.freeram;
    
#elif defined(__APPLE__) && defined(__MACH__)
    cout << "MAC " << endl;
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeData::collect_devices_properties()
{
	#ifdef __USE_GPU__
	// get devices count
	cudaGetDeviceCount(&devices_count);
    cudaError_t cuerr = cudaGetLastError();
    if(cuerr != cudaSuccess)
    {
        cout << "No CUDA devices installed in the current system." << endl;
        devices_count = 0;
        return;
    }

	for (int device_id = 0; device_id < devices_count; device_id++)
	{
		// set current device
		SAFE_CALL(cudaSetDevice(device_id));

		// get common proprties
		cudaDeviceProp tmp_property;
		SAFE_CALL(cudaGetDeviceProperties(&tmp_property, device_id));
		device_properties.push_back(tmp_property);

		// get memory data
		device_free_memory.push_back(0);
		device_total_memory.push_back(0);
		SAFE_CALL(cudaMemGetInfo(&device_free_memory[device_id], &device_total_memory[device_id]));
	}
	#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeData::set_device_properties()
{
    #ifdef __USE_GPU__
        //SAFE_CALL(cudaFuncCachePreferL1());
    #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
