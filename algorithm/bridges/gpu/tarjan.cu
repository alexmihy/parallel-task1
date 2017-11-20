#include "include/common/cuda_error_hadling.h"

#include "include/common_data_structures/containers.h"

#include <cuda_runtime_api.h>

#include <cfloat>
#include <iostream>
#include <fstream>
#include <set>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ bfs_kernel(int *_src_ids, int *_dst_ids, bool *_in_trees, long long _edges_count, int *_bfs_level, 
	                       int _current_level, bool *_terminate)
{
	register const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < _edges_count)
	{
		int src_id = _src_ids[idx];
		int dst_id = _dst_ids[idx];

        if((_bfs_level[src_id] == _current_level) && (_bfs_level[dst_id] == -1))
        {
            _bfs_level[dst_id] = _current_level + 1;
            _in_trees[idx] = true;
            _terminate[0] = false;
        }
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ init_D_kernel(int *_D, int _vertices_count)
{
	register const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < _vertices_count)
	{
		_D[idx] = 1;
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ compute_D_kernel(int *_src_ids, int *_dst_ids, int _edges_count, int *_N, int *_D, int *_bfs_level, int _current_level)
{
	register const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < _edges_count)
	{
		int src_id = _src_ids[idx];
		int dst_id = _dst_ids[idx];

		if ((_N[dst_id] > _N[src_id]) && (_bfs_level[src_id] == _current_level))
		{
			atomicAdd(&_D[src_id], _D[dst_id]);
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ init_L_H_kernel(int *_L, int *_H, int *_N, int _vertices_count)
{
	register const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < _vertices_count)
	{
		_L[idx] = _N[idx];
		_H[idx] = _N[idx];
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ optimize_L_H_using_tree_kernel(int *_src_ids, int *_dst_ids, int _edges_count, int *_L, int *_H, int *_N, 
	                                           int *_bfs_level,  int _level)
{
	register const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < _edges_count)
	{
		int src_id = _src_ids[idx];
		int dst_id = _dst_ids[idx];

		if (_bfs_level[src_id] == _level)
		{
			if (_N[dst_id] > _N[src_id])
			{
				atomicMin(&_L[src_id], _L[dst_id]);
				atomicMax(&_H[src_id], _H[dst_id]);
			}
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ optimize_L_H_using_graph_kernel(int *_src_ids, int *_dst_ids, bool *_in_trees, int _edges_count,
	                                            int *_L, int *_H, int *_N, int *_bfs_level, int _level)
{
	register const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < _edges_count)
	{
		int src_id = _src_ids[idx];
		int dst_id = _dst_ids[idx];

		if (_bfs_level[src_id] == _level)
		{
			if (!_in_trees[idx])
			{
				atomicMin(&_L[src_id], _N[dst_id]);
				atomicMax(&_H[src_id], _N[dst_id]);
			}
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ process_results_kernel(int *_src_ids, int *_dst_ids, bool *_bridges, int _edges_count,
	                                   int *_L, int *_H, int *_D, int *_N)
{
	register const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < _edges_count)
	{
		int src_id = _src_ids[idx];
		int dst_id = _dst_ids[idx];
		if (_N[dst_id] > _N[src_id])
		{
			if ((_L[dst_id] == _N[dst_id]) && (_H[dst_id] < (_N[dst_id] + _D[dst_id])))
			{
				_bridges[idx] = true;
			}
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// wrappers
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void tarjan_bfs_wrapper(int *_device_src_ids, int *_device_dst_ids, bool *_device_in_trees, long long _edges_count,
                        int *_device_bfs_level, int _vertices_count, int &_max_level, int _root)
{
	dim3 threads(1024, 1, 1);
	dim3 grid_edges((_edges_count - 1) / threads.x + 1, 1, 1);

	SAFE_CALL(cudaMemset(_device_in_trees, 0, sizeof(bool) * _edges_count));

    int current_level = 1;
    SAFE_CALL(cudaMemcpy(&_device_bfs_level[_root], &current_level, sizeof(int), cudaMemcpyHostToDevice));
    
    // do parallel bfs
    bool host_terminate = false;
    bool *device_terminate;
    SAFE_CALL(cudaMalloc((void**)&device_terminate, sizeof(bool)));
    do
    {
        host_terminate = true;
        
        SAFE_CALL(cudaMemcpy(device_terminate, &host_terminate, sizeof(bool), cudaMemcpyHostToDevice));
        
        SAFE_KERNEL_CALL((bfs_kernel <<< grid_edges, threads >>> (_device_src_ids, _device_dst_ids,
                                                                  _device_in_trees, _edges_count, _device_bfs_level,
                                                                  current_level, device_terminate)));
        
        SAFE_CALL(cudaMemcpy(&host_terminate, device_terminate, sizeof(bool), cudaMemcpyDeviceToHost));
        
        current_level++;
    } while (host_terminate == false);
    SAFE_CALL(cudaFree(device_terminate));
    
    if(current_level > _max_level)
        _max_level = current_level;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void compute_D_wrapper(int *_device_trees_src_ids, int *_device_trees_dst_ids, int _trees_edges_count,
				       int *_device_D, int *_device_N, int *_device_bfs_level, int _vertices_count, int _max_level)
{
	cout << "D wrapper " << _trees_edges_count << endl;

	dim3 threads(1024, 1, 1);
	dim3 grid_vertices((_vertices_count - 1) / threads.x + 1, 1, 1);
	dim3 grid_edges((_trees_edges_count - 1) / threads.x + 1, 1, 1);

	SAFE_KERNEL_CALL(( init_D_kernel <<< grid_vertices, threads >>> (_device_D, _vertices_count) ));

	for (int level = _max_level; level >= 0; level--)
	{
		SAFE_KERNEL_CALL(( compute_D_kernel <<< grid_edges, threads >>> (_device_trees_src_ids, _device_trees_dst_ids, 
			     _trees_edges_count, _device_N, _device_D, _device_bfs_level, level) ));
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void compute_L_H_wrapper(int *_device_src_ids, int *_device_dst_ids, bool *_device_in_trees, int _edges_count,
						 int *_device_trees_src_ids, int *_device_trees_dst_ids, int _trees_edges_count, 
	                     int *_device_L, int *_device_H, int *_device_N, int *_device_bfs_level, int _vertices_count, int _max_level)
{
	cout << "L H wrapper " << endl;

	dim3 threads(1024, 1, 1);
	dim3 grid_vertices((_vertices_count - 1) / threads.x + 1, 1, 1);
	dim3 grid_edges((_edges_count - 1) / threads.x + 1, 1, 1);
	dim3 grid_trees_edges((_trees_edges_count - 1) / threads.x + 1, 1, 1);

	// init using numbers
	SAFE_KERNEL_CALL((init_L_H_kernel <<< grid_vertices, threads >>> (_device_L, _device_H, _device_N, _vertices_count)));

	// optimize 
	for(int level = _max_level; level >= 1; level--)
    {
		SAFE_KERNEL_CALL((optimize_L_H_using_tree_kernel <<< grid_trees_edges, threads >>> 
			(_device_trees_src_ids, _device_trees_dst_ids, _trees_edges_count, _device_L, _device_H, _device_N,
			_device_bfs_level, level)));

		SAFE_KERNEL_CALL((optimize_L_H_using_graph_kernel <<< grid_edges, threads >>>
			(_device_src_ids, _device_dst_ids, _device_in_trees, _edges_count, _device_L, _device_H, _device_N,
			_device_bfs_level, level)));
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void process_results_wrapper(int *_device_src_ids, int *_device_dst_ids, bool *_device_bridges, int _edges_count,
	                         int *_device_L, int *_device_H, int *_device_D, int *_device_N)
{
	dim3 threads(1024, 1, 1);
	dim3 grid_edges((_edges_count - 1) / threads.x + 1, 1, 1);

	SAFE_KERNEL_CALL((process_results_kernel <<< grid_edges, threads >>> (_device_src_ids, _device_dst_ids, _device_bridges, _edges_count,
	                                                                      _device_L, _device_H, _device_D, _device_N)));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
