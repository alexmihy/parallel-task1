#include "include/common/cuda_error_hadling.h"
#include "forward_backward.cuh"

#include <iostream>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ init_fb_data_kernel(int *_trees, bool *_active, int *_components, int _vertices_count)
{
	register const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < _vertices_count)
	{
		_trees[idx] = INIT_TREE;
		_active[idx] = true;
		_components[idx] = INIT_COMPONENT;
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ set_degrees(int *_src_ids, int *_dst_ids, int _edges_count, int *_in_deg, int *_out_deg, bool *_active)
{
	register const int idx = (blockIdx.x * blockDim.x + threadIdx.x) + blockIdx.y * blockDim.x * gridDim.x;
	if (idx < _edges_count)
	{
		int src_id = _src_ids[idx];
		int dst_id = _dst_ids[idx];

		if (_active[src_id])
			atomicAdd(&(_out_deg[dst_id]), 1);
		if (_active[dst_id])
			atomicAdd(&(_in_deg[src_id]), 1);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ trim_kernel(int _vertices_count, int *_in_deg, int *_out_deg, bool *_active, int *_trees, int *_components, int *_last_component, bool *_changes)
{
	register const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < _vertices_count)
	{
		if (_active[idx] && ((_in_deg[idx] == 0) || (_out_deg[idx] == 0)))
		{
			int last_component = atomicAdd(&(_last_component[0]), 1);
			
			_active[idx] = false;
			_trees[idx] = INIT_TREE - 1;
			_components[idx] = last_component;
			_changes[0] = true;
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ select_pivot_kernel(int *_trees, int _tree_num, int _vertices_count, int *_pivot)
{
	register const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < _vertices_count)
	{
		if (_trees[idx] == _tree_num)
			_pivot[0] = idx;
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ init_visited(bool *_visited, int *_pivot, int _vertices_count)
{
	register const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < _vertices_count)
	{
		_visited[idx] = false;
	}
	_visited[_pivot[0]] = true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ bfs_kernel(int *_src_ids, int *_dst_ids, long long _edges_count, bool *_visited, bool *_terminate, int *_trees,
                           bool *_active)
{
	register const int idx = (blockIdx.x * blockDim.x + threadIdx.x) + blockIdx.y * blockDim.x * gridDim.x;
	if (idx < _edges_count)
	{
		int src_id = _src_ids[idx];
		int dst_id = _dst_ids[idx];

		if ((_visited[src_id] == true) && (_trees[src_id] == _trees[dst_id]) && (_visited[dst_id] == false))
		{
			_visited[dst_id] = true;
			_terminate[0] = false;
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ process_reach_result(bool *_fwd_result, bool *_bwd_result, int *_components, int *_trees, bool *_active, int _vertices_count,
	                                 int _last_tree, int _last_component)
{
	register const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < _vertices_count)
	{
		if (!_active[idx])
			return;

		int fwd_res = _fwd_result[idx];
		int bwd_res = _bwd_result[idx];

		if ((fwd_res == true) && (bwd_res == true))
		{
			_active[idx] = false;
			_components[idx] = _last_component;
			_trees[idx] = _last_tree;
		}
		else if ((fwd_res == false) && (bwd_res == false))
		{
			_trees[idx] = _last_tree + 1;
		}
		else if ((fwd_res == true) && (bwd_res == false))
		{
			_trees[idx] = _last_tree + 2;
		}
		else if ((fwd_res == false) && (bwd_res == true))
		{
			_trees[idx] = _last_tree + 3;
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void bfs(int *_src_ids, int *_dst_ids, int _vertices_count, long long _edges_count, int *_pivot, bool *_visited, int *_trees, bool *_active)
{
	dim3 threads(1024, 1, 1);
	dim3 grid_vertices((_vertices_count - 1) / threads.x + 1, 1, 1);
	dim3 grid_edges((_edges_count - 1) / threads.x + 1, 1, 1);
    
    #ifdef __USE_FERMI__
    if(grid_edges.x > 65535)
    {
        grid_edges.y = (grid_edges.x - 1) / 65535 + 1;
        grid_edges.x = 65535;
    }
    #endif

	SAFE_KERNEL_CALL(( init_visited <<<grid_vertices, threads>>> (_visited, _pivot, _vertices_count) ));

	bool *device_terminate;
	SAFE_CALL(cudaMalloc((void**)&device_terminate, sizeof(bool)));

	bool host_terminate = false;
	while (host_terminate == false)
	{
		host_terminate = true;
		SAFE_CALL(cudaMemcpy(device_terminate, &host_terminate, sizeof(bool), cudaMemcpyHostToDevice));

		SAFE_KERNEL_CALL((bfs_kernel <<< grid_edges, threads >>> (_src_ids, _dst_ids, _edges_count, _visited, device_terminate, _trees, _active)));

		SAFE_CALL(cudaMemcpy(&host_terminate, device_terminate, sizeof(bool), cudaMemcpyDeviceToHost));
	}

	SAFE_CALL(cudaFree(device_terminate));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// wrappers
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void init_fb_data_wrapper(int *_trees, bool *_active, int *_components, int _vertices_count)
{
	dim3 threads(1024, 1, 1);
	dim3 grid_vertices((_vertices_count - 1) / threads.x + 1, 1, 1);

	SAFE_KERNEL_CALL(( init_fb_data_kernel<<< grid_vertices, threads >>> (_trees, _active, _components, _vertices_count) ));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// trim edges on GPU (eliminate SCC  with size 1)
void trim_wrapper(int *_src_ids, int *_dst_ids, int _vertices_count, long long _edges_count, int *_components,
	              int *_trees, bool *_active, int &_last_component)
{
	int *in_deg;
	int *out_deg;
	SAFE_CALL(cudaMalloc((void**)&in_deg, _vertices_count * sizeof(int)));
	SAFE_CALL(cudaMalloc((void**)&out_deg, _vertices_count * sizeof(int)));

	dim3 threads_edges(1024, 1, 1);
	dim3 grid_edges((_edges_count - 1) / threads_edges.x + 1, 1, 1);
    
    #ifdef __USE_FERMI__
    if(grid_edges.x > 65535)
    {
        grid_edges.y = (grid_edges.x - 1) / 65535 + 1;
        grid_edges.x = 65535;
    }
    #endif

	dim3 threads_vertices(1024, 1, 1);
	dim3 grid_vertices((_vertices_count - 1) / threads_vertices.x + 1, 1, 1);

	int *device_last_component;
	bool *device_changes;
	SAFE_CALL(cudaMalloc((void**)&device_changes, sizeof(bool)));
	SAFE_CALL(cudaMalloc((void**)&device_last_component, sizeof(int)));
	SAFE_CALL(cudaMemcpy(device_last_component, &_last_component, sizeof(int), cudaMemcpyHostToDevice));

	bool host_changes = false;
	do
	{
		// clear data
		host_changes = false;
		SAFE_CALL(cudaMemcpy(device_changes, &host_changes, sizeof(bool), cudaMemcpyHostToDevice));

		SAFE_CALL(cudaMemset(in_deg, 0, _vertices_count * sizeof(int)));
		SAFE_CALL(cudaMemset(out_deg, 0, _vertices_count * sizeof(int)));

		SAFE_KERNEL_CALL(( set_degrees <<< grid_edges, threads_edges >>> (_src_ids, _dst_ids, _edges_count, in_deg, out_deg, _active)) );

		SAFE_KERNEL_CALL(( trim_kernel <<< grid_vertices, threads_vertices >>> (_vertices_count, in_deg, out_deg, _active, _trees, _components, 
			                                                                    device_last_component, device_changes) ));

		SAFE_CALL(cudaMemcpy(&host_changes, device_changes, sizeof(bool), cudaMemcpyDeviceToHost));
	} while (host_changes);

	SAFE_CALL(cudaMemcpy(&_last_component, device_last_component, sizeof(int), cudaMemcpyDeviceToHost));
	SAFE_CALL(cudaFree(in_deg));
	SAFE_CALL(cudaFree(out_deg));
	SAFE_CALL(cudaFree(device_changes));
	SAFE_CALL(cudaFree(device_last_component));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void forward_backward_wrapper(int *_src_ids, int *_dst_ids, int _vertices_count, long long _edges_count, int *_components, int *_trees, int _tree_num,
							  bool *_active, int _last_component)
{
	static int last_component = _last_component;
	static int last_tree = _tree_num;

	int *device_pivot;
	SAFE_CALL(cudaMalloc((void**)&device_pivot, sizeof(int)));

	dim3 threads(1024, 1, 1);
	dim3 grid_vertices((_vertices_count - 1) / threads.x + 1, 1, 1);

	int host_pivot = ERROR_IN_PIVOT;

	SAFE_CALL(cudaMemcpy(device_pivot, &host_pivot, sizeof(int), cudaMemcpyHostToDevice));
	SAFE_KERNEL_CALL(( select_pivot_kernel <<< grid_vertices, threads >>> (_trees, _tree_num, _vertices_count, device_pivot) ));
	SAFE_CALL(cudaMemcpy(&host_pivot, device_pivot, sizeof(int), cudaMemcpyDeviceToHost));

	if (host_pivot == ERROR_IN_PIVOT)
		return;

	bool *fwd_result, *bwd_result;
	SAFE_CALL(cudaMalloc((void**)&fwd_result, _vertices_count * sizeof(bool)));
	SAFE_CALL(cudaMalloc((void**)&bwd_result, _vertices_count * sizeof(bool)));

	bfs(_src_ids, _dst_ids, _vertices_count, _edges_count, device_pivot, fwd_result, _trees, _active);
	bfs(_dst_ids, _src_ids, _vertices_count, _edges_count, device_pivot, bwd_result, _trees, _active);

	SAFE_KERNEL_CALL(( process_reach_result <<< grid_vertices, threads >>> (fwd_result, bwd_result, _components, _trees, _active, _vertices_count, last_tree, last_component) ));
	last_component++;
	last_tree += 4;

	SAFE_CALL(cudaFree(fwd_result));
	SAFE_CALL(cudaFree(bwd_result));
	SAFE_CALL(cudaFree(device_pivot));

	forward_backward_wrapper(_src_ids, _dst_ids, _vertices_count, _edges_count,  _components, _trees, last_tree - 1, _active, _last_component);
	forward_backward_wrapper(_src_ids, _dst_ids, _vertices_count, _edges_count,  _components, _trees, last_tree - 2, _active, _last_component);
	forward_backward_wrapper(_src_ids, _dst_ids, _vertices_count, _edges_count,  _components, _trees, last_tree - 3, _active, _last_component);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
