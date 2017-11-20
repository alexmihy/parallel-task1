#include "include/common/cuda_error_hadling.h"

#include <cuda_runtime_api.h>

#include <cfloat>
#include <iostream>
#include <fstream>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// structure to perform union find on device
template<class _T>
class DeviceUnionFind
{
private:
	_T *parents, *ranks;

	inline __device__ void compress(int _x)
	{
		int y = parents[_x];
		if (y != _x)
		{
			int z = parents[y];
			if (parents[y] != y)
			{
				do
				{
					y = parents[y];
				} while (y != parents[y]);
				parents[_x] = y;
			}
		}
	}
public:
	DeviceUnionFind(int _size)
	{
		SAFE_CALL(cudaMalloc(&parents, sizeof(_T) * _size));
		SAFE_CALL(cudaMalloc(&ranks, sizeof(_T) * _size));
	}

	void free_memory()
	{
		SAFE_CALL(cudaFree(parents));
		SAFE_CALL(cudaFree(ranks));
	}

    // clean component
	inline __device__ void clear(int _x)
	{
		parents[_x] = _x;
		ranks[_x] = 0;
	}

    // return parent vertex to selected one
	inline __device__ int get_parent(int _x)
	{
		return parents[_x];
	}

    // optimize component
	inline __device__ int flatten(int _p)
	{
		if (parents[_p] != _p)
		{
			compress(_p);
		}
	}

    // find parent component
	inline __device__ int find_fast(int _x)
	{
		int root = _x;
		while (root != parents[root])
			root = parents[root];
		while (_x != root)
		{
			int newp = parents[_x];
			parents[_x] = root;
			_x = newp;
		}
		return root;
	}

    // merge two components
	inline __device__ bool merge(int _x, int _y)
	{
		_x = parents[_x];
		_y = parents[_y];
		while (_x != _y)
		{
			if (_y < _x)
			{
				int t = _x;
				_x = _y;
				_y = t;
			}
			int z = atomicCAS(&parents[_y], _y, _x);
			if (z == _y)
			{
				return true;
			}
			_x = parents[parents[_x]];
			_y = parents[parents[z]]; // reuse value returned by atomicCAS
		}
		return false;
	}
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init_mst_data(bool *_in_mst, int _edges_count)
{
	register const int idx = (blockIdx.x * blockDim.x + threadIdx.x) + blockIdx.y * blockDim.x * gridDim.x;
	if (idx < _edges_count)
	{
		_in_mst[idx] = false;
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void init_components_data(int _vertices_count, DeviceUnionFind<int> _components)
{
	register const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < _vertices_count)
	{
		_components.clear(idx);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void unset_cheapest_and_flatten(int *_cheapest, int _vertices_count, DeviceUnionFind<int> _components)
{
	register const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < _vertices_count)
	{
		_cheapest[idx] = -1;
		_components.flatten(idx);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// main computational kernel, requires the most part of execution time
template <typename _TEdgeWeight>
__global__ void find_minimum_edges_atomic(int *_src_ids, // source vertices ids
                                          int *_dst_ids, // destination vertices ids
                                          _TEdgeWeight *_weights, // weights
                                          int _edges_count,
                                          int *_cheapest, // cheapest indexes array
	                                      int _vertices_count,
                                          DeviceUnionFind<int> _components)
{
	register const int idx = (blockIdx.x * blockDim.x + threadIdx.x) + blockIdx.y * blockDim.x * gridDim.x; // create thread per edge
    
	if (idx < _edges_count)
	{
		register int set1 = _components.get_parent(_src_ids[idx]); // get parents for both incedent edges
		register int set2 = _components.get_parent(_dst_ids[idx]);

		if (set1 != set2) // if they belong to differnt components
		{
			register int cheapest_index = _cheapest[set1];
			while (cheapest_index == -1 || _weights[idx] < _weights[cheapest_index]) // atomic update minimum index for src_id vertex
			{
				if (atomicCAS(&_cheapest[set1], cheapest_index, idx) == cheapest_index) {
					break;
				}
				cheapest_index = _cheapest[set1];
			}

			cheapest_index = _cheapest[set2];
			while (cheapest_index == -1 || _weights[idx] < _weights[cheapest_index]) // atomic update minimum index for dst_id vertex
			{
				if (atomicCAS(&_cheapest[set2], cheapest_index, idx) == cheapest_index) {
					break;
				}
				cheapest_index = _cheapest[set2];
			}
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void merge_components(int *_src_ids, // source vertices ids
                                 int *_dst_ids, // destination vertices ids
                                 int *_cheapest, // cheapest indexes array
                                 int _vertices_count,
                                 bool *_in_mst, // result array
                                 int *_trees_num,
							     DeviceUnionFind<int> _components)
{
	register const int idx = blockIdx.x * blockDim.x + threadIdx.x; // for all vertices
	if (idx < _vertices_count)
	{
		if (_cheapest[idx] != -1)
		{
            // get parents 
			int set1 = _components.get_parent(_src_ids[_cheapest[idx]]);
			int set2 = _components.get_parent(_dst_ids[_cheapest[idx]]);

			if (set1 != set2)
			{
				if (_components.merge(set1, set2)) // try to merge using best edge
				{  
					_in_mst[_cheapest[idx]] = true;
				}
				else 
				{
					atomicAdd(_trees_num, 1); // unsuccessful merge => increase active fragment count
				}
				atomicSub(_trees_num, 1);
			}
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
void boruvka_wrapper(int *_src_ids, // source vertices ids
                     int *_dst_ids, // destination vertices ids
                     _TEdgeWeight *_weights, // weights
                     bool *_in_mst, // result array
                     int _vertices_count,
                     int _edges_count)
{
    // create grid threads
	dim3 threads(1024, 1, 1);
	dim3 grid_edges((_edges_count - 1) / threads.x + 1, 1, 1);
	dim3 grid_vertices((_vertices_count - 1) / threads.x + 1, 1, 1);
    
    #ifdef __USE_FERMI__
    if(grid_edges.x > 65535)
    {
        grid_edges.y = (grid_edges.x - 1) / 65535 + 1;
        grid_edges.x = 65535;
    }
    #endif

	DeviceUnionFind<int> components(_vertices_count);

	int *cheapest;
	SAFE_CALL(cudaMalloc(&cheapest, sizeof(int) * _vertices_count));

    // init distances result array and components data
	SAFE_KERNEL_CALL(( init_mst_data <<< grid_edges, threads >>> (_in_mst, _edges_count) ));
	SAFE_KERNEL_CALL(( init_components_data <<< grid_vertices, threads >>> (_vertices_count, components) ));
	
	int *device_num_trees;
	SAFE_CALL(cudaMalloc(&device_num_trees, sizeof(int)));
	int host_num_trees = _vertices_count, prev_num_trees = 0;
	SAFE_CALL(cudaMemcpy(device_num_trees, &host_num_trees, sizeof(int), cudaMemcpyHostToDevice));

	while (host_num_trees != prev_num_trees) // update graph while number of trees changes
	{
        // update components for all vertices
		SAFE_KERNEL_CALL(( unset_cheapest_and_flatten <<< grid_vertices, threads >>> (cheapest,  _vertices_count, components) ));

        // find cheapest edges to merge componets in the future
		SAFE_KERNEL_CALL(( find_minimum_edges_atomic <<< grid_edges, threads >>> (_src_ids, _dst_ids, _weights, _edges_count, 
		                                                                          cheapest, _vertices_count, components) ));

		prev_num_trees = host_num_trees;

        // merge components with edges, found on previous step
		SAFE_KERNEL_CALL(( merge_components <<< grid_vertices, threads >>> (_src_ids, _dst_ids, cheapest, _vertices_count, _in_mst,
			                                                                device_num_trees, components) ));

		SAFE_CALL(cudaMemcpy(&host_num_trees, device_num_trees, sizeof(int), cudaMemcpyDeviceToHost));
	}

	SAFE_CALL(cudaFree(device_num_trees));
	SAFE_CALL(cudaFree(cheapest));

	components.free_memory();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void boruvka_wrapper<float>(int *_edges_src_ids, int *_edges_dst_ids, float *_edges_weights, bool *_in_mst, int _vertices_count,
	                                 int _edges_count);
template void boruvka_wrapper<double>(int *_edges_src_ids, int *_edges_dst_ids, double *_edges_weights,bool *_in_mst, int _vertices_count, 
	                                  int _edges_count);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
