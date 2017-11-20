#include "include/common/cuda_error_hadling.h"
#include "bfs.cuh"

#include <cuda_runtime_api.h>

#include <iostream>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ level_init_kernel(bool *_visited, int _vertices_count, int _source_vertex)
{
    register const int idx = (blockIdx.x * blockDim.x + threadIdx.x) + blockIdx.y * blockDim.x * gridDim.x;
    if (idx < _vertices_count)
    {
        _visited[idx] = UNVISITED;
        if(idx == _source_vertex)
            _visited[idx] = VISITED;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ bfs_kernel(int *_src_ids, int *_dst_ids, int _edges_count, bool *_visited, bool *_terminate, int _current_level)
{
    register const int idx = (blockIdx.x * blockDim.x + threadIdx.x) + blockIdx.y * blockDim.x * gridDim.x;
    if (idx < _edges_count)
    {
        int src_id = _src_ids[idx];
        int dst_id = _dst_ids[idx];
        
        if((_visited[src_id] == VISITED) && (_visited[dst_id] == UNVISITED))
        {
            _visited[dst_id] = VISITED;
            _terminate[0] = false;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void bfs_wrapper(int *_src_ids, int *_dst_ids, int _vertices_count, long long _edges_count, bool *_visited, int _source_vertex)
{
    dim3 threads(1024, 1, 1);
    dim3 grid_vertices((_vertices_count - 1) / threads.x + 1, 1, 1);
    dim3 grid_edges((_edges_count - 1) / threads.x + 1, 1, 1);

    #ifdef __USE_FERMI__
    if(grid_vertices.x > 65535)
    {
        grid_vertices.y = (grid_vertices.x - 1) / 65535 + 1;
        grid_vertices.x = 65535;
    }
    if(grid_edges.x > 65535)
    {
        grid_edges.y = (grid_edges.x - 1) / 65535 + 1;
        grid_edges.x = 65535;
    }
    #endif
    
    SAFE_KERNEL_CALL((level_init_kernel <<< grid_vertices, threads >>> (_visited, _vertices_count, _source_vertex)));
    
    bool *device_terminate;
    SAFE_CALL(cudaMalloc((void**)&device_terminate, sizeof(bool)));
    
    bool host_terminate = false;
    int current_level = 1;
    while (host_terminate == false)
    {
        host_terminate = true;
        SAFE_CALL(cudaMemcpy(device_terminate, &host_terminate, sizeof(bool), cudaMemcpyHostToDevice));
        
        SAFE_KERNEL_CALL((bfs_kernel <<< grid_edges, threads >>> (_src_ids, _dst_ids, _edges_count, _visited,
                                                                  device_terminate, current_level)));
        
        SAFE_CALL(cudaMemcpy(&host_terminate, device_terminate, sizeof(bool), cudaMemcpyDeviceToHost));
        current_level++;
    }
    SAFE_CALL(cudaFree(device_terminate));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
