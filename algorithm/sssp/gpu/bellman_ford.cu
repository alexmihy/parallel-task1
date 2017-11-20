#include "include/common/cuda_error_hadling.h"
#include "bellman_ford.cuh"

#include <cfloat>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// init distances
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ init_kernel(float *_distances, int _vertices_count, int _source_vertex)
{
    register const int idx = (blockIdx.x * blockDim.x + threadIdx.x) + blockIdx.y * blockDim.x * gridDim.x;
    
    // not source vertices have +inf distance
    if (idx < _vertices_count)
        _distances[idx] = FLT_MAX;
    
    // sourse vertex has zero distance
    _distances[_source_vertex] = 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ init_kernel(double *_distances, int _vertices_count, int _source_vertex)
{
    register const int idx = (blockIdx.x * blockDim.x + threadIdx.x) + blockIdx.y * blockDim.x * gridDim.x;
    
    // not source vertices have +inf distance
    if (idx < _vertices_count)
        _distances[idx] = DBL_MAX;
    
    // sourse vertex has zero distance
    _distances[_source_vertex] = 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
void init_distances_wrapper(_TEdgeWeight *_distances, int _vertices_count, int _source_vertex)
{
    dim3 init_threads(MAX_THREADS);
    dim3 init_blocks((_vertices_count - 1) / init_threads.x + 1);
    
    #ifdef __USE_FERMI__
    if(init_blocks.x > 65535)
    {
        init_blocks.y = (init_blocks.x - 1) / 65535 + 1;
        init_blocks.x = 65535;
    }
    #endif
    
    // call init kernel
    SAFE_KERNEL_CALL((init_kernel <<< init_blocks, init_threads >>> (_distances, _vertices_count, _source_vertex)));
    
    cudaDeviceSynchronize();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// main computational algorithm
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
void __global__ bellman_ford_kernel(_TEdgeWeight *_distances,
                                    int *_src_ids,
                                    int *_dst_ids,
                                    _TEdgeWeight *_weights,
                                    int _vertices_count,
                                    long long _edges_count,
                                    int *_modif,
                                    int _iter)
{
    register const int idx = (blockIdx.x * blockDim.x + threadIdx.x) + blockIdx.y * blockDim.x * gridDim.x;
    
    if (idx < _edges_count) // for all graph edges do
    {
        register int src_id = _src_ids[idx];
        register int dst_id = _dst_ids[idx];
        register _TEdgeWeight weight = _weights[idx]; // get edge info
        
        #ifdef __USE_FERMI__
        register _TEdgeWeight src_distance = _distances[src_id]; // get current distance
        register _TEdgeWeight dst_distance = _distances[dst_id]; // to incident vertices
        #else
        register _TEdgeWeight src_distance = __ldg(&_distances[src_id]); // get current distance
        register _TEdgeWeight dst_distance = __ldg(&_distances[dst_id]); // to incident vertices
        #endif
        
        if (dst_distance > src_distance + weight) // if current edge offers better distance, update distanse array
        {
            _distances[dst_id] = src_distance + weight;
            _modif[0] = _iter + 1; // set that changes occured on current step
        }
        //if (src_distance > dst_distance + weight)
        //{
            //_distances[src_id] = dst_distance + weight;
            //_modif[0] = _iter + 1; // set that changes occured on current step
        //}
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TEdgeWeight>
void bellman_ford_wrapper(int *_src_ids, int *_dst_ids, _TEdgeWeight *_weights, int _vertices_count, long long _edges_count,
                          int _source_vertex, _TEdgeWeight *_distances)
{
    // device variable to stop iterations, for each source vertex
    int *device_modif;
    int host_modif;
    SAFE_CALL(cudaMalloc((void**)&device_modif, sizeof(int)));
    SAFE_CALL(cudaMemset(device_modif, 0, sizeof(int)));
    
    // set grid size
    dim3 compute_threads(MAX_THREADS);
    dim3 compute_blocks((_edges_count - 1) / compute_threads.x + 1);
    
    #ifdef __USE_FERMI__
    if(compute_blocks.x > 65535)
    {
        compute_blocks.y = (compute_blocks.x - 1) / 65535 + 1;
        compute_blocks.x = 65535;
    }
    #endif
    
    // compute shortest paths
    for (int cur_iteration = 0; cur_iteration < _vertices_count; cur_iteration++) // do o(|v|) iterations in worst case
    {
        // call main computaional algorithm to update distanses
        SAFE_KERNEL_CALL((bellman_ford_kernel<_TEdgeWeight> <<< compute_blocks, compute_threads >>>
                          (_distances, _src_ids, _dst_ids, _weights, _vertices_count, _edges_count, device_modif, cur_iteration)));
        
        // copy changes flag
        SAFE_CALL(cudaMemcpy(&host_modif, device_modif, sizeof(int), cudaMemcpyDeviceToHost));
        
        // check if no changes occured on current iteration
        if (host_modif == cur_iteration)
        {
            cout << "GPU iterations: " << cur_iteration << endl;
            break;
        }
    }
    
    SAFE_CALL(cudaFree(device_modif));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// templates to call wrappers outside .cu files
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template void init_distances_wrapper<float>(float *_distances, int _vertices_count, int _source_vertex);
template void init_distances_wrapper<double>(double *_distances, int _vertices_count, int _source_vertex);

template void bellman_ford_wrapper<float>(int *_src_ids, int *_dst_ids, float *_weights, int _vertices_count, long long _edges_count,
                                          int _source_vertex, float *_distances);
template void bellman_ford_wrapper<double>(int *_src_ids, int *_dst_ids, double *_weights, int _vertices_count, long long _edges_count,
                                           int _source_vertex, double *_distances);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
