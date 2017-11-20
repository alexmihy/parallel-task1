#pragma once

#include "include/graph_data_structures/graph.h"

#ifdef __USE_GPU__
#include "include/common/cuda_error_hadling.h"
#endif

#include <queue>
#include "gpu/bfs.cuh"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class BFS
{
private:
    int omp_threads;
public:
    BFS(int _omp_threads = 1) { omp_threads = _omp_threads; }
    void set_omp_threads(int _omp_threads) { omp_threads = _omp_threads; }
    
    // anjacnecy list
    void cpu_sequential_bfs(Graph<_TVertexValue, _TEdgeWeight> &_graph, bool *_result, int _source_vertex);
    
    // edges list
    void cpu_parallel_bfs(Graph<_TVertexValue, _TEdgeWeight> &_graph, bool *_result, int _source_vertex);
    
    // edges list
    void nec_sx_parallel_bfs(Graph<_TVertexValue, _TEdgeWeight> &_graph, bool *_result, int _source_vertex);
    
    #ifdef __USE_GPU__
    void gpu_bfs(Graph<_TVertexValue, _TEdgeWeight> &_graph, bool *_result, int _source_vertex);
    #endif
    
    // edges list
    #ifdef __USE_KNL__
    void knl_bfs(Graph<_TVertexValue, _TEdgeWeight> &_graph, bool *_result, int _source_vertex);
    #endif
    
    // try new apprach
    void new_parallel_bfs(Graph<_TVertexValue, _TEdgeWeight> &_graph, bool *_result, vector<int> &_source_vertices);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "bfs.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
