#pragma once

#include <vector>
#include <stdlib.h>
#include <stdio.h>

#ifdef __USE_KNL__
#include <hbwmalloc.h>
#endif

#ifdef __USE_GPU__
#include "gpu/bellman_ford.cuh"
#endif

#include <map>

#include "include/graph_data_structures/graph.h"

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class SingleSourceShortestPaths
{
private:
    int omp_threads;
public:
    SingleSourceShortestPaths(int _omp_threads = 1) { omp_threads = _omp_threads; }
    void set_omp_threads(int _omp_threads) { omp_threads = _omp_threads; }
    
    void cpu_bellman_ford(Graph<_TVertexValue, _TEdgeWeight> &_graph, _TEdgeWeight *_distances, int _source_vertex);
    
    #ifdef __USE_KNL__
    void knl_bellman_ford(Graph<_TVertexValue, _TEdgeWeight> &_graph, _TEdgeWeight *_distances, int _source_vertex);
    #endif
    
    #ifdef __USE_GPU__
    double gpu_bellman_ford(Graph<_TVertexValue, _TEdgeWeight> &_graph, _TEdgeWeight *_distances, int _source_vertex);
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "sssp.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
