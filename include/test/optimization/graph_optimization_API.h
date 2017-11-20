#pragma once

#include "include/graph_data_structures/graph.h"
#include "include/common/random_numbers.h"
#include "include/node_data/node_data.h"

#include <algorithm>
#include <map>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum GPU_ARCHITECTURE
{
    KEPLER,
    PASCAL
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class GraphOptimizationAPI
{
private:
    static int get_segment_num(int _id, int _cache_line_size);
    
    // common for all architectures optimizations
    static void remove_loops_and_multiple_arcs(Graph<_TVertexValue, _TEdgeWeight> &_input_graph);
    static void reorder_edges_in_adjacency_list_format(Graph<_TVertexValue, _TEdgeWeight> &_input_graph);
    
    // cache optimization for cache
    static void reorder_edges_for_cache(Graph<_TVertexValue, _TEdgeWeight> &_input_graph, int _cache_size, int elem_size);
public:
    // perform all optimizations for CPU
    static void optimize_graph_for_CPU(Graph<_TVertexValue, _TEdgeWeight> &_input_graph, int elem_size);
    
    // perform all optimizations for GPU
    static void optimize_graph_for_GPU(Graph<_TVertexValue, _TEdgeWeight> &_input_graph, int elem_size, GPU_ARCHITECTURE _arch);
    
    // perform all optimizations for Intel Knights Landing
    static void optimize_graph_for_KNL(Graph<_TVertexValue, _TEdgeWeight> &_input_graph, int elem_size);
    
    // calculate transactions count to GPU memory with specifed graph
    static long long check_gpu_memory_transactions_count(Graph<_TVertexValue, _TEdgeWeight> &_input_graph, int elem_size);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_optimization_API.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
