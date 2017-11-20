#pragma once

#include "include/graph_data_structures/graph.h"
#include "include/node_data/node_data.h"
#include "include/common/computational_mode.h"
#include "algorithm/scc/gpu/forward_backward.cuh"

#include <stack>
#include <map>
#include <queue>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class TransitiveClosure
{
private:
    int omp_threads;
    
    int *device_src_ids, *device_dst_ids;
    bool *device_result;
   
    // gpu bfs functions (with optimized data copies)
    void allocate_and_copy_device_arrays(Graph<_TVertexValue, _TEdgeWeight> &_graph);
    void gpu_bfs(Graph<_TVertexValue, _TEdgeWeight> &_graph, bool *_result, int _source_vertex);
    void free_device_arrays();
    
    void edges_list_bfs(int *_src_ids, int *_dst_ids, bool *_distances, int _source, int _vertices_count, long long _edges_count);
    void adj_list_bfs(vector<vector<int>> &_adj_ids, bool *_result, int _source_vertex, int _vertices_count);
public:
    TransitiveClosure(int _omp_threads = 1) {omp_threads = _omp_threads; };
    
    // new cpu variation of the algorithm
    double cpu_purdom(Graph<_TVertexValue, _TEdgeWeight> &_input_graph, vector<pair<int, int>> _pairs_to_check, vector<bool> &_answer);
    
    // in adj list
    void cpu_purdom2(Graph<_TVertexValue, _TEdgeWeight> &_input_graph, vector<pair<int, int>> _pairs_to_check, vector<bool> &_answer);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "transitive_closure.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
