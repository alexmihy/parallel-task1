#pragma once

#include "include/graph_data_structures/graph.h"
#include "include/common_data_structures/containers.h"

#include "gpu/tarjan.cuh"

#include <stack>
#include <map>
#include <queue>
#include <list>

using namespace std;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class BridgesDetection
{
private:
    int DFS_counter;
    
    void DFS_pre_order(vector <int> &_vertices_to_edges_ptrs, vector <int> &_dst_ids, int _root,
                       vector<int> &_pre_order_numbers, vector<bool> &_DFS_visited);
    
    void cpu_parallel_bfs(Graph<_TVertexValue, _TEdgeWeight> &_graph, int _root, vector<int> &_bfs_level,
                          bool *_in_tree, int &_max_level, int _omp_threads_count);
    
    void cpu_sequential_bfs(Graph<_TVertexValue, _TEdgeWeight> &_graph, int _root, vector<int> &_bfs_level,
                            bool *_in_tree, int &_max_level);
    
public:
    void parallel_cpu_tarjan(Graph<_TVertexValue, _TEdgeWeight> &_graph, bool *_bridges, int _omp_treads_count);
    
    #ifdef __USE_GPU__
    void parallel_gpu_tarjan(Graph<_TVertexValue, _TEdgeWeight> &_graph, bool *_bridges);
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "bridges.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
