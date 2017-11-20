#pragma once

#include "include/graph_data_structures/graph.h"
#include "include/node_data/node_data.h"
#include "include/common/computational_mode.h"

#ifdef __USE_GPU__
#include "algorithm/scc/gpu/forward_backward.cuh"
#endif

#include <stack>
#include <map>
#include <queue>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define ERROR_IN_PIVOT -10
#define INIT_COMPONENT 1
#define INIT_TREE 1

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class StronglyConnectedComponents
{
private:
    // common graph data
    long long edges_count;
    int vertices_count;
    long long *vertices_to_edges_ptrs;
    int *src_ids, *dst_ids;
    
    // pointers
    NodeData *node_data_ptr; //!< pointer node data object
    
    // extra data
    int partitions_count;
    long long max_edges_in_gpu_partition;
    
    // omp threads count
    int omp_threads;
    
    // internal FB functions
    int select_pivot(int *_trees, int _tree_num);
    void trim(int *_components, int *_trees, bool *_active, int *_last_component);
    void bfs_reach(int *_src_ids, int *_dst_ids, int _pivot, bool *_visited, int *_trees);
    void bfs_kernel(int *_src_ids, int *_dst_ids, bool *_visited, bool &_terminate, int *_trees, int _pivot);
    void process_result(bool *_fwd_result, bool *_bwd_result, int *_components, int *_trees, bool *_active,
                        int *_last_component, int *_last_tree);
    void FB_on_host(int *_components, int *_trees, int _tree_num, bool *_active, int *_last_component);
    
    // internal tarjan functions
    void tarjan_kernel(int _u, int *_disc, int *_low, stack<int> &_st, bool *_stack_member, int *_components);
public:
    StronglyConnectedComponents(int _omp_threads = 1) { omp_threads = _omp_threads; };
    
    // Forward backward algorithm for SCC computational on cpu in parallel
    void cpu_forward_backward(Graph<_TVertexValue, _TEdgeWeight> &_graph, int *_components);
    
    // Tarjan algorithm for SCC computational (sequential CPU mode)
    void cpu_tarjan(Graph<_TVertexValue, _TEdgeWeight> &_graph, int *_components);
    
    #ifdef __USE_GPU__
    // internal Forward-Backward functions
    void gpu_forward_backward(Graph<_TVertexValue, _TEdgeWeight> &_graph, int *_components);
    #endif
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "scc.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
