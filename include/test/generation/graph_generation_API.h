#pragma once

#include "include/graph_data_structures/graph.h"
#include "include/common/random_numbers.h"
#include "include/node_data/node_data.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class GraphGenerationAPI
{
public:
	static void random_uniform(Graph<_TVertexValue, _TEdgeWeight> &_graph, int _vertices_count, int _average_degree,
                               bool _directed, bool _optimized);

	static void R_MAT(Graph<_TVertexValue, _TEdgeWeight> &_graph, int _vertices_count, int _average_degree,
                      int _a_prob, int _b_prob, int _c_prob, int _d_prob, bool _directed, bool _optimized);
    
    static void R_MAT_parallel(Graph<_TVertexValue, _TEdgeWeight> &_graph, int _vertices_count,
                               int _average_connections, int _a_prob, int _b_prob, int _c_prob,
                               int _d_prob, int _omp_threads, bool _directed);

	static void SSCA2(Graph<_TVertexValue, _TEdgeWeight> &_graph, int _vertices_count, int _max_clique_size, bool _directed);

	static void convert_into_single_SCC(Graph<_TVertexValue, _TEdgeWeight> &_graph, bool _directed, NodeData &_node_data);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_generation_API.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
