#pragma once

//! \cond DOXYGEN_SHOULD_SKIP_THIS

#include "include/graph_data_structures/graph.h"

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <boost/graph/bellman_ford_shortest_paths.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <boost/graph/strong_components.hpp>
#include <boost/graph/vf2_sub_graph_iso.hpp>

#include <typeinfo>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace boost;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class BoostAPI
{
private:
	typedef adjacency_list<vecS, vecS, undirectedS, property<vertex_distance_t, int>, property <edge_weight_t, _TEdgeWeight> > GraphBoostUndirected;
	typedef adjacency_list<vecS, vecS, directedS, property<vertex_distance_t, int>, property <edge_weight_t, _TEdgeWeight> > GraphBoostDirected;

	static void convert_to_boost_graph(Graph<_TVertexValue, _TEdgeWeight> &_input_graph, GraphBoostDirected **_output_graph);
	static void convert_to_boost_graph(Graph<_TVertexValue, _TEdgeWeight> &_input_graph, GraphBoostUndirected **_output_graph);
public:
	static void kruskal_minimum_spanning_tree(Graph<_TVertexValue, _TEdgeWeight> &_input_graph, Graph<_TVertexValue, _TEdgeWeight> &_output_MST);
	static double prim_minimum_spanning_tree(Graph<_TVertexValue, _TEdgeWeight> &_input_graph, Graph<_TVertexValue, _TEdgeWeight> &_output_MST);
    
	static void dijkstra_shortest_paths(Graph<_TVertexValue, _TEdgeWeight> &_input_graph, vector<int> _source_vertices, ShortestPathsSaveResult<_TEdgeWeight> *_save_obj);
	static void bellman_ford_shortest_paths(Graph<_TVertexValue, _TEdgeWeight> &_input_graph, vector<_TEdgeWeight> &_output_distance);
    
    static void tarjan_scc(Graph<_TVertexValue, _TEdgeWeight> &_input_graph, vector<int> &_scc_result);
    
    static void VF2(Graph<_TVertexValue, _TEdgeWeight> &_big_graph, Graph<_TVertexValue, _TEdgeWeight> &_small_graph);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "boost_API.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//! \endcond

