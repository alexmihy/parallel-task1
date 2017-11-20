#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//! \class  GraphContainerData graph_container_data.h "src/data_structures/graph_container_data.h"
//! \brief  This class stores pointers to data in graph container, if it is located on GPU
//! \author Ilya Afanasyev
//!
//! \details Stores pointers to different data structures for all available containers. \n
//!          Some pointers, not suitable for current continer, store nullptr value      \n
template <typename _TVertexValue, typename _TEdgeWeight>
struct GraphContainerData
{
	int *vertices_ids;           //!< IDs of vertices. Used in COMPRESSED_ADJACENCY_LIST format
	int *edges_src_ids;          //!< IDs of source verices, adjacent to corresponding edges. Used in EDGES_LIST format
	int *edges_dst_ids;          //!< IDs of destination verices, adjacent to corresponding edges. Used in EDGES_LIST and COMPRESSED_ADJACENCY_LIST formats

	long long *vertices_to_edges_ptrs; //!< Index of first edge, corresponding to current vertex. Used in COMPRESSED_ADJACENCY_LIST format

	_TVertexValue *vertices_values; //!< Values of vertices. Used in COMPRESSED_ADJACENCY_LIST format.
	_TEdgeWeight  *edges_weights;   //!< Weights of edges. Used in Used in EDGES_LIST and COMPRESSED_ADJACENCY_LIST formats

	GraphContainerData();
	GraphContainerData(int *_vertices_ids, long long *_vertex_to_edge_ptrs, _TVertexValue *_vertices_values, int *_edges_src_ids,
		               int *_edges_dst_ids, _TEdgeWeight *_edges_weights);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
GraphContainerData<_TVertexValue, _TEdgeWeight>::GraphContainerData()
{
	vertices_ids           = NULL;
	vertices_to_edges_ptrs = NULL;
	vertices_values        = NULL;
	edges_src_ids          = NULL;
	edges_dst_ids          = NULL;
	edges_weights          = NULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
GraphContainerData<_TVertexValue, _TEdgeWeight>::GraphContainerData(int *_vertices_ids, long long *_vertices_to_edges_ptrs, 
	                                                                _TVertexValue *_vertices_values, int *_edges_src_ids,
													                int *_edges_dst_ids, _TEdgeWeight *_edges_weights)
{
	vertices_ids           = _vertices_ids;
	vertices_to_edges_ptrs = _vertices_to_edges_ptrs;
	vertices_values        = _vertices_values;
	edges_src_ids          = _edges_src_ids;
    edges_dst_ids          = _edges_dst_ids;
	edges_weights          = _edges_weights;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
