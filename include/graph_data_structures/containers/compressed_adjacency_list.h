#pragma once

#include "base_container.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//! \cond DOXYGEN_SHOULD_SKIP_THIS

template <typename _TVertexValue>
struct VerticesInCompressedAdjacencyList
{
	int *ids;
	long long *edge_ptrs;
	_TVertexValue *values;
	int top, size;
};

template <typename _TEdgeWeight>
struct EdgesInCompressedAdjacencyList
{
	int *adj_ids;
	_TEdgeWeight *weights;
	long long top, size;
};

//! \endcond

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class CompressedAdjacencyList : public BaseContainer<_TVertexValue, _TEdgeWeight>
{
private:
	VerticesInCompressedAdjacencyList<_TVertexValue> vertices;
	EdgesInCompressedAdjacencyList<_TEdgeWeight> edges;
public:
	CompressedAdjacencyList(int _vertices_count, long long _edges_count, bool _directed);
	~CompressedAdjacencyList();

	Vertex<_TVertexValue> iterate_vertices      (int _vertex_pos);
	Edge<_TEdgeWeight>    iterate_edges         (long long _edge_pos);
	Edge<_TEdgeWeight>    iterate_adjacent_edges(int _src_vertex_id, int _edge_pos);

	Vertex<_TVertexValue> get_vertex_by_id(int _id);
	Edge<_TEdgeWeight>    get_edge_by_id  (int _src_vertex_id, int _dst_vertex_id);
	int get_vertices_count() const;
	long long get_edges_count() const;
	int get_vertex_connections_count(int _id);
    
    void set_vertex(int _id, _TVertexValue _value);

	void add_vertex(int _id, _TVertexValue _value);
	void add_vertex(int _id, _TVertexValue _value, const vector<int> &_adj_ids, const vector<_TEdgeWeight> &_adj_weights);

	void add_edge(int _src_id, int _dst_id, _TEdgeWeight _weight);

	void empty();

	GraphContainerData<_TVertexValue, _TEdgeWeight> get_container_data();

	CompressedAdjacencyList<_TVertexValue, _TEdgeWeight>* clone() const;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "compressed_adjacency_list.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
