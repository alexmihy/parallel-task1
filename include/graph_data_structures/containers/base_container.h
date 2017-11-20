#pragma once

#include "include/common/memory.h"
#include "../vertex.h"
#include "../edge.h"
#include "../graph_container_data.h"

#include <vector>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
class BaseContainer
{
protected:
	bool directed;
public:
	BaseContainer() {};
	virtual ~BaseContainer() {};

	virtual Vertex<_TVertexValue> iterate_vertices      (int _pos) = 0;
	virtual Edge<_TEdgeWeight>    iterate_edges         (long long _edge_pos) = 0;
	virtual Edge<_TEdgeWeight>    iterate_adjacent_edges(int _src_vertex_id, int _edge_pos) = 0;

	virtual Vertex<_TVertexValue> get_vertex_by_id(int _id) = 0;
	virtual Edge<_TEdgeWeight>    get_edge_by_id  (int _src_vertex_id, int _dst_vertex_id) = 0;
	virtual int get_vertices_count() const = 0;
	virtual long long get_edges_count() const = 0;
	virtual int get_vertex_connections_count(int _id) = 0;
    
    virtual void set_vertex(int _id, _TVertexValue _value) = 0;

	virtual void add_vertex(int _id, _TVertexValue _value) = 0;
	virtual void add_vertex(int _id, _TVertexValue _value, const vector<int> &_adj_ids, const vector<_TEdgeWeight> &_adj_weights) = 0;

	virtual void add_edge(int _src_id, int _dst_id, _TEdgeWeight _weight) = 0;

	virtual void empty() = 0;

	virtual GraphContainerData<_TVertexValue, _TEdgeWeight> get_container_data() = 0;

	virtual BaseContainer<_TVertexValue, _TEdgeWeight>* clone() const = 0;

	bool check_if_directed() { return directed; }
	void set_directed(bool _directed) { directed = _directed; }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
