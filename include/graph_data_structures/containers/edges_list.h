#pragma once

#include "base_container.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//! \cond DOXYGEN_SHOULD_SKIP_THIS

template <typename _TVertexValue>
struct VerticesInEdgesList
{
	_TVertexValue *values;
	int top, size;
};

template <typename _TEdgeWeight>
struct EdgesInEdgesList
{
	int *src_ids;
	int *dst_ids;
	_TEdgeWeight *weights;
	long long top, size;
};

//! \endcond

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//! \class EdgesList edges_list.h "src/data_structures/containers/edges_list.h"
//!
//! \brief This class is storage container of data in edges list format
//!
//! \author Ilya Afanasyev
//!
//! \details This container corresponds to EDGES_LIST type. \n
template <typename _TVertexValue, typename _TEdgeWeight>
class EdgesList : public BaseContainer<_TVertexValue, _TEdgeWeight>
{
private:
	VerticesInEdgesList<_TVertexValue> vertices;
	EdgesInEdgesList<_TEdgeWeight> edges;

	int vertices_count;
public:
	EdgesList(int _vertices_count, long long _edges_count, bool _directed, bool _empty);
	~EdgesList();

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

	EdgesList<_TVertexValue, _TEdgeWeight>* clone() const;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "edges_list.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
