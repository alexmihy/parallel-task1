#pragma once

#include "base_container.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//! \cond DOXYGEN_SHOULD_SKIP_THIS

template <typename _TEdgeWeight>
struct AdjListEdge
{
	int adj_id;
	_TEdgeWeight weight;

	AdjListEdge(int _adj_id = 0, _TEdgeWeight _weight = 0) { adj_id = _adj_id; weight = _weight; };
};

//! \endcond

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//! \class AdjacencyList adjacency_list.h "src/data_structures/containers/adjacency_list.h"
//!
//! \brief This class is storage container of data in adjacency list format
//!
//! \author Ilya Afanasyev
//!
//! \details This container corresponds to ADJACENCY_LIST type. Graph is represented as 2 vectors: \n
//! 1. vector of Vertex type, stores information about vertices
//! 2. vector of AdjListEdge type, stores information about edges
template <typename _TVertexValue, typename _TEdgeWeight>
class AdjacencyList : public BaseContainer<_TVertexValue, _TEdgeWeight>
{
private:
	vector <Vertex<_TVertexValue>> vertices;
	vector <vector<AdjListEdge<_TEdgeWeight>>> edges;

	int vertices_count;
	long long edges_count;
public:
	AdjacencyList(int _vertices_count, bool _directed);
	~AdjacencyList();

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

	AdjacencyList<_TVertexValue, _TEdgeWeight>* clone() const;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "adjacency_list.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
