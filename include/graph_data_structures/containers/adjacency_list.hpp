/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
AdjacencyList<_TVertexValue, _TEdgeWeight>::AdjacencyList(int _vertices_count, bool _directed):
vertices(0)
{
	vertices_count   = 0;
	edges_count      = 0;
	BaseContainer<_TVertexValue, _TEdgeWeight>::directed = _directed;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
AdjacencyList<_TVertexValue, _TEdgeWeight>::~AdjacencyList()
{
	
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
Vertex<_TVertexValue> AdjacencyList<_TVertexValue, _TEdgeWeight>::iterate_vertices(int _vertex_pos)
{
	if ((unsigned int)_vertex_pos >= vertices.size())
		throw "out of range in get_vertex_by_pos";

	return vertices[_vertex_pos];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
Edge<_TEdgeWeight> AdjacencyList<_TVertexValue, _TEdgeWeight>::iterate_edges(long long _edge_pos)
{
	throw "not implemented yet";

	return Edge<_TEdgeWeight>(-1, -1, -1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
Edge<_TEdgeWeight> AdjacencyList<_TVertexValue, _TEdgeWeight>::iterate_adjacent_edges(int _src_vertex_id, int _edge_pos)
{
	return Edge<_TEdgeWeight>(_src_vertex_id, edges[_src_vertex_id][_edge_pos].adj_id, edges[_src_vertex_id][_edge_pos].weight);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
Vertex<_TVertexValue> AdjacencyList<_TVertexValue, _TEdgeWeight>::get_vertex_by_id(int _id)
{
	return iterate_vertices(_id);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
Edge<_TEdgeWeight> AdjacencyList<_TVertexValue, _TEdgeWeight>::get_edge_by_id(int _src_id, int _dst_id)
{
	if (!this->check_if_directed())
	{
		if (_src_id > _dst_id)
			swap(_src_id, _dst_id);
	}

	for (int edge_pos = 0; edge_pos < edges[_src_id].size(); edge_pos++)
	{
		if (edges[_src_id][edge_pos].adj_id == _dst_id)
			return Edge<_TEdgeWeight>(_src_id, _dst_id, edges[_src_id][edge_pos].weight);
	}
	return Edge<_TEdgeWeight>(-1, -1, -1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
int AdjacencyList<_TVertexValue, _TEdgeWeight>::get_vertices_count() const
{
	return vertices_count;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
long long AdjacencyList<_TVertexValue, _TEdgeWeight>::get_edges_count() const
{
	return edges_count;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
int AdjacencyList<_TVertexValue, _TEdgeWeight>::get_vertex_connections_count(int _id)
{
	/*if (!check_if_directed())
	{
		throw "get_vertex_connections_count not implemented yet for non-directed graphs";
	}*/

	return (int)edges[_id].size();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void AdjacencyList<_TVertexValue, _TEdgeWeight>::set_vertex(int _id, _TVertexValue _value)
{
    if(_id >= vertices.size())
        throw "out of range in SET_VERTEX function";
    
    vertices[_id].value = _value;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void AdjacencyList<_TVertexValue, _TEdgeWeight>::add_vertex(int _id, _TVertexValue _value)
{
	if (vertices.size() > 0 && (vertices[vertices.size() - 1].id + 1) != _id)
		throw "invalid vertex ID in add_vertex function";

	vertices.push_back(Vertex<_TVertexValue>(_id, _value));
	edges.push_back(vector<AdjListEdge<_TEdgeWeight>>(0));

	vertices_count++;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void AdjacencyList<_TVertexValue, _TEdgeWeight>::add_vertex(int _id, _TVertexValue _value, const vector<int> &_adj_ids,
															const vector<_TEdgeWeight> &_adj_weights)
{
	if (vertices.size() > 0 && (vertices[vertices.size() - 1].id + 1) != _id)
		throw "invalid vertex ID in add_vertex function";

	vertices.push_back(Vertex<_TVertexValue>(_id, _value));

	int adj_edges_count = (int)_adj_ids.size();

	vector<AdjListEdge<_TEdgeWeight>> tmp_edges;
	for (int edge_pos = 0; edge_pos < adj_edges_count; edge_pos++)
	{
		tmp_edges.push_back(AdjListEdge<_TEdgeWeight>(_adj_ids[edge_pos], _adj_weights[edge_pos]));
	}

	edges.push_back(tmp_edges);

	vertices_count++;
	edges_count += adj_edges_count;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void AdjacencyList<_TVertexValue, _TEdgeWeight>::add_edge(int _src_id, int _dst_id, _TEdgeWeight _weight)
{
	if (!this->check_if_directed())
	{
		if (_src_id > _dst_id)
			swap(_src_id, _dst_id);
	}

	edges[_src_id].push_back(AdjListEdge<_TEdgeWeight>(_dst_id, _weight));
	edges_count++;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
void AdjacencyList<_TVertexValue, _TEdgeWeight>::empty()
{
	vertices.clear();

	for (auto it = edges.begin(); it != edges.end(); it++)
		it->clear();

	edges.clear();

	vertices_count = 0;
	edges_count = 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
GraphContainerData<_TVertexValue, _TEdgeWeight> AdjacencyList<_TVertexValue, _TEdgeWeight>::get_container_data()
{
	throw "can not return data pointers in AdjacencyList format";
	return GraphContainerData<_TVertexValue, _TEdgeWeight>(NULL, NULL, NULL, NULL, NULL, NULL);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _TVertexValue, typename _TEdgeWeight>
AdjacencyList<_TVertexValue, _TEdgeWeight>* AdjacencyList<_TVertexValue, _TEdgeWeight>::clone() const
{
    int vertices_count    = this->get_vertices_count();
    long long edges_count = this->get_edges_count();
    
    AdjacencyList<_TVertexValue, _TEdgeWeight> *clone =
    new AdjacencyList<_TVertexValue, _TEdgeWeight>(vertices_count, this->directed);
    
    clone->vertices = this->vertices;
    clone->edges = this->edges;
    clone->vertices_count = this->vertices_count;
    clone->edges_count = this->edges_count;
    
    return clone;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
